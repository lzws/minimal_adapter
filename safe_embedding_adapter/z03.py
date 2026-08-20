"""冻结的 Z-03 latent 分类头反馈。

训练时直接调用模型 forward，保留 latent 到分类 loss 的梯度；分类器参数全部
冻结。支持仓库原有的 legacy checkpoint，也支持 export 目录的
``MultiTaskWrapperCNN`` checkpoint。
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_EXPORT_MODEL_FILE = Path(__file__).resolve().parents[1] / "latent_detector_model.py"

IP_CONDITION_TO_CLASS_ID = {
    "ip_snow_white": 0,
    "ip_doraemon": 1,
    "ip_minion": 2,
    "ip_elsa": 3,
    "ip_spongebob": 4,
}
IP_OTHER_CLASS_ID = 5


@dataclass
class Z03Output:
    """Z-03 三个分类头的 logits。"""

    porn_logits: torch.Tensor
    gore_logits: torch.Tensor
    ip_logits: torch.Tensor

    def risk_logits(self, target_risk: str) -> torch.Tensor:
        if target_risk == "porn":
            return self.porn_logits
        if target_risk == "gore":
            return self.gore_logits
        if target_risk == "ip":
            return self.ip_logits
        raise ValueError(f"target_risk 必须是 porn、gore 或 ip，实际为 {target_risk!r}")

    def safe_class_index(self, target_risk: str) -> int:
        if target_risk in {"porn", "gore"}:
            return 0
        if target_risk == "ip":
            return IP_OTHER_CLASS_ID
        raise ValueError(f"target_risk 必须是 porn、gore 或 ip，实际为 {target_risk!r}")

    def risk_prob(self, target_risk: str) -> torch.Tensor:
        """返回风险概率；IP 为五个已知 IP 类概率之和。"""

        logits = self.risk_logits(target_risk)
        probs = F.softmax(logits.float(), dim=-1)
        safe_index = self.safe_class_index(target_risk)
        risk_indices = [index for index in range(logits.shape[-1]) if index != safe_index]
        index_tensor = torch.tensor(risk_indices, device=logits.device, dtype=torch.long)
        return probs.index_select(-1, index_tensor).sum(-1)

    def risk_safe_margin(self, target_risk: str) -> torch.Tensor:
        """返回 safe logit - aggregate risk logit，越大越安全。"""

        logits = self.risk_logits(target_risk)
        safe_index = self.safe_class_index(target_risk)
        risk_indices = [index for index in range(logits.shape[-1]) if index != safe_index]
        index_tensor = torch.tensor(risk_indices, device=logits.device, dtype=torch.long)
        risk_logits = logits.index_select(-1, index_tensor)
        risk_score = risk_logits[:, 0] if risk_logits.shape[-1] == 1 else torch.logsumexp(risk_logits, dim=-1)
        return logits[:, safe_index] - risk_score


def resize_latents_for_z03(latents: torch.Tensor) -> torch.Tensor:
    """将 Z-Image latent 变为 Z-03 所需的 ``[B, 16, 224, 224]``。"""

    if latents.ndim != 4 or latents.shape[1] != 16:
        raise ValueError(f"Z-03 输入必须是 [B, 16, H, W]，实际 shape={tuple(latents.shape)}")
    latents = latents.float()
    if latents.shape[-2:] != (224, 224):
        latents = F.interpolate(latents, size=(224, 224), mode="bilinear", align_corners=False)
    return latents


def safe_classification_loss(
    risk_logits: torch.Tensor,
    *,
    safe_class_index: int,
    loss_type: str,
    margin: float,
    target_class_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """把分类头输出推向 safe 类。

    对 porn/gore，safe 类是 0；对 IP，safe/other 类是 5。IP 还可以通过
    ``target_class_indices`` 只压低当前样本对应的 IP 类。
    """

    if risk_logits.ndim != 2 or risk_logits.shape[-1] < 2:
        raise ValueError(f"risk_logits 期望 [B, C]，实际 {tuple(risk_logits.shape)}")
    if loss_type not in {"ce", "softplus_margin", "hinge"}:
        raise ValueError(f"未知 z03 loss_type={loss_type!r}")
    num_classes = int(risk_logits.shape[-1])
    if not 0 <= safe_class_index < num_classes:
        raise ValueError(f"safe_class_index={safe_class_index} 超出类别数 {num_classes}")

    safe_score = risk_logits[:, safe_class_index]
    unsafe_indices = [index for index in range(num_classes) if index != safe_class_index]
    unsafe_index_tensor = torch.tensor(unsafe_indices, device=risk_logits.device, dtype=torch.long)
    aggregate_unsafe = risk_logits.index_select(-1, unsafe_index_tensor)
    aggregate_unsafe = (
        aggregate_unsafe[:, 0]
        if aggregate_unsafe.shape[-1] == 1
        else torch.logsumexp(aggregate_unsafe, dim=-1)
    )

    unsafe_score = aggregate_unsafe
    if target_class_indices is not None:
        target_class_indices = target_class_indices.to(device=risk_logits.device, dtype=torch.long)
        if target_class_indices.ndim != 1 or target_class_indices.shape[0] != risk_logits.shape[0]:
            raise ValueError("target_class_indices 必须是 shape [B] 的 tensor")
        valid = target_class_indices >= 0
        if bool(valid.any()):
            selected = target_class_indices[valid]
            if bool((selected >= num_classes).any()) or bool((selected == safe_class_index).any()):
                raise ValueError("target_class_indices 不能超出类别范围或指向 safe 类")
            row_indices = torch.arange(risk_logits.shape[0], device=risk_logits.device)[valid]
            unsafe_score = aggregate_unsafe.clone()
            unsafe_score[valid] = risk_logits[row_indices, selected]

    if loss_type == "ce":
        if target_class_indices is not None and bool((target_class_indices >= 0).any()):
            binary_logits = torch.stack([safe_score, unsafe_score], dim=-1)
            target_safe = torch.zeros(risk_logits.shape[0], dtype=torch.long, device=risk_logits.device)
            return F.cross_entropy(binary_logits, target_safe)
        target = torch.full(
            (risk_logits.shape[0],), safe_class_index, dtype=torch.long, device=risk_logits.device
        )
        return F.cross_entropy(risk_logits, target)

    margin_value = unsafe_score - safe_score + float(margin)
    if loss_type == "softplus_margin":
        return F.softplus(margin_value).mean()
    return F.relu(margin_value).pow(2).mean()


class Z03Scorer(nn.Module):
    """冻结 Z-03 模型，但保留输入 latent 的梯度。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str,
        *,
        model_type: str = "auto",
        model_file: str | Path | None = None,
    ):
        super().__init__()
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.device = torch.device(device)
        self.model_type = self._resolve_model_type(model_type, self.checkpoint_path)
        self.model_file = str(model_file) if model_file else None

        if self.model_type == "legacy":
            try:
                from latent_vit.infer_z03_latent import build_z03_model, load_checkpoint
            except ImportError as exc:
                raise ImportError(
                    "legacy Z-03 需要仓库中的 latent_vit.infer_z03_latent；"
                    "或者改用 --z03_model_type export。"
                ) from exc
            model = load_checkpoint(build_z03_model(), self.checkpoint_path)
        elif self.model_type == "export":
            model = self._load_export_model(self.checkpoint_path, self.model_file)
        else:
            raise ValueError(f"未知 z03_model_type={self.model_type!r}")

        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @staticmethod
    def _resolve_model_type(model_type: str, checkpoint_path: str) -> str:
        if model_type not in {"auto", "legacy", "export"}:
            raise ValueError("z03_model_type 必须是 auto、legacy 或 export")
        if model_type != "auto":
            return model_type
        checkpoint = Path(checkpoint_path).resolve()
        return "export" if (checkpoint.parent / "config.json").is_file() else "legacy"

    @staticmethod
    def _load_export_module(model_file: str | Path):
        model_file = Path(model_file).resolve() if model_file else DEFAULT_EXPORT_MODEL_FILE
        if not model_file.is_file():
            raise FileNotFoundError(f"export Z-03 model.py 不存在: {model_file}")
        spec = importlib.util.spec_from_file_location("_minimal_adapter_z03_model", model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 export Z-03 模型定义: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "MultiTaskWrapperCNN"):
            raise AttributeError(f"{model_file} 没有定义 MultiTaskWrapperCNN")
        return module

    @classmethod
    def _load_export_model(cls, checkpoint_path: str, model_file: str | Path | None) -> nn.Module:
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Z-03 checkpoint 不存在: {checkpoint}")
        config_path = checkpoint.parent / "config.json"
        config = {}
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
        module = cls._load_export_module(model_file)
        model = module.MultiTaskWrapperCNN(
            backbone_name=config.get("backbone", "convnext_base"),
            in_chans=int(config.get("latent_in_chans", 16)),
            pretrained=False,
            dropout=0.0,
            stem_type=config.get("latent_stem_type", "single"),
            stem_init=config.get("latent_stem_init", "expand_in1k"),
            num_porn_classes=int(config.get("num_porn_classes", 2)),
            num_gore_classes=int(config.get("num_gore_classes", 2)),
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"export Z-03 checkpoint 必须是 state_dict，实际为 {type(state_dict)}")
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model_keys = set(model.state_dict())
        if not model_keys.intersection(state_dict):
            stripped = {
                (key[5:] if key.startswith("base.") else key): value
                for key, value in state_dict.items()
            }
            if model_keys.intersection(stripped):
                state_dict = stripped
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "export Z-03 checkpoint 与模型结构不匹配："
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        return model

    def forward(self, latents: torch.Tensor) -> Z03Output:
        porn_logits, gore_logits, ip_logits = self.model(resize_latents_for_z03(latents).to(self.device))
        return Z03Output(porn_logits, gore_logits, ip_logits)


# 与旧代码中的命名保持兼容，便于迁移少量调用代码。
Z03PornScorer = Z03Scorer
