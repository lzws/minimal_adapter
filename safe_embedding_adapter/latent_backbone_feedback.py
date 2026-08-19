"""Latent backbone feature similarity feedback.

This module mirrors the DINO cosine feedback path, but it works directly on
Z-Image latent_x1 tensors. The backbone is the headless latent detector export
under latent_vit/latent_detector_export, so no VAE decode or image encoder is
needed during training.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATENT_BACKBONE_CHECKPOINT = (
    REPO_ROOT
    .parent
    / "backbone_iter100000.pth"
)
DEFAULT_LATENT_BACKBONE_CONFIG = REPO_ROOT.parent / "backbone_config.json"
DEFAULT_LATENT_BACKBONE_MODEL_FILE = REPO_ROOT.parent / "latent_detector_model.py"
DEFAULT_LATENT_REFERENCE_PROMPTS = {
    "Snow White": "Snow White from Snow White cartoon character",
    "Doraemon": "Doraemon from Doraemon cartoon character",
    "Minions": "Minions from Despicable Me cartoon character",
    "Elsa": "Queen Elsa from Frozen cartoon character",
    "SpongeBob SquarePants": "SpongeBob SquarePants cartoon character",
}


def load_export_model_module(model_file: str | Path = DEFAULT_LATENT_BACKBONE_MODEL_FILE):
    """Load the self-contained latent detector export model.py."""

    model_file = Path(model_file).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"latent detector export model.py 不存在: {model_file}")

    module_name = "_mysld_latent_detector_backbone_model"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 latent detector export 模型定义: {model_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "_build_cnn_for_latent"):
        raise AttributeError(f"{model_file} 没有定义 _build_cnn_for_latent")
    return module


def read_backbone_config(config_path: str | Path | None) -> dict[str, Any]:
    """Read export config.json with conservative defaults."""

    path = Path(config_path or DEFAULT_LATENT_BACKBONE_CONFIG)
    config: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    config.setdefault("backbone", "convnext_base")
    config.setdefault("latent_in_chans", 16)
    config.setdefault("latent_stem_type", "single")
    config.setdefault("latent_stem_init", "trunc_normal")
    config.setdefault("latent_stretch_target_hw", [224, 224])
    return config


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"latent backbone checkpoint 应为 state_dict，实际类型为 {type(state_dict)}")
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }


def _strip_prefix_if_present(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor] | None:
    stripped = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    return stripped or None


def load_backbone_state_dict(backbone: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a headless latent detector backbone checkpoint strictly."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"latent backbone checkpoint 不存在: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(payload)
    model_keys = set(backbone.state_dict().keys())

    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [("raw", dict(state_dict))]
    for prefix in ("backbone.", "base.backbone.", "model.backbone."):
        stripped = _strip_prefix_if_present(state_dict, prefix)
        if stripped is not None:
            candidates.append((prefix, stripped))

    best_name = ""
    best_state: dict[str, torch.Tensor] | None = None
    best_overlap = -1
    for name, candidate in candidates:
        overlap = len(set(candidate.keys()) & model_keys)
        if overlap > best_overlap:
            best_name = name
            best_state = candidate
            best_overlap = overlap

    if best_state is None or best_overlap <= 0:
        raise RuntimeError(
            "latent backbone checkpoint 与模型 key 没有交集，"
            f"checkpoint={checkpoint_path}"
        )

    missing, unexpected = backbone.load_state_dict(best_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "latent backbone checkpoint 与模型结构不匹配："
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_prefix": best_name,
        "num_keys": len(best_state),
    }


class LatentBackboneFeatureEncoder(nn.Module):
    """Frozen latent detector backbone used as a differentiable feature encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_LATENT_BACKBONE_CHECKPOINT,
        *,
        config_path: str | Path | None = None,
        model_file: str | Path = DEFAULT_LATENT_BACKBONE_MODEL_FILE,
        device: torch.device | str,
        torch_dtype: torch.dtype = torch.float32,
        input_size: int | tuple[int, int] | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.config = read_backbone_config(config_path or Path(checkpoint_path).parent / "config.json")
        export_module = load_export_model_module(model_file)
        self.backbone, self.feature_dim = export_module._build_cnn_for_latent(
            self.config.get("backbone", "convnext_base"),
            int(self.config.get("latent_in_chans", 16)),
            False,
            stem_type=self.config.get("latent_stem_type", "single"),
            stem_init=self.config.get("latent_stem_init", "trunc_normal"),
        )
        self.load_metadata = load_backbone_state_dict(self.backbone, checkpoint_path)
        self.backbone.to(device=self.device, dtype=torch_dtype)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        if input_size is None:
            target_hw = self.config.get("latent_stretch_target_hw", [224, 224])
            input_size = (int(target_hw[0]), int(target_hw[1]))
        elif isinstance(input_size, int):
            input_size = (int(input_size), int(input_size))
        self.input_size = tuple(int(value) for value in input_size)
        self.model_dtype = torch_dtype

    def preprocess(self, latent_x1: torch.Tensor) -> torch.Tensor:
        if latent_x1.ndim != 4:
            raise ValueError(f"latent backbone 输入必须是 [B,C,H,W]，实际 shape={tuple(latent_x1.shape)}")
        expected_chans = int(self.config.get("latent_in_chans", 16))
        if latent_x1.shape[1] != expected_chans:
            raise ValueError(
                f"latent backbone 期望 {expected_chans} 通道，实际通道数={latent_x1.shape[1]}"
            )
        latents = latent_x1.float()
        if latents.shape[-2:] != self.input_size:
            latents = F.interpolate(latents, size=self.input_size, mode="bilinear", align_corners=False)
        return latents.to(device=self.device, dtype=self.model_dtype)

    def encode_latents(self, latent_x1: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized backbone features while preserving input gradients."""

        features = self.backbone(self.preprocess(latent_x1)).float()
        return F.normalize(features, dim=-1)

    def forward(self, latent_x1: torch.Tensor) -> torch.Tensor:
        return self.encode_latents(latent_x1)


def _select_reference_features(
    payload: Any,
    *,
    target_key: str | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if isinstance(payload, torch.Tensor):
        return payload, metadata
    if not isinstance(payload, dict):
        raise TypeError(f"不支持的 latent reference feature 类型: {type(payload)}")

    metadata = {key: value for key, value in payload.items() if key not in {"features", "prototype", "reference_features"}}
    if "prototype" in payload:
        return payload["prototype"], metadata
    if "features" in payload:
        return payload["features"], metadata
    if "reference_features" not in payload:
        raise ValueError("latent reference 文件缺少 prototype/features/reference_features 字段")

    banks = payload["reference_features"]
    if not isinstance(banks, dict) or not banks:
        raise ValueError("reference_features 必须是非空 dict")
    if not target_key:
        raise ValueError("reference_features bank 需要指定 target_key")

    normalized_key = str(target_key).strip().lower()
    selected_key = None
    for key in banks:
        if str(key).strip().lower() == normalized_key:
            selected_key = key
            break
    if selected_key is None:
        available = ", ".join(sorted(str(key) for key in banks))
        raise KeyError(f"reference_features 中没有 {target_key!r}；可选: {available}")

    metadata["reference_key"] = selected_key
    return banks[selected_key], metadata


def load_latent_reference_payload(
    path: str | Path,
    *,
    target_key: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a latent reference prototype or feature bank."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"latent reference feature 文件不存在: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    features, metadata = _select_reference_features(payload, target_key=target_key)
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"{path} 中选中的 reference feature 不是 torch.Tensor")
    if features.ndim == 1:
        prototype = features.float()
        num_features = 1
    elif features.ndim == 2:
        if features.shape[0] <= 0 or features.shape[1] <= 0:
            raise ValueError(f"{path} 的 feature shape 非法: {tuple(features.shape)}")
        prototype = features.float().mean(dim=0)
        num_features = int(features.shape[0])
    else:
        raise ValueError(f"latent reference feature 期望 [D] 或 [K,D]，实际 shape={tuple(features.shape)}")
    prototype = F.normalize(prototype.flatten(), dim=0)
    if not bool(torch.isfinite(prototype).all()):
        raise ValueError(f"{path} 的 prototype 含有 NaN/Inf")
    metadata["source_path"] = str(path)
    metadata["num_reference_features"] = num_features
    metadata["feature_dim"] = int(prototype.shape[0])
    return prototype, metadata


def load_latent_reference_prototypes(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and normalize all prototypes from a multi-IP reference bank."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"latent reference feature 文件不存在: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 中的 latent reference payload 必须是 dict")

    prototypes = payload.get("prototypes")
    if prototypes is None:
        reference_features = payload.get("reference_features")
        if not isinstance(reference_features, dict):
            raise ValueError(f"{path} 缺少 prototypes/reference_features 字段")
        prototypes = {}
        for key, features in reference_features.items():
            if not isinstance(features, torch.Tensor) or features.ndim not in (1, 2):
                raise ValueError(f"{path} 中 {key!r} 的 reference feature shape 非法")
            prototype = features.float() if features.ndim == 1 else features.float().mean(dim=0)
            prototypes[key] = prototype

    if not isinstance(prototypes, dict) or not prototypes:
        raise ValueError(f"{path} 中的 prototypes 必须是非空 dict")

    normalized_prototypes: dict[str, torch.Tensor] = {}
    for key, prototype in prototypes.items():
        if not isinstance(prototype, torch.Tensor) or prototype.ndim != 1:
            raise ValueError(f"{path} 中 {key!r} 的 prototype 必须是 [D] Tensor")
        prototype = F.normalize(prototype.float(), dim=0)
        if not bool(torch.isfinite(prototype).all()):
            raise ValueError(f"{path} 中 {key!r} 的 prototype 含有 NaN/Inf")
        normalized_prototypes[str(key)] = prototype.cpu()

    feature_dims = {int(prototype.shape[0]) for prototype in normalized_prototypes.values()}
    if len(feature_dims) != 1:
        raise ValueError(f"{path} 中各 prototype 的维度不一致: {sorted(feature_dims)}")

    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "source_path": str(path),
            "reference_keys": list(normalized_prototypes.keys()),
            "feature_dim": int(next(iter(feature_dims))),
        }
    )
    return normalized_prototypes, metadata


@torch.no_grad()
def build_reference_prototype_from_prompts(
    *,
    proxy_runner,
    encoder: LatentBackboneFeatureEncoder,
    prompts: Sequence[str],
    seeds: Sequence[int],
    target_steps: Sequence[int],
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a latent-space prototype with the same proxy runner used for training."""

    if not prompts:
        raise ValueError("latent reference prompts 不能为空")
    if not seeds:
        raise ValueError("latent reference seeds 不能为空")
    if not target_steps:
        raise ValueError("latent reference target_steps 不能为空")

    device = torch.device(device)
    all_features = []
    for prompt in prompts:
        prompt_embeds, negative_embeds = proxy_runner.encode_prompts([prompt])
        prompt_embeds = [embed.to(device) for embed in prompt_embeds]
        negative_embeds = [embed.to(device) for embed in negative_embeds]
        for target_step in target_steps:
            for seed in seeds:
                latent_x1 = proxy_runner.denoise_to_step(
                    prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    target_step=int(target_step),
                    seeds=[int(seed)],
                )
                feature = encoder.encode_latents(latent_x1)
                all_features.append(feature.detach().cpu())

    features = torch.cat(all_features, dim=0)
    prototype = F.normalize(features.float().mean(dim=0), dim=0)
    metadata = {
        "source": "generated_from_reference_prompts",
        "prompts": list(prompts),
        "seeds": [int(seed) for seed in seeds],
        "target_steps": [int(step) for step in target_steps],
        "num_reference_features": int(features.shape[0]),
        "feature_dim": int(features.shape[-1]),
    }
    return prototype.cpu(), metadata


class LatentBackboneCosineScorer(nn.Module):
    """Cosine scorer for one prototype or per-sample multi-IP prototypes."""

    def __init__(
        self,
        *,
        encoder: LatentBackboneFeatureEncoder,
        target_prototype: torch.Tensor | None = None,
        target_prototypes: Mapping[str, torch.Tensor] | None = None,
        default_target_key: str | None = None,
        reference_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        if (target_prototype is None) == (target_prototypes is None):
            raise ValueError("target_prototype 和 target_prototypes 必须二选一")

        self.is_multi_ip = target_prototypes is not None
        if not self.is_multi_ip:
            prototype = F.normalize(target_prototype.float().flatten(), dim=0)
            self._validate_prototype_dim(prototype)
            self.register_buffer("target_prototype", prototype.to(self.encoder.device), persistent=True)
            self.prototype_keys: tuple[str, ...] = ()
            self.default_target_key = None
        else:
            normalized_prototypes = {
                str(key): F.normalize(value.float().flatten(), dim=0)
                for key, value in (target_prototypes or {}).items()
            }
            if not normalized_prototypes:
                raise ValueError("target_prototypes 不能为空")
            for prototype in normalized_prototypes.values():
                self._validate_prototype_dim(prototype)

            self.prototype_keys = tuple(normalized_prototypes.keys())
            prototype_matrix = torch.stack(
                [normalized_prototypes[key] for key in self.prototype_keys],
                dim=0,
            )
            self.register_buffer(
                "target_prototypes",
                prototype_matrix.to(self.encoder.device),
                persistent=True,
            )
            self.default_target_key = default_target_key or self.prototype_keys[0]
            self._validate_target_key(self.default_target_key)

        self.reference_metadata = dict(reference_metadata or {})

    def _validate_prototype_dim(self, prototype: torch.Tensor) -> None:
        if prototype.shape[0] != int(self.encoder.feature_dim):
            raise ValueError(
                "latent reference feature 维度不匹配："
                f"reference={prototype.shape[0]}, model={self.encoder.feature_dim}"
            )

    @staticmethod
    def _condition_alias(target_key: str) -> str:
        aliases = {
            "ip_snow_white": "Snow White",
            "ip_doraemon": "Doraemon",
            "ip_minion": "Minions",
            "ip_minions": "Minions",
            "ip_elsa": "Elsa",
            "ip_spongebob": "SpongeBob SquarePants",
        }
        normalized = str(target_key).strip().casefold()
        return aliases.get(normalized, str(target_key).strip())

    def _validate_target_key(self, target_key: str) -> None:
        self._resolve_target_key(target_key)

    def _resolve_target_key(self, target_key: str) -> str:
        if not self.is_multi_ip:
            raise RuntimeError("单 prototype scorer 不支持 target key")

        normalized_keys = {
            key.casefold(): key
            for key in self.prototype_keys
        }
        candidate = self._condition_alias(target_key)
        if candidate.casefold() in normalized_keys:
            return normalized_keys[candidate.casefold()]
        if str(target_key).strip().casefold() == "ip":
            return str(self.default_target_key)
        available = ", ".join(self.prototype_keys)
        raise KeyError(f"prototypes 中没有 {target_key!r}；可选: {available}")

    def forward(
        self,
        latent_x1: torch.Tensor,
        *,
        target_keys: Sequence[str] | None = None,
    ) -> torch.Tensor:
        features = self.encoder.encode_latents(latent_x1)
        if not self.is_multi_ip:
            return torch.matmul(features, self.target_prototype.to(features.dtype)).float()

        if target_keys is None:
            target_keys = [str(self.default_target_key)] * int(features.shape[0])
        if len(target_keys) != int(features.shape[0]):
            raise ValueError(
                f"target_keys 数量 ({len(target_keys)}) "
                f"和 batch size ({features.shape[0]}) 不一致"
            )

        key_to_index = {
            key: index
            for index, key in enumerate(self.prototype_keys)
        }
        indices = [
            key_to_index[self._resolve_target_key(target_key)]
            for target_key in target_keys
        ]
        index_tensor = torch.tensor(indices, device=features.device, dtype=torch.long)
        selected_prototypes = self.target_prototypes.to(features.dtype)[index_tensor]
        return (features * selected_prototypes).sum(dim=-1).float()
