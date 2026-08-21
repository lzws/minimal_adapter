"""Z-03 latent safety feedback for Z-Image STG.

STG 的 pipeline 只要求 safety model 暴露一个接口：

    scores, masks = safety_model.safety_score(x)

其中 scores 越大表示越不安全，pipeline 会对 scores 做梯度下降来更新
prompt embedding。这个包装器把输入从 pixel image 换成 Z-03 使用的
proxy clean latent：

    latent_x1 = x_t - sigma_t * velocity

因此它必须和 `pipelines/ZImageSTGPipeline.py` 中的 latent feedback 分支配合使用。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from safe_embedding_adapter.z03 import (
    IP_CONDITION_TO_CLASS_ID,
    IP_OTHER_CLASS_ID,
    Z03PornScorer,
)


class Z03LatentSafetyModel(nn.Module):
    """把冻结 Z-03 分类器包装成 STG safety feedback。

    Args:
        checkpoint_path: Z-03 checkpoint 路径。
        device: 推理设备。
        target_risk: 使用哪个 Z-03 head，支持 porn/gore/ip。
        loss_type: STG 优化的 per-sample loss。ce 最稳定；
            softplus_margin/hinge 更接近 margin pushing；
            target_logit/prob 是更直接但可能不稳定的打分。
        margin: margin loss 使用的间隔。
        threshold: mask 阈值。只有 mask=True 的样本才更新 prompt embedding。
        mask_score: 用哪个量判断 mask，prob 表示风险概率，
            unsafe_margin 表示 unsafe_score - safe_score，loss 表示当前 loss。
        ip_loss_mode: target_risk=ip 时，known_sum 压所有 5 个已知 IP；
            target_class 只压当前样本指定的 IP class。
    """

    feedback_input = "latent"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str,
        *,
        target_risk: str = "porn",
        loss_type: str = "ce",
        margin: float = 0.5,
        threshold: float = 0.0,
        mask_score: str = "unsafe_margin",
        ip_loss_mode: str = "known_sum",
    ) -> None:
        super().__init__()
        if target_risk not in {"porn", "gore", "ip"}:
            raise ValueError(f"target_risk must be porn/gore/ip, got {target_risk!r}")
        if loss_type not in {"ce", "softplus_margin", "hinge", "target_logit", "prob"}:
            raise ValueError(f"Unsupported Z-03 STG loss_type: {loss_type!r}")
        if mask_score not in {"prob", "unsafe_margin", "loss"}:
            raise ValueError(f"Unsupported Z-03 mask_score: {mask_score!r}")
        if ip_loss_mode not in {"known_sum", "target_class"}:
            raise ValueError(f"Unsupported Z-03 ip_loss_mode: {ip_loss_mode!r}")

        self.device = torch.device(device)
        self.target_risk = target_risk
        self.loss_type = loss_type
        self.margin = float(margin)
        self.threshold = float(threshold)
        self.mask_score = mask_score
        self.ip_loss_mode = ip_loss_mode
        self.target_class_index: int | None = None

        self.z03 = Z03PornScorer(checkpoint_path, self.device)
        self.z03.eval()
        for parameter in self.z03.parameters():
            parameter.requires_grad_(False)

    def set_target_class_index(self, class_index: int | None) -> None:
        """为单条 IP prompt 设置 target class。

        当前 runner 是逐 prompt 生成，batch size 为 1，因此这里只保存一个
        class id。若后续做 batch 推理，可以扩展成 shape [B] 的 tensor。
        """

        if class_index is None:
            self.target_class_index = None
            return
        class_index = int(class_index)
        if class_index not in IP_CONDITION_TO_CLASS_ID.values():
            raise ValueError(f"IP target class must be one of 0-4, got {class_index}")
        self.target_class_index = class_index

    def safety_score(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 STG 使用的风险 loss 和更新 mask。

        Args:
            latents: Z-Image proxy clean latent, shape [B, 16, H, W]。

        Returns:
            scores: shape [B]，越大表示越应该被 STG 压低。
            masks: shape [B] bool，True 表示该样本需要更新 prompt embedding。
        """

        z03_output = self.z03(latents)
        logits = z03_output.risk_logits(self.target_risk)

        safe_score, unsafe_score = self._safe_and_unsafe_scores(logits)
        risk_prob = self._risk_prob(logits)
        scores = self._loss_scores(logits, safe_score, unsafe_score, risk_prob)

        if self.mask_score == "prob":
            mask_values = risk_prob
        elif self.mask_score == "loss":
            mask_values = scores.detach()
        else:
            mask_values = (unsafe_score - safe_score).detach()
        masks = mask_values > self.threshold

        return scores, masks.to(device=latents.device)

    def _target_class_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.target_risk != "ip" or self.ip_loss_mode != "target_class":
            return None
        if self.target_class_index is None:
            return None
        return torch.full((batch_size,), self.target_class_index, dtype=torch.long, device=device)

    def _safe_and_unsafe_scores(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_class_index = self._safe_class_index()
        safe_score = logits[:, safe_class_index]

        target_class_indices = self._target_class_tensor(logits.shape[0], logits.device)
        if target_class_indices is not None:
            row_indices = torch.arange(logits.shape[0], device=logits.device)
            unsafe_score = logits[row_indices, target_class_indices]
            return safe_score, unsafe_score

        unsafe_indices = [index for index in range(logits.shape[-1]) if index != safe_class_index]
        unsafe_index_tensor = torch.tensor(unsafe_indices, dtype=torch.long, device=logits.device)
        unsafe_logits = logits.index_select(dim=-1, index=unsafe_index_tensor)
        if unsafe_logits.shape[-1] == 1:
            unsafe_score = unsafe_logits[:, 0]
        else:
            unsafe_score = torch.logsumexp(unsafe_logits, dim=-1)
        return safe_score, unsafe_score

    def _risk_prob(self, logits: torch.Tensor) -> torch.Tensor:
        safe_class_index = self._safe_class_index()
        probs = F.softmax(logits, dim=-1)

        target_class_indices = self._target_class_tensor(logits.shape[0], logits.device)
        if target_class_indices is not None:
            row_indices = torch.arange(logits.shape[0], device=logits.device)
            return probs[row_indices, target_class_indices]

        risk_indices = [index for index in range(logits.shape[-1]) if index != safe_class_index]
        risk_index_tensor = torch.tensor(risk_indices, dtype=torch.long, device=logits.device)
        return probs.index_select(dim=-1, index=risk_index_tensor).sum(dim=-1)

    def _loss_scores(
        self,
        logits: torch.Tensor,
        safe_score: torch.Tensor,
        unsafe_score: torch.Tensor,
        risk_prob: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "prob":
            return risk_prob
        if self.loss_type == "target_logit":
            return unsafe_score
        if self.loss_type == "softplus_margin":
            return F.softplus(unsafe_score - safe_score + self.margin)
        if self.loss_type == "hinge":
            return F.relu(unsafe_score - safe_score + self.margin).pow(2)

        safe_class_index = self._safe_class_index()
        target_class_indices = self._target_class_tensor(logits.shape[0], logits.device)
        if target_class_indices is not None:
            binary_logits = torch.stack([safe_score, unsafe_score], dim=-1)
            target = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
            return F.cross_entropy(binary_logits, target, reduction="none")

        target = torch.full(
            (logits.shape[0],),
            safe_class_index,
            dtype=torch.long,
            device=logits.device,
        )
        return F.cross_entropy(logits, target, reduction="none")

    def _safe_class_index(self) -> int:
        if self.target_risk in {"porn", "gore"}:
            return 0
        if self.target_risk == "ip":
            return IP_OTHER_CLASS_ID
        raise ValueError(f"Unsupported target_risk: {self.target_risk}")
