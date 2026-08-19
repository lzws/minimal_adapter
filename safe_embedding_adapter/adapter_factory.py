"""Adapter 构造与 checkpoint 兼容工具。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.nn as nn

from .attention_models import (
    BottleneckSelfAttentionAdapter,
    RiskQueryAttentionGateAdapter,
    RiskQueryFiLMGateAdapter,
    ZImageAdaLNClassifierConditionAdapter,
    ZImageAdaLNTextConditionAdapter,
)
from .model import SafeEmbeddingAdapter


ADAPTER_TYPES = {
    "mlp",
    "risk_query_attention_gate",
    "risk_query_film_gate",
    "zimage_adaln_text_condition",
    "zimage_adaln_classifier_condition",
    "bottleneck_self_attention",
}


def config_to_dict(config: Any) -> dict[str, Any]:
    """把 dataclass/dict/对象配置统一成普通 dict。"""

    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def normalize_adapter_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """去掉 torch.compile/DDP 可能加上的 key 前缀。"""

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def infer_adapter_type_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """从参数名推断 adapter 类型，兼容少数缺少 adapter_config 的 checkpoint。"""

    normalized_keys = normalize_adapter_state_dict(state_dict).keys()
    if any(key.startswith("class_condition_mlp.") for key in normalized_keys):
        return "zimage_adaln_classifier_condition"
    if any(key.startswith("blocks.") and ".adaLN_modulation." in key for key in normalized_keys):
        return "zimage_adaln_text_condition"
    if any(key.startswith("blocks.") and ".self_attn." in key for key in normalized_keys):
        return "bottleneck_self_attention"
    if any(key.startswith("blocks.") and ".film_proj." in key for key in normalized_keys):
        return "risk_query_film_gate"
    if any(key.startswith("blocks.") and ".query_proj." in key for key in normalized_keys):
        return "risk_query_attention_gate"
    return "mlp"


def infer_gate_type_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """从旧 MLP checkpoint 参数名推断 gate 类型。"""

    normalized_keys = normalize_adapter_state_dict(state_dict).keys()
    if any(key.startswith("token_gate_blocks.") for key in normalized_keys):
        return "token"
    if "gate_logits" in normalized_keys:
        return "global"
    return "none"


def infer_embedding_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    """从 checkpoint 参数形状推断 text embedding dim。"""

    normalized = normalize_adapter_state_dict(state_dict)
    for key in (
        "input_norm.weight",
        "risk_embedding.weight",
        "condition_embeddings",
        "blocks.0.token_norm.weight",
        "blocks.0.input_norm.weight",
    ):
        tensor = normalized.get(key)
        if tensor is None:
            continue
        if key in {"risk_embedding.weight", "condition_embeddings"}:
            return int(tensor.shape[-1])
        return int(tensor.numel())
    raise ValueError("无法从 adapter_state_dict 推断 embedding_dim")


def build_adapter_from_config(
    config: Any,
    *,
    embedding_dim: int,
    condition_embeddings: torch.Tensor | None = None,
) -> nn.Module:
    """按 AdapterConfig 构造 adapter。

    支持的 adapter_type:
        mlp:                         原 SafeEmbeddingAdapter。
        risk_query_attention_gate:   risk condition query -> token attention gate。
        risk_query_film_gate:        risk condition query gate + FiLM bottleneck delta。
        zimage_adaln_text_condition: 目标概念文本 condition + Z-Image AdaLN 分支调制。
        zimage_adaln_classifier_condition:
                                    Z-03 classifier one-hot condition + Z-Image AdaLN。
        bottleneck_self_attention:   低维 self-attention residual adapter。
    """

    data = config_to_dict(config)
    adapter_type = str(data.get("adapter_type") or "mlp")
    if adapter_type not in ADAPTER_TYPES:
        raise ValueError(f"未知 adapter_type: {adapter_type}, 可选 {sorted(ADAPTER_TYPES)}")

    common = {
        "embedding_dim": int(embedding_dim),
        "adapter_depth": int(data.get("adapter_depth", 1)),
        "residual_scale": float(data.get("residual_scale", 0.1)),
        "dropout": float(data.get("dropout", 0.0)),
        "use_risk_condition": bool(data.get("use_risk_condition", True)),
        "num_risk_types": int(data.get("num_risk_types", 9)),
        "clamp_delta": bool(data.get("clamp_delta", True)),
    }
    bottleneck_dim = data.get("bottleneck_dim")
    attention_dim = data.get("attention_dim")

    if adapter_type == "risk_query_attention_gate":
        return RiskQueryAttentionGateAdapter(
            **common,
            bottleneck_dim=bottleneck_dim,
            attention_dim=attention_dim,
            gate_init=float(data.get("gate_init", 0.2)),
            zero_init=bool(data.get("zero_init_depth2", True)),
        )

    if adapter_type == "risk_query_film_gate":
        return RiskQueryFiLMGateAdapter(
            **common,
            bottleneck_dim=bottleneck_dim,
            attention_dim=attention_dim,
            gate_init=float(data.get("gate_init", 0.2)),
            zero_init=bool(data.get("zero_init_depth2", True)),
        )

    if adapter_type == "zimage_adaln_text_condition":
        return ZImageAdaLNTextConditionAdapter(
            **common,
            hidden_dim=int(attention_dim or 256),
            num_heads=int(data.get("attention_heads", 4)),
            gate_init=float(data.get("gate_init", 0.2)),
            zero_init=bool(data.get("zero_init_depth2", True)),
            condition_embeddings=condition_embeddings,
        )

    if adapter_type == "zimage_adaln_classifier_condition":
        return ZImageAdaLNClassifierConditionAdapter(
            **common,
            hidden_dim=int(attention_dim or 256),
            num_heads=int(data.get("attention_heads", 4)),
            gate_init=float(data.get("gate_init", 0.2)),
            zero_init=bool(data.get("zero_init_depth2", True)),
            num_classifier_classes=int(data.get("num_classifier_classes", 5)),
            classifier_condition_hidden_dim=data.get("classifier_condition_hidden_dim"),
        )

    if adapter_type == "bottleneck_self_attention":
        return BottleneckSelfAttentionAdapter(
            **common,
            attention_dim=attention_dim,
            num_heads=int(data.get("attention_heads", 4)),
            zero_init=bool(data.get("zero_init_depth2", True)),
            ffn_multiplier=int(data.get("attention_ffn_multiplier", 4)),
        )

    gate_type = data.get("gate_type")
    if gate_type is None:
        gate_type = "global" if bool(data.get("learnable_gate", False)) else "none"
    return SafeEmbeddingAdapter(
        **common,
        bottleneck_dim=bottleneck_dim,
        gate_type=str(gate_type),
        learnable_gate=str(gate_type) != "none",
        gate_init=float(data.get("gate_init", 0.5)),
        zero_init_depth2=bool(data.get("zero_init_depth2", True)),
    )
