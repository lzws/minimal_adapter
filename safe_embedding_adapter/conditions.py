"""训练启动时预计算的文本 condition。"""

from __future__ import annotations

import torch

from .attention_models import IP_CONDITION_TEXTS
from .model import IP_CONDITION_NAMES, RISK_NAME_TO_ID


@torch.no_grad()
def precompute_ip_condition_embeddings(
    proxy_runner,
    *,
    num_risk_types: int,
    embedding_dim: int,
) -> torch.Tensor:
    """用 Z-Image text encoder 预计算 5 个 IP condition embedding。

    Args:
        proxy_runner:
            已加载 Z-Image pipeline 的 ZImageProxyLatentRunner。
        num_risk_types:
            adapter risk condition table 的行数，通常为 9。
        embedding_dim:
            Z-Image text encoder hidden dim D。

    Returns:
        condition_table:
            shape [num_risk_types, embedding_dim]。5 个 IP-specific condition
            使用对应目标概念文本的有效 token mean pooling；通用 ip/porn/gore/
            benign condition 使用 5 个 IP condition 的均值作为回退。

    说明：
        这些 embedding 由冻结 text encoder 产生，返回后会作为 adapter 的
        persistent buffer 保存到 checkpoint，不参与反向传播。
    """

    concept_prompts = [IP_CONDITION_TEXTS[name] for name in IP_CONDITION_NAMES]
    concept_embeds, _ = proxy_runner.encode_prompts(concept_prompts)
    if len(concept_embeds) != len(IP_CONDITION_NAMES):
        raise ValueError(
            "目标概念 embedding 数量不正确："
            f"expected={len(IP_CONDITION_NAMES)} actual={len(concept_embeds)}"
        )

    pooled_embeddings = []
    for condition_name, embed in zip(IP_CONDITION_NAMES, concept_embeds):
        if embed.ndim != 2 or embed.shape[-1] != embedding_dim:
            raise ValueError(
                f"{condition_name} embedding shape 不正确，"
                f"expected=[T,{embedding_dim}] actual={tuple(embed.shape)}"
            )
        if embed.shape[0] == 0:
            raise ValueError(f"{condition_name} 没有有效 token embedding")
        pooled_embeddings.append(embed.float().mean(dim=0))

    ip_embeddings = torch.stack(pooled_embeddings, dim=0)  # [5, D]
    fallback_embedding = ip_embeddings.mean(dim=0)  # [D]
    table = torch.zeros(num_risk_types, embedding_dim, dtype=torch.float32, device=ip_embeddings.device)
    for condition_name, embedding in zip(IP_CONDITION_NAMES, ip_embeddings):
        condition_id = RISK_NAME_TO_ID[condition_name]
        table[condition_id] = embedding

    # 对旧的通用 risk_id 提供可用回退，避免 metadata 缺失时查到全零 condition。
    for fallback_name in ("porn", "gore", "ip", "benign"):
        fallback_id = RISK_NAME_TO_ID[fallback_name]
        table[fallback_id] = fallback_embedding
    return table

