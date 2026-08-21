"""Z-Image 版 SAFREE 文本空间投影工具。

官方 SAFREE Stable Diffusion 实现是一种 training-free 的文本 embedding 投影方法：

1. 用一组危险概念 prompt 构造危险概念子空间。
2. 对当前输入 prompt 构造 masked-input 子空间。
3. 将被选中的 token embedding 替换为“危险子空间正交方向”上的投影结果。

Z-Image 的文本编码会套 chat template，并且 pipeline 接收的是变长 token embedding
列表。因此这里把 SAFREE 的核心思想适配为单条 prompt 的 ``[T, D]`` 张量：
``T`` 是去掉 padding 后的有效 token 数，``D`` 是 text encoder hidden size。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class EncodedPrompt:
    """一条 Z-Image prompt 的 embedding 以及 user content token mask。

    张量形状：
        embeds: ``[T, D]``，T 是非 padding 的有效 token 数。
        content_mask: ``[T]``，True 表示该位置属于用户原始 prompt 内容。
    """

    embeds: torch.Tensor
    content_mask: torch.Tensor


@dataclass
class SAFREEProjectionResult:
    """SAFREE 投影后的 prompt embedding 以及调试指标。"""

    embeds: torch.Tensor
    projected_tokens: int
    content_tokens: int
    beta: float
    adaptive_end_step: int | None
    mean_token_cosine: float


def projection_matrix_from_rows(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算由 ``vectors`` 行向量张成子空间的投影矩阵。

    Args:
        vectors: 形状为 ``[N, D]``。N 是子空间基向量个数，D 是 embedding 维度。
        eps: 加到 Gram 矩阵对角线上的小扰动，用于数值稳定。

    Returns:
        形状为 ``[D, D]`` 的投影矩阵 P。任意 ``x: [D]`` 可通过 ``P @ x``
        投影到 ``vectors`` 张成的子空间。
    """

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be [N, D], got {tuple(vectors.shape)}")
    if vectors.shape[0] == 0:
        raise ValueError("Cannot build a projection matrix from zero vectors")

    vectors = vectors.float()
    # vectors: [N, D] -> basis: [D, N]，后续按列向量基来写投影公式。
    basis = vectors.transpose(0, 1)

    # gram: [N, N]，等价于 B^T B。使用 pinv 支持基向量线性相关的情况。
    gram = basis.transpose(0, 1) @ basis
    if eps > 0:
        gram = gram + eps * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)

    # 返回 P = B (B^T B)^+ B^T，形状 [D, D]。
    return basis @ torch.linalg.pinv(gram) @ basis.transpose(0, 1)


def format_zimage_user_prompt(tokenizer, prompt: str) -> str:
    """套用 Z-Image pipeline 中实际使用的 chat template。"""

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def content_token_boundary_counts(pipe, max_sequence_length: int) -> tuple[int, int]:
    """计算 user content 前后固定 chat-template token 数。

    Z-Image 会把用户 prompt 包进稳定的 chat template。这里用一个 marker prompt
    做一次 tokenization，找到 marker 对应 token 的位置，从而得到：

    - ``prefix_count``：用户内容前面的 template token 数。
    - ``suffix_count``：用户内容后面的 template token 数。

    这两个值只依赖 tokenizer、template 和 ``max_sequence_length``，所以会缓存起来。
    """

    cache_name = "_safree_content_token_boundary_cache"
    if not hasattr(pipe, cache_name):
        setattr(pipe, cache_name, {})
    cache: dict[int, tuple[int, int]] = getattr(pipe, cache_name)

    cache_key = int(max_sequence_length)
    if cache_key in cache:
        return cache[cache_key]

    marker = "SAFREE_ZIMAGE_USER_CONTENT_MARKER_20260806"
    formatted_prompt = format_zimage_user_prompt(pipe.tokenizer, marker)
    marker_start = formatted_prompt.find(marker)
    if marker_start < 0:
        raise ValueError("Cannot locate user content marker in Z-Image chat template")
    marker_end = marker_start + len(marker)

    text_inputs = pipe.tokenizer(
        [formatted_prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = getattr(text_inputs, "offset_mapping", None)
    if offset_mapping is None:
        raise ValueError("Tokenizer did not return offset_mapping; cannot build content token mask")

    # offset_mapping[0]: [L, 2]，L=max_sequence_length，每个 token 在字符串中的字符范围。
    # attention_mask[0]: [L]，True 表示非 padding token。
    attention_mask = text_inputs.attention_mask[0].bool()

    # valid_offsets: [T_template, 2]，只保留非 padding 的有效 token。
    valid_offsets = offset_mapping[0][attention_mask]
    content_indices: list[int] = []
    for token_index, (token_start, token_end) in enumerate(valid_offsets.tolist()):
        token_start = int(token_start)
        token_end = int(token_end)
        if max(token_start, marker_start) < min(token_end, marker_end):
            content_indices.append(token_index)

    if not content_indices:
        raise ValueError("Failed to locate marker tokens after tokenization")

    # marker token 在有效 token 序列中的首尾位置决定 template 边界。
    prefix_count = int(content_indices[0])
    suffix_count = int(valid_offsets.shape[0]) - int(content_indices[-1]) - 1
    cache[cache_key] = (prefix_count, suffix_count)
    return prefix_count, suffix_count


def build_content_mask(
    pipe,
    *,
    valid_token_count: int,
    max_sequence_length: int,
    device: torch.device,
    prompt: str,
) -> torch.Tensor:
    """构造用户原始 prompt 内容对应的 ``[T]`` bool mask。"""

    prefix_count, suffix_count = content_token_boundary_counts(pipe, max_sequence_length)
    content_start = prefix_count
    content_end = int(valid_token_count) - suffix_count
    if content_start >= content_end:
        raise ValueError(
            "Empty SAFREE content token mask. The prompt may be empty, heavily truncated, "
            f"or incompatible with the cached chat-template boundary. prompt={prompt!r}"
        )

    # mask: [T]，T=valid_token_count；中间 user content 位置置为 True。
    mask = torch.zeros(int(valid_token_count), dtype=torch.bool, device=device)
    mask[content_start:content_end] = True
    return mask


@torch.no_grad()
def encode_prompt_with_content_mask(
    pipe,
    prompt: str,
    *,
    device: torch.device,
    max_sequence_length: int,
) -> EncodedPrompt:
    """编码一条 Z-Image prompt，并返回 user content token mask。"""

    formatted_prompt = format_zimage_user_prompt(pipe.tokenizer, prompt)
    text_inputs = pipe.tokenizer(
        [formatted_prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    # input_ids / attention_mask: [1, L]，L=max_sequence_length。
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()

    # hidden: [1, L, D]。Z-Image pipeline 通常取倒数第二层 hidden states。
    hidden = pipe.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2]

    # valid_mask: [L]；embeds: [T, D]，T 是去掉 padding 后的有效 token 数。
    valid_mask = attention_mask[0]
    embeds = hidden[0][valid_mask]

    # content_mask: [T]，只标记用户 prompt 内容，不包含 chat template/special token。
    content_mask = build_content_mask(
        pipe,
        valid_token_count=int(valid_mask.sum().item()),
        max_sequence_length=max_sequence_length,
        device=device,
        prompt=prompt,
    )
    return EncodedPrompt(embeds=embeds, content_mask=content_mask)


@torch.no_grad()
def encode_concept_vectors(
    pipe,
    concept_prompts: Sequence[str],
    *,
    device: torch.device,
    max_sequence_length: int,
) -> torch.Tensor:
    """把危险概念 prompt 编码成 mean-pooled user-content 向量。

    Returns:
        ``[N_concept, D]``，每一行是一个危险概念 prompt 的内容 token 平均向量。
    """

    vectors: list[torch.Tensor] = []
    for concept_prompt in concept_prompts:
        encoded = encode_prompt_with_content_mask(
            pipe,
            concept_prompt,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        # content_embeds: [T_content, D]，只取用户内容 token 做危险概念表示。
        content_embeds = encoded.embeds[encoded.content_mask]
        if content_embeds.numel() == 0:
            raise ValueError(f"Unsafe concept prompt has no content tokens: {concept_prompt!r}")

        # mean(dim=0): [D]，得到该危险概念 prompt 的一个向量表示。
        vectors.append(content_embeds.float().mean(dim=0))

    # [N_concept, D]，后续用这些行向量构造危险概念投影矩阵 P_risk。
    return torch.stack(vectors, dim=0)


def masked_prompt_vectors(content_embeds: torch.Tensor) -> torch.Tensor:
    """用 token embedding 近似 SAFREE 的 masked prompt 向量。

    SD 版 SAFREE 会对每个被 mask 的 token 重新编码一次 prompt，并使用 text encoder
    pooler output。这里 Z-Image 暴露的是 token embedding，因此我们用“去掉当前 token
    后剩余 user-content token 的均值”来近似对应的 masked prompt 表示。

    Args:
        content_embeds: ``[T_content, D]``，只包含用户内容 token。

    Returns:
        ``[T_content, D]``。第 i 行表示“第 i 个内容 token 被 mask 后”的近似向量。
    """

    if content_embeds.ndim != 2:
        raise ValueError(f"content_embeds must be [T_content, D], got {tuple(content_embeds.shape)}")
    token_count = int(content_embeds.shape[0])
    if token_count == 0:
        raise ValueError("Cannot build masked vectors for zero content tokens")
    if token_count == 1:
        return content_embeds.float().clone()

    content_embeds = content_embeds.float()

    # total: [1, D]；(total - content_embeds): [T_content, D]。
    total = content_embeds.sum(dim=0, keepdim=True)
    return (total - content_embeds) / float(token_count - 1)


def mean_without_self(values: torch.Tensor) -> torch.Tensor:
    """计算每个位置“除自己之外”的均值，复现 SAFREE 的 token 阈值逻辑。"""

    if values.ndim != 1:
        raise ValueError(f"values must be [T], got {tuple(values.shape)}")
    if values.numel() == 1:
        return values.clone()

    total = values.sum()
    return (total - values) / float(values.numel() - 1)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def f_beta(beta: float, *, upperbound_timestep: int = 10, concept_type: str = "nudity") -> int:
    """复现 SAFREE self-validation filter 中 beta 到步数上界的映射。"""

    if "artists-" in concept_type:
        midpoint = 5.5
        sharpness = 3.5
    else:
        midpoint = 5.333
        sharpness = 2.5
    value = sigmoid(2.0 * sharpness * (10.0 * beta - midpoint))
    return round(float(upperbound_timestep) * value)


@torch.no_grad()
def apply_safree_projection(
    prompt_embeds: torch.Tensor,
    content_mask: torch.Tensor,
    concept_projection: torch.Tensor,
    *,
    alpha: float = 0.01,
    scale: float = 1.0,
    use_self_validation_filter: bool = False,
    up_t: int = 10,
    concept_type: str = "nudity",
) -> SAFREEProjectionResult:
    """对一条 Z-Image prompt embedding 应用 SAFREE 文本投影。

    Args:
        prompt_embeds: 原始 Z-Image prompt embedding，形状 ``[T, D]``。
        content_mask: 形状 ``[T]``，True 表示 user content token。
        concept_projection: 危险概念子空间投影矩阵，形状 ``[D, D]``。
        alpha: SAFREE token 选择阈值的容忍系数。
        scale: 最终 delta 的插值强度。1.0 表示完全使用投影结果，0.0 表示保持原 embedding。
        use_self_validation_filter: 是否计算 SAFREE 的自适应 denoise step 截断 beta。
        up_t: ``f_beta`` 使用的最大 step 上界。
        concept_type: ``f_beta`` 使用的 SAFREE 概念类型字符串。

    Returns:
        ``SAFREEProjectionResult``，其中 ``embeds`` 仍是 ``[T, D]``。
    """

    if prompt_embeds.ndim != 2:
        raise ValueError(f"prompt_embeds must be [T, D], got {tuple(prompt_embeds.shape)}")
    if content_mask.ndim != 1 or content_mask.shape[0] != prompt_embeds.shape[0]:
        raise ValueError(
            "content_mask must be [T] and match prompt_embeds. "
            f"embeds={tuple(prompt_embeds.shape)}, mask={tuple(content_mask.shape)}"
        )

    original_dtype = prompt_embeds.dtype
    device = prompt_embeds.device

    # full_embeds: [T, D]，转 float32 计算投影，最后再转回原 dtype。
    full_embeds = prompt_embeds.float()

    # concept_projection: [D, D]，P_risk。
    concept_projection = concept_projection.to(device=device, dtype=torch.float32)

    # content_embeds: [T_content, D]，只对用户输入内容做 SAFREE 修改。
    content_embeds = full_embeds[content_mask]

    if content_embeds.shape[0] == 0:
        return SAFREEProjectionResult(
            embeds=prompt_embeds,
            projected_tokens=0,
            content_tokens=0,
            beta=0.0,
            adaptive_end_step=None,
            mean_token_cosine=1.0,
        )

    # masked_vectors: [T_content, D]，每行是 mask 掉对应 token 后的 prompt 近似表示。
    masked_vectors = masked_prompt_vectors(content_embeds)

    # masked_projection: [D, D]，当前 prompt 自身 masked-input 子空间的投影矩阵。
    masked_projection = projection_matrix_from_rows(masked_vectors)
    dim = int(content_embeds.shape[-1])
    identity = torch.eye(dim, device=device, dtype=torch.float32)

    # orthogonal_concept: [D, D]，I - P_risk，把向量投到危险概念子空间的正交补。
    orthogonal_concept = identity - concept_projection

    # dist_to_orthogonal: [T_content]。
    # 对每个 masked prompt 向量，计算其落到危险正交补后的范数。
    dist_to_orthogonal = torch.linalg.vector_norm(
        orthogonal_concept @ masked_vectors.transpose(0, 1),
        dim=0,
    )

    # threshold: [T_content]。每个 token 用其它 token 的均值作为参照。
    threshold = (1.0 + float(alpha)) * mean_without_self(dist_to_orthogonal)

    # keep_original: [T_content]。True 表示该 token 不投影，False 表示需要替换。
    keep_original = dist_to_orthogonal < threshold

    # projected_content: [T_content, D]。
    # 先投到当前 prompt 的 masked-input 子空间，再去掉危险概念子空间分量。
    projected_content = (
        orthogonal_concept @ masked_projection @ content_embeds.transpose(0, 1)
    ).transpose(0, 1)

    # merged_content: [T_content, D]。逐 token 选择保留原 embedding 或使用投影结果。
    merged_content = torch.where(keep_original[:, None], content_embeds, projected_content)

    if scale != 1.0:
        # 允许调节 SAFREE 修改强度：delta = projected - original。
        merged_content = content_embeds + float(scale) * (merged_content - content_embeds)

    # modified: [T, D]。只把 user content token 的位置替换掉，template token 保持不变。
    modified = full_embeds.clone()
    modified[content_mask] = merged_content

    # token_cos: [T_content]，用于记录投影前后 token embedding 的平均余弦相似度。
    token_cos = F.cosine_similarity(projected_content, content_embeds, dim=-1)
    beta = float((1.0 - token_cos.mean()).detach().cpu().item())
    adaptive_end_step = None
    if use_self_validation_filter:
        # adaptive_end_step 是 SAFREE 自适应策略给出的最大生效 denoise step。
        adaptive_end_step = f_beta(beta, upperbound_timestep=up_t, concept_type=concept_type)

    projected_tokens = int((~keep_original).sum().detach().cpu().item())
    mean_token_cosine = float(token_cos.mean().detach().cpu().item())

    return SAFREEProjectionResult(
        embeds=modified.to(dtype=original_dtype),
        projected_tokens=projected_tokens,
        content_tokens=int(content_embeds.shape[0]),
        beta=beta,
        adaptive_end_step=adaptive_end_step,
        mean_token_cosine=mean_token_cosine,
    )
