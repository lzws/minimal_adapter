"""SafeEmbeddingAdapter 模型定义。

Adapter 的职责非常窄：输入 Z-Image text encoder 产生的变长 token embeddings，
输出同形状的修正 embeddings。它不接触 tokenizer、denoise、Z-03，也不决定 loss。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


RISK_NAME_TO_ID = {
    "porn": 0,
    "gore": 1,
    "ip": 2,
    "benign": 3,
    "ip_snow_white": 4,
    "ip_doraemon": 5,
    "ip_minion": 6,
    "ip_elsa": 7,
    "ip_spongebob": 8,
}

IP_CONDITION_NAMES = (
    "ip_snow_white",
    "ip_doraemon",
    "ip_minion",
    "ip_elsa",
    "ip_spongebob",
)

IP_SUBCATEGORY_TO_RISK_NAME = {
    "ip_snow_white": "ip_snow_white",
    "ip_doraemon": "ip_doraemon",
    "ip_minion": "ip_minion",
    "ip_elsa": "ip_elsa",
    "ip_spongebob": "ip_spongebob",
}

IP_ORIGINAL_CATEGORY_TO_RISK_NAME = {
    "snow white": "ip_snow_white",
    "白雪公主": "ip_snow_white",
    "doraemon": "ip_doraemon",
    "哆啦a梦": "ip_doraemon",
    "哆啦A梦": "ip_doraemon",
    "minions": "ip_minion",
    "minion": "ip_minion",
    "小黄人": "ip_minion",
    "elsa": "ip_elsa",
    "艾莎": "ip_elsa",
    "spongebob squarepants": "ip_spongebob",
    "sponge bob squarepants": "ip_spongebob",
    "spongebob": "ip_spongebob",
    "海绵宝宝": "ip_spongebob",
}


def normalize_condition_text(value: object) -> str:
    """把 metadata 字段规整成可比较字符串。"""

    return str(value or "").strip().casefold()


def ip_condition_from_metadata(metadata: dict | None, *, fallback: str = "ip") -> str:
    """从样本 metadata 中推断 IP-specific condition。

    训练 CSV 通常使用 sub_category=ip_spongebob 这类字段；正式测试集可能使用
    original_category=SpongeBob SquarePants / Doraemon 等英文名称。这里统一映射
    到 adapter 的 risk condition 名称，缺失时回退到通用 "ip"。
    """

    metadata = metadata or {}
    for key in ("preserve_condition", "target_condition", "risk_id", "target_risk_id"):
        explicit_condition = normalize_condition_text(metadata.get(key))
        if explicit_condition in IP_SUBCATEGORY_TO_RISK_NAME:
            return IP_SUBCATEGORY_TO_RISK_NAME[explicit_condition]
        if explicit_condition in IP_ORIGINAL_CATEGORY_TO_RISK_NAME:
            return IP_ORIGINAL_CATEGORY_TO_RISK_NAME[explicit_condition]

    related_to_ip = normalize_condition_text(metadata.get("related_to_ip"))
    if related_to_ip in IP_SUBCATEGORY_TO_RISK_NAME:
        return IP_SUBCATEGORY_TO_RISK_NAME[related_to_ip]
    if related_to_ip in IP_ORIGINAL_CATEGORY_TO_RISK_NAME:
        return IP_ORIGINAL_CATEGORY_TO_RISK_NAME[related_to_ip]

    sub_category = normalize_condition_text(metadata.get("sub_category"))
    if sub_category in IP_SUBCATEGORY_TO_RISK_NAME:
        return IP_SUBCATEGORY_TO_RISK_NAME[sub_category]

    original_category = normalize_condition_text(metadata.get("original_category"))
    if original_category in IP_ORIGINAL_CATEGORY_TO_RISK_NAME:
        return IP_ORIGINAL_CATEGORY_TO_RISK_NAME[original_category]

    sample_id = normalize_condition_text(metadata.get("sample_id") or metadata.get("id"))
    for prefix, condition in IP_SUBCATEGORY_TO_RISK_NAME.items():
        if sample_id.startswith(prefix):
            return condition

    return fallback


def adapter_supports_risk_id(adapter: nn.Module, risk_id: str) -> bool:
    """判断当前 adapter checkpoint 是否支持某个 risk condition。"""

    risk_index = RISK_NAME_TO_ID.get(risk_id)
    if risk_index is None:
        return False
    risk_embedding = getattr(adapter, "risk_embedding", None)
    if risk_embedding is None and hasattr(adapter, "condition_embeddings"):
        condition_embeddings = getattr(adapter, "condition_embeddings")
        return risk_index < int(condition_embeddings.shape[0])
    if risk_embedding is None:
        return True
    return risk_index < int(risk_embedding.num_embeddings)


def _normalize_single_token_mask(
    token_mask: torch.Tensor | None,
    *,
    original_embed: torch.Tensor,
) -> torch.Tensor | None:
    """校验并规整单条 prompt 的 user content token mask。

    Args:
        token_mask: 可选 mask，shape [T] 或 [T, 1]。True/1 表示该 token
            属于原始 user prompt 内容。
        original_embed: 原始 prompt embedding，shape [T, D]。

    Returns:
        mask: shape [T]，dtype bool，device 与 original_embed 一致；None 表示
            未启用 user content token 限制。
    """

    if token_mask is None:
        return None
    if token_mask.ndim == 1:
        mask = token_mask
    elif token_mask.ndim == 2 and token_mask.shape[1] == 1:
        mask = token_mask.view(-1)
    else:
        raise ValueError(
            "token_mask 期望 shape [T] 或 [T, 1]，"
            f"实际 {tuple(token_mask.shape)}，embedding shape={tuple(original_embed.shape)}"
        )

    if mask.shape[0] != original_embed.shape[0]:
        raise ValueError(
            "token_mask 长度必须和 prompt embedding 的 token 数一致："
            f"mask={tuple(mask.shape)}, embedding={tuple(original_embed.shape)}"
        )

    mask = mask.to(device=original_embed.device, dtype=torch.bool)
    if int(mask.sum().item()) <= 0:
        raise ValueError("token_mask 中没有任何 True token，adapter 没有可编辑的 user content token")
    return mask


def select_adapter_input_tokens(
    original_embed: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """选择真正送入 adapter 的 token embeddings。

    Args:
        original_embed: 原始 prompt embedding，shape [T, D]。
        token_mask: 可选 mask，shape [T] 或 [T, 1]。启用时，只有 True
            位置的 user content token 会进入 adapter。

    Returns:
        adapter_input:
            未启用 mask 时 shape [T, D]；
            启用 mask 时 shape [T_content, D]。

    这一步是“只修改 user content token”消融的关键：对 attention adapter 来说，
    chat template / special token 不再出现在 Q/K/V 里。
    """

    mask = _normalize_single_token_mask(token_mask, original_embed=original_embed)
    original_embed = original_embed.float()
    if mask is None:
        return original_embed
    return original_embed[mask]


def apply_token_mask_to_safe_embed(
    original_embed: torch.Tensor,
    safe_embed: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """把 adapter 输出还原成完整 prompt embedding。

    Args:
        original_embed: 原始完整 prompt embedding，shape [T, D]。
        safe_embed:
            未启用 mask 时通常是完整 adapter 输出，shape [T, D]；
            启用 mask 且先经过 select_adapter_input_tokens 时，是只包含
            user content token 的 adapter 输出，shape [T_content, D]。
        token_mask: 可选 mask，shape [T] 或 [T, 1]。True/1 表示该 token
            可以被修改；False/0 表示输出必须等于 original_embed。

    Returns:
        masked_safe_embed: 完整序列，shape [T, D]。
    """

    mask_bool = _normalize_single_token_mask(token_mask, original_embed=original_embed)
    if mask_bool is None:
        return safe_embed
    if safe_embed.ndim != 2 or safe_embed.shape[-1] != original_embed.shape[-1]:
        raise ValueError(
            "safe_embed 期望 shape [T, D] 或 [T_content, D]，"
            f"实际 {tuple(safe_embed.shape)}，original={tuple(original_embed.shape)}"
        )

    original_full = original_embed.to(device=safe_embed.device, dtype=safe_embed.dtype)
    mask_bool = mask_bool.to(device=safe_embed.device)
    if safe_embed.shape == original_full.shape:
        # 兼容旧调用：adapter 已经在完整序列上算完，只在最终 residual 上做 mask。
        mask = mask_bool.view(-1, 1).to(dtype=safe_embed.dtype)
        return original_full + (safe_embed - original_full) * mask

    editable_count = int(mask_bool.sum().item())
    if safe_embed.shape[0] != editable_count:
        raise ValueError(
            "safe_embed 的 token 数必须等于完整 T 或 T_content："
            f"safe={tuple(safe_embed.shape)}, T={original_full.shape[0]}, T_content={editable_count}"
        )

    output = original_full.clone()
    output[mask_bool] = safe_embed
    return output


def normalize_token_masks(
    token_masks: Sequence[torch.Tensor] | None,
    batch_size: int,
) -> list[torch.Tensor | None]:
    """把可选 token mask batch 统一成长度为 B 的列表。"""

    if token_masks is None:
        return [None for _ in range(batch_size)]
    if len(token_masks) != batch_size:
        raise ValueError(f"token_masks 长度必须为 batch size={batch_size}，实际为 {len(token_masks)}")
    return list(token_masks)


class TokenResidualMLPBlock(nn.Module):
    """一个逐 token 的 residual delta block。

    Args:
        x: shape [T_i, D]。

    Returns:
        delta: shape [T_i, D]。

    这个 block 本身只预测 delta，不负责把 delta 加回原 embedding。外层
    SafeEmbeddingAdapter 统一控制 residual_scale 和 tanh clamp。
    """

    def __init__(self, embedding_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(x))


class TokenScalarGateBlock(nn.Module):
    """逐 token gate，输出 shape [T_i, 1]。

    这个 gate 只决定每个 token 的 residual delta 强度，不改变 delta 的方向。
    最后一层 bias 初始化为 gate_init 对应的 logit，weight 使用极小随机值，
    使训练初始时 gate 接近常数，但仍保留 token 条件路径的梯度。
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        last_linear = self.net[-1]
        nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
        nn.init.constant_(last_linear.bias, float(gate_logit))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(x))


class SafeEmbeddingAdapter(nn.Module):
    """token-wise residual prompt embedding adapter。

    输入/输出都是 list[Tensor]，每个 Tensor 形状为 [seq_len_i, hidden_dim]。
    也就是说，它不是把整句 prompt 压成一个向量再改，而是对每个 token 的
    embedding 独立预测一个 residual delta：

        E_orig_i:  [T_i, D]
        delta_i:   [T_i, D]
        gate_i:    [T_i, 1] token gate，或 scalar global gate
        E_safe_i:  [T_i, D] = E_orig_i + residual_scale * gate_i * delta_i

    其中：
        B: batch size，即 list 长度；
        T_i: 第 i 条 prompt 的有效 token 数；
        D: text encoder hidden dim，也就是 embedding_dim。

    使用 list 而不是 padding 后的 batch tensor，是因为 Z-Image 原生 text encoder
    就返回变长 token embeddings，pipeline 的 transformer 也直接接受 list。
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int | None = None,
        adapter_depth: int = 1,
        learnable_gate: bool = False,
        gate_type: str | None = None,
        gate_init: float = 0.5,
        residual_scale: float = 0.1,
        dropout: float = 0.0,
        use_risk_condition: bool = True,
        num_risk_types: int = len(RISK_NAME_TO_ID),
        clamp_delta: bool = True,
        zero_init_depth2: bool = True,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")
        if adapter_depth <= 0:
            raise ValueError("adapter_depth 必须大于 0")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init 必须在 (0, 1) 之间")
        if gate_type is None:
            gate_type = "global" if learnable_gate else "none"
        if learnable_gate and gate_type == "none":
            gate_type = "global"
        if gate_type not in {"none", "global", "token"}:
            raise ValueError(f"未知 gate_type: {gate_type}")

        bottleneck_dim = bottleneck_dim or max(16, embedding_dim // 4)
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.adapter_depth = int(adapter_depth)
        self.gate_type = str(gate_type)
        self.learnable_gate = self.gate_type != "none"
        self.gate_init = float(gate_init)
        self.residual_scale = float(residual_scale)
        self.use_risk_condition = bool(use_risk_condition)
        self.clamp_delta = bool(clamp_delta)
        self.zero_init_depth2 = bool(zero_init_depth2)
        # 运行时推理控制量，默认完全等价于原始 MLP：
        #   condition_scale = 1.0 表示 E + c_ip；
        #   risk_residual_scales 全 1.0 表示不改变最终 residual 幅度。
        # persistent=False 保证旧 checkpoint strict load 不会因为新增 buffer 失败。
        self.runtime_condition_scale = 1.0
        self.register_buffer(
            "_runtime_risk_residual_scales",
            torch.ones(num_risk_types, dtype=torch.float32),
            persistent=False,
        )

        # 第一个 block 保留旧版参数名 input_norm/net，保证旧的 1-block
        # checkpoint 仍然可以 strict load。
        self.input_norm = nn.LayerNorm(self.embedding_dim)

        # risk embedding 是轻量条件输入。当前第一版所有训练样本都用 porn 条件，
        # 但保留 gore/IP 槽位，后续扩展多 head 时不需要改模型结构。
        if self.use_risk_condition:
            self.risk_embedding = nn.Embedding(num_risk_types, self.embedding_dim)
        else:
            self.risk_embedding = None

        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.bottleneck_dim, self.embedding_dim),
        )
        self.extra_blocks = nn.ModuleList(
            [
                TokenResidualMLPBlock(self.embedding_dim, self.bottleneck_dim, dropout)
                for _ in range(self.adapter_depth - 1)
            ]
        )
        self.token_gate_blocks = nn.ModuleList()
        if self.gate_type == "global":
            # 每个 block 一个 scalar gate。实际 scale 是：
            # residual_scale * sigmoid(gate_logits[block_index])。
            # gate_init=0.5 时初始有效 scale 为 residual_scale 的一半，
            # 比固定 residual_scale 更保守，降低语义漂移风险。
            gate_logit = math.log(self.gate_init / (1.0 - self.gate_init))
            self.gate_logits = nn.Parameter(torch.full((self.adapter_depth,), float(gate_logit)))
        elif self.gate_type == "token":
            self.register_parameter("gate_logits", None)
            self.token_gate_blocks = nn.ModuleList(
                [
                    TokenScalarGateBlock(
                        self.embedding_dim,
                        self.bottleneck_dim,
                        dropout,
                        self.gate_init,
                    )
                    for _ in range(self.adapter_depth)
                ]
            )
        else:
            self.register_parameter("gate_logits", None)
        self.delta_activation = nn.Tanh()
        self._last_gate_values: list[float] = []
        # 仅用于推理诊断，不进入 checkpoint。每次 forward 后按
        # [sample][block][token] 保存 token gate。
        self._last_token_gate_tensors: list[list[torch.Tensor]] = []
        if self.zero_init_depth2 and self.adapter_depth == 2:
            self._zero_init_residual_outputs()

    def _zero_init_residual_outputs(self) -> None:
        """depth=2 时把 residual 输出层置零，让 adapter 从 identity 开始。

        这样训练初期 E_safe == E_orig，Z-03 梯度会先更新最后一层；随后再逐步
        传到前面的 bottleneck 层，比随机 residual 初始化更不容易破坏风格。
        """

        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        for block in self.extra_blocks:
            block_last_linear = block.net[-1]
            nn.init.zeros_(block_last_linear.weight)
            nn.init.zeros_(block_last_linear.bias)

    def gate_values(self) -> list[float]:
        """返回当前每个 block 的 gate 值，仅用于日志。"""

        if self.gate_type == "global" and self.gate_logits is not None:
            return [float(value.detach().cpu()) for value in torch.sigmoid(self.gate_logits)]
        if self.gate_type == "token":
            return list(self._last_gate_values)
        return []

    def set_runtime_condition_scale(self, condition_scale: float) -> None:
        """设置推理时 IP condition 注入强度 alpha。

        只改变 forward 行为，不写入 checkpoint，不改变任何可学习参数。
        alpha=1.0 等价于训练时原始行为；alpha 越小，c_ip 对所有 token 的
        直接偏置越弱。
        """

        condition_scale = float(condition_scale)
        if condition_scale < 0:
            raise ValueError("condition_scale 必须大于等于 0")
        self.runtime_condition_scale = condition_scale

    def set_runtime_risk_residual_scales(self, scales_by_risk_id: dict[str, float]) -> None:
        """设置推理时每个 risk_id 的 residual 乘子 s_ip。

        Args:
            scales_by_risk_id: 例如 {"ip_doraemon": 1.2, "ip_spongebob": 0.7}。

        没有显式指定的 risk_id 保持 1.0。这个乘子作用在每个 residual block
        的最终 delta 上：
            E_next = E + s_ip * residual_scale * delta
        """

        scales = torch.ones_like(self._runtime_risk_residual_scales, dtype=torch.float32)
        for risk_id, scale in scales_by_risk_id.items():
            if risk_id not in RISK_NAME_TO_ID:
                raise ValueError(f"未知 risk_id: {risk_id}")
            scale = float(scale)
            if scale < 0:
                raise ValueError(f"{risk_id} 的 residual scale 必须大于等于 0，实际 {scale}")
            risk_index = RISK_NAME_TO_ID[risk_id]
            if risk_index >= scales.numel():
                raise ValueError(
                    f"{risk_id} 的 index={risk_index} 超出当前 adapter risk_embedding 范围 {scales.numel()}"
                )
            scales[risk_index] = scale
        self._runtime_risk_residual_scales.copy_(scales.to(device=self._runtime_risk_residual_scales.device))

    def runtime_risk_residual_scale(self, risk_index: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """读取当前样本的 residual 乘子，返回 scalar tensor。"""

        return self._runtime_risk_residual_scales[risk_index].to(
            device=risk_index.device,
            dtype=dtype,
        )

    def _gate_for_block(
        self,
        block_input: torch.Tensor,
        *,
        block_index: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """返回当前 block 的 gate。

        Returns:
            None: 不使用 gate；
            global gate: scalar tensor；
            token gate: shape [T_i, 1]。
        """

        if self.gate_type == "none":
            return None
        if self.gate_type == "global":
            return torch.sigmoid(self.gate_logits[block_index]).to(
                device=block_input.device,
                dtype=dtype,
            )
        gate_logits = self.token_gate_blocks[block_index](block_input)
        return torch.sigmoid(gate_logits).to(dtype=dtype)

    def _apply_delta_block(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor | None,
        *,
        block_index: int,
        residual_multiplier: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """应用一个 residual MLP block。

        Args:
            hidden: 当前 token embeddings，shape [T_i, D]。
            condition: risk condition，shape [1, D]，可为 None。
            block_index: 第几个 block，0 表示兼容旧 checkpoint 的 input_norm/net。
            residual_multiplier: 推理时按 risk_id 额外乘上的 s_ip；训练默认是 1。

        Returns:
            hidden_next: shape [T_i, D]。
        """

        block_input = hidden + condition if condition is not None else hidden
        if block_index == 0:
            delta = self.net(self.input_norm(block_input))
        else:
            delta = self.extra_blocks[block_index - 1](block_input)
        if self.clamp_delta:
            delta = self.delta_activation(delta)
        scale = self.residual_scale * residual_multiplier
        gate = self._gate_for_block(block_input, block_index=block_index, dtype=hidden.dtype)
        if gate is not None:
            scale = scale * gate
        return hidden + delta * scale, gate

    def _normalize_risk_ids(
        self,
        risk_ids: Sequence[int | str] | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """把 risk 条件统一成 [B] long tensor。

        Args:
            risk_ids: None、长度为 B 的字符串/int 序列，或 shape [B] 的 tensor。
            batch_size: B。

        Returns:
            risk_tensor: shape [B]，dtype=torch.long。
        """

        if risk_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if isinstance(risk_ids, torch.Tensor):
            return risk_ids.to(device=device, dtype=torch.long)

        normalized = []
        for item in risk_ids:
            if isinstance(item, str):
                if item not in RISK_NAME_TO_ID:
                    raise ValueError(f"未知 risk name: {item}")
                normalized.append(RISK_NAME_TO_ID[item])
            else:
                normalized.append(int(item))
        return torch.tensor(normalized, dtype=torch.long, device=device)

    def forward(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        risk_ids: Sequence[int | str] | torch.Tensor | None = None,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """返回修正后的 prompt embeddings。

        Args:
            prompt_embeds:
                list 长度为 B。第 i 个 tensor 形状为 [T_i, D]，
                T_i 是该 prompt 的有效 token 数，D 必须等于 self.embedding_dim。
            risk_ids:
                可选条件，shape/长度为 [B]。当前第一版通常全部是 "porn"。
            token_masks:
                可选 token 修改 mask。第 i 个 mask shape [T_i] 或 [T_i, 1]；
                True 表示允许修改该 token，False 表示输出保持原 embedding。

        Returns:
            safe_embeds:
                list 长度为 B。第 i 个 tensor 形状仍为 [T_i, D]，
                可直接传给 Z-Image transformer 作为 prompt_embeds。

        注意：这里不能 detach，也不能 no_grad。Z-03 loss 需要沿着
        E_safe -> denoise -> latent -> Z-03 反传回 adapter 参数。
        """

        if not prompt_embeds:
            raise ValueError("prompt_embeds 不能为空")

        device = prompt_embeds[0].device
        risk_tensor = self._normalize_risk_ids(risk_ids, len(prompt_embeds), device)
        if risk_tensor.numel() != len(prompt_embeds):
            raise ValueError("risk_ids 长度必须与 prompt_embeds batch 大小一致")
        token_mask_list = normalize_token_masks(token_masks, len(prompt_embeds))

        safe_embeds: list[torch.Tensor] = []
        gate_values_by_block: list[list[torch.Tensor]] = [[] for _ in range(self.adapter_depth)]
        token_gate_tensors_by_sample: list[list[torch.Tensor]] = []
        for index, embed in enumerate(prompt_embeds):
            if embed.ndim != 2:
                raise ValueError(f"每个 prompt embedding 必须是 [seq_len, dim]，当前 shape={tuple(embed.shape)}")
            if embed.shape[-1] != self.embedding_dim:
                raise ValueError(f"embedding dim 不匹配，期望 {self.embedding_dim}，实际 {embed.shape[-1]}")

            original_dtype = embed.dtype
            # embed: [T_i, D]。如果 token_mask 存在，x 只包含 user content
            # token，shape [T_content_i, D]；这样 chat template/special token
            # 不会进入 adapter 内部计算。
            x = select_adapter_input_tokens(embed, token_mask_list[index])

            condition = None
            if self.risk_embedding is not None:
                # condition: [1, D]，通过 broadcast 加到每个 token 上，
                # 使同一个 prompt 内所有 token 使用同一个 risk 条件。
                condition = self.risk_embedding(risk_tensor[index]).view(1, -1).float()
                condition = condition * float(self.runtime_condition_scale)

            # hidden: [T_i, D] 或 [T_content_i, D]。多个 block 时，每个 block
            # 都在当前 hidden 上预测一个逐 token delta，再做 residual 更新。
            hidden = x
            sample_token_gates: list[torch.Tensor] = []
            residual_multiplier = self.runtime_risk_residual_scale(
                risk_tensor[index],
                dtype=hidden.dtype,
            )
            for block_index in range(self.adapter_depth):
                hidden, gate = self._apply_delta_block(
                    hidden,
                    condition,
                    block_index=block_index,
                    residual_multiplier=residual_multiplier,
                )
                if gate is not None:
                    gate_values_by_block[block_index].append(gate.detach().float().mean())
                    if self.gate_type == "token":
                        sample_token_gates.append(gate.detach().float().view(-1).cpu())
            token_gate_tensors_by_sample.append(sample_token_gates)

            # 如果启用了 token_mask，这里把 [T_content_i, D] scatter 回完整
            # [T_i, D]；非 user content token 保持原 embedding。
            # 最后 cast 回原 dtype，兼容 bf16 Z-Image transformer；cast 操作本身
            # 仍保留梯度路径。
            hidden = apply_token_mask_to_safe_embed(embed.float(), hidden, token_mask_list[index])
            safe_embeds.append(hidden.to(dtype=original_dtype))

        self._last_gate_values = [
            float(torch.stack(block_values).mean().cpu())
            for block_values in gate_values_by_block
            if block_values
        ]
        self._last_token_gate_tensors = token_gate_tensors_by_sample
        return safe_embeds


def embedding_delta_stats(
    original_embeds: Sequence[torch.Tensor],
    safe_embeds: Sequence[torch.Tensor],
) -> dict[str, float]:
    """计算 embedding 修正幅度，主要用于日志。

    Args:
        original_embeds: list 长度 B，第 i 项 shape [T_i, D]。
        safe_embeds: list 长度 B，第 i 项 shape [T_i, D]。

    这里返回 Python float，所以只能用于 logging，不参与反传。
    """

    if len(original_embeds) != len(safe_embeds):
        raise ValueError("original_embeds 和 safe_embeds 长度不一致")

    means = []
    max_values = []
    for orig, safe in zip(original_embeds, safe_embeds):
        delta = (safe.float() - orig.float()).reshape(-1)
        means.append(delta.pow(2).mean().sqrt().detach())
        max_values.append(delta.abs().max().detach())

    return {
        "delta_rms": float(torch.stack(means).mean().cpu()),
        "delta_abs_max": float(torch.stack(max_values).max().cpu()),
    }


def freeze_module(module: nn.Module) -> nn.Module:
    """冻结模块参数并切到 eval。

    冻结参数不等于禁止梯度图。训练 denoise 时不能包 no_grad，因为输入
    prompt_embeds 需要梯度；这里只是避免更新基座模型参数。
    """

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
