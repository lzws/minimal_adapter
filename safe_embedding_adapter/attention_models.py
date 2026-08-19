"""带 attention 的 prompt embedding adapter 结构。

这个文件只定义模型结构，不改现有训练入口的默认行为。attention adapter 都保持和
SafeEmbeddingAdapter 相同的外部接口：

    input:
        prompt_embeds: list[Tensor]，长度 B；第 i 个 tensor shape [T_i, D]
        risk_ids:      list[str/int] 或 Tensor，长度/shape [B]
    output:
        safe_embeds:   list[Tensor]，长度 B；第 i 个 tensor shape [T_i, D]

其中 T_i 是第 i 条 prompt 的有效 token 数，D 是 Z-Image text encoder hidden dim。
由于 Z-Image pipeline 原生接受变长 prompt embeddings，这里继续逐条 prompt 处理，
不做 padding 和 attention mask，避免引入额外的对齐逻辑。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .model import (
    RISK_NAME_TO_ID,
    apply_token_mask_to_safe_embed,
    normalize_token_masks,
    select_adapter_input_tokens,
)


IP_CONDITION_TEXTS = {
    "ip_snow_white": "Snow White",
    "ip_doraemon": "Doraemon",
    "ip_minion": "Minion",
    "ip_elsa": "Elsa",
    "ip_spongebob": "SpongeBob SquarePants",
}

# 编号必须和 Z-03 IP 分类头保持一致：0-4 是五个目标 IP，5 是“其它”。
# 这个映射是 classifier condition 的语义来源，不依赖目标 IP 的文本 embedding。
CLASSIFIER_CONDITION_ID_BY_NAME = {
    "ip_snow_white": 0,
    "ip_doraemon": 1,
    "ip_minion": 2,
    "ip_elsa": 3,
    "ip_spongebob": 4,
}


def _default_bottleneck_dim(embedding_dim: int) -> int:
    """给 residual MLP 选择一个保守 bottleneck 维度。"""

    return max(16, embedding_dim // 4)


def _default_attention_dim(bottleneck_dim: int) -> int:
    """attention 维度不宜过大，否则显存和语义漂移风险都会上升。"""

    return min(512, max(64, bottleneck_dim))


def _zero_init_linear(linear: nn.Linear) -> None:
    """把 residual 输出层置零，让 adapter 从近似 identity 开始训练。"""

    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class RiskConditionMixin:
    """复用 risk condition embedding 和 risk_ids 解析逻辑。

    子类需要在 __init__ 里设置：
        self.use_risk_condition: bool
        self.embedding_dim: int
        self.risk_embedding: nn.Embedding | None
    """

    use_risk_condition: bool
    embedding_dim: int
    risk_embedding: nn.Embedding | None

    def _normalize_risk_ids(
        self,
        risk_ids: Sequence[int | str] | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """把 risk 条件统一成 shape [B] 的 long tensor。"""

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
        if len(normalized) != batch_size:
            raise ValueError("risk_ids 长度必须与 prompt_embeds batch 大小一致")
        return torch.tensor(normalized, dtype=torch.long, device=device)

    def _condition_for_sample(self, risk_index: torch.Tensor) -> torch.Tensor | None:
        """返回单条样本的 risk condition，shape [1, D]。"""

        if self.risk_embedding is None:
            return None
        return self.risk_embedding(risk_index).view(1, -1).float()


class RiskQueryAttentionGateBlock(nn.Module):
    """用 risk condition 作为 query 的 token attention gate block。

    输入:
        hidden:    [T, D]，当前 prompt token embeddings。
        condition: [1, D]，当前风险条件 embedding；例如 ip_doraemon。

    输出:
        delta: [T, D]，逐 token residual 方向。
        gate:  [T, 1]，逐 token 编辑强度，范围 (0, 1)。

    结构:
        q = W_q(condition)          # [1, A]
        k = W_k(LN(hidden + cond))  # [T, A]
        gate = sigmoid(k q^T / sqrt(A) + bias)
        delta = MLP(LN(hidden + cond))

    直觉:
        gate 越大，说明该 token 与当前 risk condition 越相关，adapter 对它的
        residual 修改越强。这样比普通 token gate 更容易把修改集中到目标 IP token
        或与目标 IP 强相关的 token 上。
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int,
        attention_dim: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim 必须大于 0")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init 必须在 (0, 1) 之间")

        self.embedding_dim = int(embedding_dim)
        self.attention_dim = int(attention_dim)
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.condition_norm = nn.LayerNorm(embedding_dim)
        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)

        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_logit)))
        self.gate_dropout = nn.Dropout(dropout)

        self.delta_net = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embedding_dim),
        )

    def zero_init_output(self) -> None:
        """只把 delta 输出置零，不破坏 gate 的初始分布。"""

        _zero_init_linear(self.delta_net[-1])

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError(f"hidden 期望 shape [T, D]，实际 {tuple(hidden.shape)}")

        block_input = hidden + condition if condition is not None else hidden
        token_features = self.token_norm(block_input)

        if condition is None:
            # 无 risk condition 时，用 prompt mean 作为 query 的退化版本。
            query_source = token_features.mean(dim=0, keepdim=True)
        else:
            query_source = self.condition_norm(condition)

        query = self.query_proj(query_source)  # [1, A]
        key = self.key_proj(token_features)  # [T, A]
        gate_logits = key @ query.transpose(0, 1)
        gate_logits = gate_logits / math.sqrt(float(self.attention_dim))
        gate = torch.sigmoid(gate_logits + self.gate_bias)  # [T, 1]
        gate = self.gate_dropout(gate)

        delta = self.delta_net(token_features)  # [T, D]
        return delta, gate


class RiskQueryAttentionGateAdapter(RiskConditionMixin, nn.Module):
    """risk-query attention gate adapter。

    这是用于缓解“误伤无关概念”的优先结构。它仍然使用 token-wise residual MLP
    预测 delta，但每个 token 的 delta 会乘上一个由 risk condition query 计算出的
    attention gate：

        E'_i = E_i + residual_scale * gate_i * delta_i

    Shapes:
        单条 prompt 输入 E: [T, D]
        condition r:       [1, D]
        gate:              [T, 1]
        delta:             [T, D]
        输出 E':           [T, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int | None = None,
        attention_dim: int | None = None,
        adapter_depth: int = 1,
        gate_init: float = 0.2,
        residual_scale: float = 0.5,
        dropout: float = 0.0,
        use_risk_condition: bool = True,
        num_risk_types: int = len(RISK_NAME_TO_ID),
        clamp_delta: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        if adapter_depth <= 0:
            raise ValueError("adapter_depth 必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")

        bottleneck_dim = bottleneck_dim or _default_bottleneck_dim(embedding_dim)
        attention_dim = attention_dim or _default_attention_dim(bottleneck_dim)
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.attention_dim = int(attention_dim)
        self.adapter_depth = int(adapter_depth)
        self.gate_init = float(gate_init)
        self.residual_scale = float(residual_scale)
        self.dropout = float(dropout)
        self.use_risk_condition = bool(use_risk_condition)
        self.clamp_delta = bool(clamp_delta)
        self.zero_init = bool(zero_init)

        if self.use_risk_condition:
            self.risk_embedding = nn.Embedding(num_risk_types, self.embedding_dim)
        else:
            self.risk_embedding = None

        self.blocks = nn.ModuleList(
            [
                RiskQueryAttentionGateBlock(
                    self.embedding_dim,
                    self.bottleneck_dim,
                    self.attention_dim,
                    self.dropout,
                    self.gate_init,
                )
                for _ in range(self.adapter_depth)
            ]
        )
        self.delta_activation = nn.Tanh()
        self._last_gate_values: list[float] = []
        self._last_gate_tensors: list[torch.Tensor] = []
        if self.zero_init:
            for block in self.blocks:
                block.zero_init_output()

    def gate_values(self) -> list[float]:
        """返回每个 block 最近一次 forward 的平均 gate，仅用于日志。"""

        return list(self._last_gate_values)

    def gate_sparse_loss(self) -> torch.Tensor | None:
        """返回最近一次 forward 的 gate 稀疏正则项。

        训练脚本后续可以把这个项乘以 w_gate_sparse 加入总 loss：

            L_gate_sparse = mean(gate)

        该方法保留梯度图，因此只应在同一个 forward 对应的训练 step 内调用。
        """

        if not self._last_gate_tensors:
            return None
        return torch.stack([gate.float().mean() for gate in self._last_gate_tensors]).mean()

    def forward(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        risk_ids: Sequence[int | str] | torch.Tensor | None = None,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if not prompt_embeds:
            raise ValueError("prompt_embeds 不能为空")

        device = prompt_embeds[0].device
        risk_tensor = self._normalize_risk_ids(risk_ids, len(prompt_embeds), device)
        if risk_tensor.numel() != len(prompt_embeds):
            raise ValueError("risk_ids 长度必须与 prompt_embeds batch 大小一致")
        token_mask_list = normalize_token_masks(token_masks, len(prompt_embeds))

        safe_embeds: list[torch.Tensor] = []
        gate_values_by_block: list[list[torch.Tensor]] = [[] for _ in range(self.adapter_depth)]
        gate_tensors: list[torch.Tensor] = []
        for sample_index, embed in enumerate(prompt_embeds):
            if embed.ndim != 2:
                raise ValueError(f"每个 prompt embedding 必须是 [T, D]，实际 {tuple(embed.shape)}")
            if embed.shape[-1] != self.embedding_dim:
                raise ValueError(f"embedding dim 不匹配，期望 {self.embedding_dim}，实际 {embed.shape[-1]}")

            original_dtype = embed.dtype
            # token_masks 启用时，hidden 只包含 user content token：
            # [T_content, D]。因此 risk-query gate 不会看到 chat template/special token。
            hidden = select_adapter_input_tokens(embed, token_mask_list[sample_index])
            condition = self._condition_for_sample(risk_tensor[sample_index])  # [1, D] or None

            for block_index, block in enumerate(self.blocks):
                delta, gate = block(hidden, condition)
                if self.clamp_delta:
                    delta = self.delta_activation(delta)
                hidden = hidden + self.residual_scale * gate.to(hidden.dtype) * delta
                gate_values_by_block[block_index].append(gate.detach().float().mean())
                gate_tensors.append(gate)

            hidden = apply_token_mask_to_safe_embed(embed.float(), hidden, token_mask_list[sample_index])
            safe_embeds.append(hidden.to(dtype=original_dtype))

        self._last_gate_values = [
            float(torch.stack(block_values).mean().cpu())
            for block_values in gate_values_by_block
            if block_values
        ]
        self._last_gate_tensors = gate_tensors
        return safe_embeds


class ZImageAdaLNClassifierConditionAdapter(RiskConditionMixin, nn.Module):
    """使用 Z-03 classifier class id 作为 condition 的 Z-Image AdaLN adapter。

    这个版本不使用目标 IP 的文本 embedding 定义擦除目标。Z-03 虽然输出 6 类，
    但 adapter 只为 5 个待擦除 IP 建立 condition；Z-03 的“其它”类不是擦除目标，
    因此不进入 adapter 的 one-hot condition：

        ip_snow_white  -> one_hot(0, 5) -> Z-03 class 0
        ip_doraemon    -> one_hot(1, 5) -> Z-03 class 1
        ip_minion      -> one_hot(2, 5) -> Z-03 class 2
        ip_elsa        -> one_hot(3, 5) -> Z-03 class 3
        ip_spongebob   -> one_hot(4, 5) -> Z-03 class 4

    one-hot 经过可学习的 class_condition_mlp 投影到 [1, D]，再作为
    ZImageAdaLNTextConditionBlock 的 AdaLN condition。Z-03 的 latent logits
    仍然是实际的风险监督信号，class condition 只用于选择当前目标类别对应
    的 adapter 修正策略。

    单条 prompt 的数据流:

        prompt embedding E:       [T, D]
        classifier one-hot u:     [5]
        class condition c:        [1, D]
        block hidden states:      [T, d]
        token gate:               [T, 1]
        delta:                    [T, D]
        safe embedding E_safe:    [T, D]

    注意:
        - risk_ids 可以传目标 classifier class id 整数 0-4；
        - 也可以传 ip_snow_white/ip_doraemon 等 IP condition name；
        - 通用的 "ip" 不对应单个 one-hot 类别，因此会明确报错，避免
          把未知 IP 样本错误地训练成某一个固定类别；
        - token gate 只在最终 residual 上使用一次。
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        adapter_depth: int = 1,
        gate_init: float = 0.2,
        residual_scale: float = 0.2,
        dropout: float = 0.0,
        use_risk_condition: bool = True,
        num_risk_types: int = len(RISK_NAME_TO_ID),
        num_classifier_classes: int = 5,
        classifier_condition_hidden_dim: int | None = None,
        clamp_delta: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于 0")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if adapter_depth <= 0:
            raise ValueError("adapter_depth 必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")
        if not use_risk_condition:
            raise ValueError(
                "ZImageAdaLNClassifierConditionAdapter 必须使用 classifier class condition，"
                "不能设置 use_risk_condition=False"
            )
        if num_classifier_classes != len(CLASSIFIER_CONDITION_ID_BY_NAME):
            raise ValueError(
                "num_classifier_classes 必须为 5；adapter condition 只覆盖五个待擦除 IP，"
                "不包含 Z-03 的“其它”类"
            )

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.adapter_depth = int(adapter_depth)
        self.gate_init = float(gate_init)
        self.residual_scale = float(residual_scale)
        self.dropout = float(dropout)
        self.use_risk_condition = True
        # 保留统一 AdapterConfig 的字段，便于训练 checkpoint 兼容；新模型的
        # condition 由 classifier class 数量决定，因此 num_risk_types 不参与计算。
        self.num_risk_types = int(num_risk_types)
        self.num_classifier_classes = int(num_classifier_classes)
        self.classifier_condition_hidden_dim = int(
            classifier_condition_hidden_dim
            or max(64, min(256, embedding_dim // 2))
        )
        self.clamp_delta = bool(clamp_delta)
        self.zero_init = bool(zero_init)

        # one-hot [K] -> condition [D]。这个投影是可学习的，但不改变
        # classifier class id 的定义；风险监督仍来自冻结的 Z-03 logits。
        self.class_condition_mlp = nn.Sequential(
            nn.Linear(self.num_classifier_classes, self.classifier_condition_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.classifier_condition_hidden_dim, self.embedding_dim),
        )

        self.blocks = nn.ModuleList(
            [
                ZImageAdaLNTextConditionBlock(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    condition_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    gate_init=gate_init,
                )
                for _ in range(adapter_depth)
            ]
        )
        self.delta_activation = nn.Tanh()
        self._last_gate_values: list[float] = []
        self._last_gate_tensors: list[torch.Tensor] = []
        if self.zero_init:
            for block in self.blocks:
                block.zero_init_output(gate_init)

    def _normalize_classifier_ids(
        self,
        risk_ids: Sequence[int | str] | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """将 classifier class id/name 统一为 shape [B] 的 long tensor。"""

        if risk_ids is None:
            raise ValueError(
                "ZImageAdaLNClassifierConditionAdapter.forward 必须提供 risk_ids，"
                "risk_ids 应为 Z-03 class id 或 ip_* condition name"
            )
        if isinstance(risk_ids, torch.Tensor):
            class_ids = risk_ids.to(device=device, dtype=torch.long).flatten()
        else:
            normalized_ids: list[int] = []
            for item in risk_ids:
                if isinstance(item, str):
                    normalized_name = item.strip().casefold()
                    if normalized_name not in CLASSIFIER_CONDITION_ID_BY_NAME:
                        raise ValueError(
                            f"未知 classifier condition: {item!r}。"
                            "请传 0-5 的 class id 或 ip_snow_white/ip_doraemon 等名称；"
                            "通用 ip 不能映射为单个 one-hot 类别。"
                        )
                    normalized_ids.append(CLASSIFIER_CONDITION_ID_BY_NAME[normalized_name])
                else:
                    normalized_ids.append(int(item))
            class_ids = torch.tensor(normalized_ids, dtype=torch.long, device=device)

        if class_ids.numel() != batch_size:
            raise ValueError(
                "classifier class id 数量必须与 prompt embedding batch 大小一致："
                f"expected={batch_size}, actual={class_ids.numel()}"
            )
        if bool((class_ids < 0).any()) or bool((class_ids >= self.num_classifier_classes).any()):
            raise ValueError(
                "adapter classifier condition id 超出范围："
                f"expected=[0, {self.num_classifier_classes - 1}], "
                f"actual={class_ids.tolist()}"
            )
        return class_ids

    def _condition_from_class_id(self, class_id: torch.Tensor) -> torch.Tensor:
        """将一个 class id 转成 block 使用的 [1, D] condition。"""

        one_hot = torch.nn.functional.one_hot(
            class_id.view(1),
            num_classes=self.num_classifier_classes,
        ).float()
        return self.class_condition_mlp(one_hot)  # [1, D]

    def gate_values(self) -> list[float]:
        """返回每个 block 最近一次 forward 的平均 token gate。"""

        return list(self._last_gate_values)

    def gate_sparse_loss(self) -> torch.Tensor | None:
        """返回最近一次 forward 的 gate 平均值，供可选稀疏正则使用。"""

        if not self._last_gate_tensors:
            return None
        return torch.stack([gate.float().mean() for gate in self._last_gate_tensors]).mean()

    def forward(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        risk_ids: Sequence[int | str] | torch.Tensor | None = None,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """根据 classifier condition 修正一批变长 prompt embeddings。"""

        if not prompt_embeds:
            raise ValueError("prompt_embeds 不能为空")

        device = prompt_embeds[0].device
        class_ids = self._normalize_classifier_ids(risk_ids, len(prompt_embeds), device)
        token_mask_list = normalize_token_masks(token_masks, len(prompt_embeds))

        safe_embeds: list[torch.Tensor] = []
        gate_values_by_block: list[list[torch.Tensor]] = [[] for _ in range(self.adapter_depth)]
        gate_tensors: list[torch.Tensor] = []
        for sample_index, embed in enumerate(prompt_embeds):
            if embed.ndim != 2:
                raise ValueError(f"每个 prompt embedding 必须是 [T, D]，实际 {tuple(embed.shape)}")
            if embed.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"embedding dim 不匹配，期望 {self.embedding_dim}，实际 {embed.shape[-1]}"
                )

            original_dtype = embed.dtype
            # token_masks 启用时，hidden 只包含 user content token：
            # [T_content, D]。AdaLN attention 的 Q/K/V 也只来自这段子序列。
            hidden = select_adapter_input_tokens(embed, token_mask_list[sample_index])
            condition = self._condition_from_class_id(class_ids[sample_index])  # [1, D]
            for block_index, block in enumerate(self.blocks):
                delta, token_gate = block(hidden, condition)
                if self.clamp_delta:
                    delta = self.delta_activation(delta)
                # token_gate 只在最终 residual 上使用一次。
                hidden = hidden + self.residual_scale * token_gate.to(hidden.dtype) * delta
                gate_values_by_block[block_index].append(token_gate.detach().float().mean())
                gate_tensors.append(token_gate)
            hidden = apply_token_mask_to_safe_embed(embed.float(), hidden, token_mask_list[sample_index])
            safe_embeds.append(hidden.to(dtype=original_dtype))

        self._last_gate_values = [
            float(torch.stack(block_values).mean().cpu())
            for block_values in gate_values_by_block
            if block_values
        ]
        self._last_gate_tensors = gate_tensors
        return safe_embeds


class RiskQueryFiLMGateBlock(nn.Module):
    """risk-query gate + FiLM bottleneck delta block。

    这个 block 是比 RiskQueryAttentionGateBlock 更保守的版本：condition 不再
    broadcast 加到 token embedding 上，而是只通过两个受控路径生效：

    1. gate 路径：condition 作为 query，和 token key 做 attention，决定每个 token
       的编辑强度。
    2. delta 路径：token 先降维到 bottleneck，再由 condition 产生 gamma/beta 做
       FiLM 调制，决定当前 IP 的修正策略。

    输入:
        hidden:    [T, D]，当前 prompt token embeddings。
        condition: [1, D]，当前风险条件 embedding；例如 ip_spongebob。

    输出:
        delta: [T, D]，逐 token residual 方向。
        gate:  [T, 1]，逐 token 编辑强度。

    结构:
        token_features = LN(hidden)                  # [T, D]

        q = W_q(LN(condition))                       # [1, A]
        k = W_k(token_features)                      # [T, A]
        gate = sigmoid(k q^T / sqrt(A) + bias)       # [T, 1]

        h = W_down(token_features)                   # [T, d]
        gamma, beta = W_film(LN(condition))          # [1, d], [1, d]
        h = h * (1 + gamma) + beta                   # [T, d]
        delta = W_up(GELU(h))                        # [T, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int,
        attention_dim: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim 必须大于 0")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim 必须大于 0")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init 必须在 (0, 1) 之间")

        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.attention_dim = int(attention_dim)

        self.token_norm = nn.LayerNorm(embedding_dim)
        self.condition_norm = nn.LayerNorm(embedding_dim)

        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_logit)))
        self.gate_dropout = nn.Dropout(dropout)

        self.down_proj = nn.Linear(embedding_dim, bottleneck_dim)
        self.film_proj = nn.Linear(embedding_dim, bottleneck_dim * 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, embedding_dim)

    def zero_init_output(self) -> None:
        """从 identity 附近开始：delta 输出和 FiLM 调制都初始化为 0。"""

        _zero_init_linear(self.up_proj)
        _zero_init_linear(self.film_proj)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError(f"hidden 期望 shape [T, D]，实际 {tuple(hidden.shape)}")

        # token_features: [T, D]。注意这里不加 condition，避免把目标 IP
        # 向量直接 broadcast 注入所有 token。
        token_features = self.token_norm(hidden)
        if condition is None:
            # 无 risk condition 时，用 prompt mean 作为 query / FiLM source 的退化版本。
            condition_features = token_features.mean(dim=0, keepdim=True)
        else:
            condition_features = self.condition_norm(condition)

        # gate: [T, 1]，只决定哪些 token 值得改。
        query = self.query_proj(condition_features)  # [1, A]
        key = self.key_proj(token_features)  # [T, A]
        gate_logits = key @ query.transpose(0, 1)
        gate_logits = gate_logits / math.sqrt(float(self.attention_dim))
        gate = torch.sigmoid(gate_logits + self.gate_bias)
        gate = self.gate_dropout(gate)

        # h: [T, d]。condition 只在低维 bottleneck 空间调制修正策略，
        # 不直接进入原始 embedding 空间。
        h = self.down_proj(token_features)
        gamma_beta = self.film_proj(condition_features)  # [1, 2d]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = self.dropout(self.activation(h))
        delta = self.up_proj(h)  # [T, D]
        return delta, gate


class RiskQueryFiLMGateAdapter(RiskConditionMixin, nn.Module):
    """Risk-query FiLM gate adapter。

    这是为“无关概念被改成目标概念”问题准备的保守 attention adapter。
    它和 RiskQueryAttentionGateAdapter 的区别是：condition 不再以
    hidden + condition 的形式进入所有 token，而是：

    - 用 condition query 计算 token gate，决定改哪些 token；
    - 用 condition 在 bottleneck 空间做 FiLM，决定当前 IP 的 delta 策略；
    - delta 的 token 输入仍然只来自原始 token features。

    Shapes:
        单条 prompt 输入 E: [T, D]
        condition r:       [1, D]
        gate:              [T, 1]
        bottleneck h:      [T, d]
        delta:             [T, D]
        输出 E':           [T, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        bottleneck_dim: int | None = None,
        attention_dim: int | None = None,
        adapter_depth: int = 1,
        gate_init: float = 0.2,
        residual_scale: float = 0.5,
        dropout: float = 0.0,
        use_risk_condition: bool = True,
        num_risk_types: int = len(RISK_NAME_TO_ID),
        clamp_delta: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        if adapter_depth <= 0:
            raise ValueError("adapter_depth 必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")

        bottleneck_dim = bottleneck_dim or _default_bottleneck_dim(embedding_dim)
        attention_dim = attention_dim or _default_attention_dim(bottleneck_dim)
        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.attention_dim = int(attention_dim)
        self.adapter_depth = int(adapter_depth)
        self.gate_init = float(gate_init)
        self.residual_scale = float(residual_scale)
        self.dropout = float(dropout)
        self.use_risk_condition = bool(use_risk_condition)
        self.clamp_delta = bool(clamp_delta)
        self.zero_init = bool(zero_init)

        if self.use_risk_condition:
            self.risk_embedding = nn.Embedding(num_risk_types, self.embedding_dim)
        else:
            self.risk_embedding = None

        self.blocks = nn.ModuleList(
            [
                RiskQueryFiLMGateBlock(
                    self.embedding_dim,
                    self.bottleneck_dim,
                    self.attention_dim,
                    self.dropout,
                    self.gate_init,
                )
                for _ in range(self.adapter_depth)
            ]
        )
        self.delta_activation = nn.Tanh()
        self._last_gate_values: list[float] = []
        self._last_gate_tensors: list[torch.Tensor] = []
        if self.zero_init:
            for block in self.blocks:
                block.zero_init_output()

    def gate_values(self) -> list[float]:
        """返回每个 block 最近一次 forward 的平均 gate，仅用于日志。"""

        return list(self._last_gate_values)

    def gate_sparse_loss(self) -> torch.Tensor | None:
        """返回最近一次 forward 的 gate 稀疏正则项，供训练脚本后续接入。"""

        if not self._last_gate_tensors:
            return None
        return torch.stack([gate.float().mean() for gate in self._last_gate_tensors]).mean()

    def forward(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        risk_ids: Sequence[int | str] | torch.Tensor | None = None,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if not prompt_embeds:
            raise ValueError("prompt_embeds 不能为空")

        device = prompt_embeds[0].device
        risk_tensor = self._normalize_risk_ids(risk_ids, len(prompt_embeds), device)
        if risk_tensor.numel() != len(prompt_embeds):
            raise ValueError("risk_ids 长度必须与 prompt_embeds batch 大小一致")
        token_mask_list = normalize_token_masks(token_masks, len(prompt_embeds))

        safe_embeds: list[torch.Tensor] = []
        gate_values_by_block: list[list[torch.Tensor]] = [[] for _ in range(self.adapter_depth)]
        gate_tensors: list[torch.Tensor] = []
        for sample_index, embed in enumerate(prompt_embeds):
            if embed.ndim != 2:
                raise ValueError(f"每个 prompt embedding 必须是 [T, D]，实际 {tuple(embed.shape)}")
            if embed.shape[-1] != self.embedding_dim:
                raise ValueError(f"embedding dim 不匹配，期望 {self.embedding_dim}，实际 {embed.shape[-1]}")

            original_dtype = embed.dtype
            # token_masks 启用时，hidden 只包含 user content token：
            # [T_content, D]。FiLM/gate 路径都不会接触 template token。
            hidden = select_adapter_input_tokens(embed, token_mask_list[sample_index])
            condition = self._condition_for_sample(risk_tensor[sample_index])  # [1, D] or None

            for block_index, block in enumerate(self.blocks):
                delta, gate = block(hidden, condition)
                if self.clamp_delta:
                    delta = self.delta_activation(delta)
                hidden = hidden + self.residual_scale * gate.to(hidden.dtype) * delta
                gate_values_by_block[block_index].append(gate.detach().float().mean())
                gate_tensors.append(gate)

            hidden = apply_token_mask_to_safe_embed(embed.float(), hidden, token_mask_list[sample_index])
            safe_embeds.append(hidden.to(dtype=original_dtype))

        self._last_gate_values = [
            float(torch.stack(block_values).mean().cpu())
            for block_values in gate_values_by_block
            if block_values
        ]
        self._last_gate_tensors = gate_tensors
        return safe_embeds


class BottleneckSelfAttentionBlock(nn.Module):
    """低维 bottleneck self-attention residual block。

    输入:
        hidden:    [T, D]
        condition: [1, D] 或 None

    内部:
        先把 [T, D] 降到 [T, A]，在低维空间做 self-attention 和 FFN，
        最后投回 [T, D] 作为 delta。

    这个结构比 risk-query gate 表达力更强，可以建模 token 间组合关系；
    但也更容易把目标 IP 梯度传播到风格/场景 token，因此建议配合
    unsafe_only risk loss、related identity set 和较小 residual_scale 使用。
    """

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim 必须大于 0")
        if num_heads <= 0:
            raise ValueError("num_heads 必须大于 0")
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim 必须能被 num_heads 整除")

        self.input_norm = nn.LayerNorm(embedding_dim)
        self.down_proj = nn.Linear(embedding_dim, attention_dim)
        self.attn_norm = nn.LayerNorm(attention_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(attention_dim)
        hidden_dim = int(attention_dim * ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attention_dim),
            nn.Dropout(dropout),
        )
        self.up_proj = nn.Linear(attention_dim, embedding_dim)

    def zero_init_output(self) -> None:
        """把最终上投影置零，让整个 attention block 初始输出 delta=0。"""

        _zero_init_linear(self.up_proj)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError(f"hidden 期望 shape [T, D]，实际 {tuple(hidden.shape)}")

        block_input = hidden + condition if condition is not None else hidden
        token_features = self.input_norm(block_input)  # [T, D]
        states = self.down_proj(token_features).unsqueeze(0)  # [1, T, A]

        attn_input = self.attn_norm(states)
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )
        states = states + attn_output
        states = states + self.ffn(self.ffn_norm(states))

        delta = self.up_proj(states.squeeze(0))  # [T, D]
        return delta


class BottleneckSelfAttentionAdapter(RiskConditionMixin, nn.Module):
    """低维 self-attention residual adapter。

    结构:
        E_cond = E + risk_condition
        H = W_down(LN(E_cond))
        H = H + MHA(LN(H))
        H = H + FFN(LN(H))
        delta = W_up(H)
        E' = E + residual_scale * delta

    Shapes:
        单条 prompt 输入 E: [T, D]
        attention hidden H: [T, A]
        delta:             [T, D]
        输出 E':           [T, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int | None = None,
        num_heads: int = 4,
        adapter_depth: int = 1,
        residual_scale: float = 0.2,
        dropout: float = 0.0,
        use_risk_condition: bool = True,
        num_risk_types: int = len(RISK_NAME_TO_ID),
        clamp_delta: bool = True,
        zero_init: bool = True,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        if adapter_depth <= 0:
            raise ValueError("adapter_depth 必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")

        attention_dim = attention_dim or 256
        self.embedding_dim = int(embedding_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.adapter_depth = int(adapter_depth)
        self.residual_scale = float(residual_scale)
        self.dropout = float(dropout)
        self.use_risk_condition = bool(use_risk_condition)
        self.clamp_delta = bool(clamp_delta)
        self.zero_init = bool(zero_init)
        self.ffn_multiplier = int(ffn_multiplier)

        if self.use_risk_condition:
            self.risk_embedding = nn.Embedding(num_risk_types, self.embedding_dim)
        else:
            self.risk_embedding = None

        self.blocks = nn.ModuleList(
            [
                BottleneckSelfAttentionBlock(
                    self.embedding_dim,
                    self.attention_dim,
                    self.num_heads,
                    self.dropout,
                    self.ffn_multiplier,
                )
                for _ in range(self.adapter_depth)
            ]
        )
        self.delta_activation = nn.Tanh()
        if self.zero_init:
            for block in self.blocks:
                block.zero_init_output()

    def gate_values(self) -> list[float]:
        """保持和 SafeEmbeddingAdapter 的日志接口一致。"""

        return []

    def forward(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        risk_ids: Sequence[int | str] | torch.Tensor | None = None,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if not prompt_embeds:
            raise ValueError("prompt_embeds 不能为空")

        device = prompt_embeds[0].device
        risk_tensor = self._normalize_risk_ids(risk_ids, len(prompt_embeds), device)
        if risk_tensor.numel() != len(prompt_embeds):
            raise ValueError("risk_ids 长度必须与 prompt_embeds batch 大小一致")
        token_mask_list = normalize_token_masks(token_masks, len(prompt_embeds))

        safe_embeds: list[torch.Tensor] = []
        for sample_index, embed in enumerate(prompt_embeds):
            if embed.ndim != 2:
                raise ValueError(f"每个 prompt embedding 必须是 [T, D]，实际 {tuple(embed.shape)}")
            if embed.shape[-1] != self.embedding_dim:
                raise ValueError(f"embedding dim 不匹配，期望 {self.embedding_dim}，实际 {embed.shape[-1]}")

            original_dtype = embed.dtype
            # token_masks 启用时，hidden 只包含 user content token：
            # [T_content, D]。self-attention 的 Q/K/V 不包含 template/special token。
            hidden = select_adapter_input_tokens(embed, token_mask_list[sample_index])
            condition = self._condition_for_sample(risk_tensor[sample_index])  # [1, D] or None

            for block in self.blocks:
                delta = block(hidden, condition)
                if self.clamp_delta:
                    delta = self.delta_activation(delta)
                hidden = hidden + self.residual_scale * delta

            hidden = apply_token_mask_to_safe_embed(embed.float(), hidden, token_mask_list[sample_index])
            safe_embeds.append(hidden.to(dtype=original_dtype))

        return safe_embeds


class ZImageAdaLNTextConditionBlock(nn.Module):
    """仿照 Z-Image 的 AdaLN 调制 block。

    Z-Image 原始 block 的核心形式是：

        mod = Linear(condition)                 # [1, 4d]
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4)
        h = h + gate_msa * Attention(Norm(h) * (1 + scale_msa))
        h = h + gate_mlp * FFN(Norm(h) * (1 + scale_mlp))

    本 adapter 使用 token_gate 控制最终 residual：

        h = h + gate_msa * Attention(...)
        h = h + gate_mlp * FFN(...)
        delta = UpProj(h)
        output = input + token_gate * delta

    condition 是目标概念的冻结文本 embedding，shape [1, D]；它只生成全局的
    AdaLN 分支调制参数。token_gate 决定当前 prompt 中哪些 token 最终接受 residual
    修改，从而避免目标概念 condition 直接把所有 token 同等改写。

    输入/输出 shape:
        hidden:    [T, d]
        condition: [1, D]
        delta:     [T, D_out]，这里 D_out=embedding_dim
        token_gate:[T, 1]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        condition_dim: int,
        num_heads: int,
        dropout: float,
        gate_init: float,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于 0")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if condition_dim <= 0:
            raise ValueError("condition_dim 必须大于 0")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init 必须在 (0, 1) 之间")

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(condition_dim)
        self.num_heads = int(num_heads)

        self.token_norm = nn.LayerNorm(embedding_dim)
        self.down_proj = nn.Linear(embedding_dim, hidden_dim)

        # token gate 使用目标概念 condition 作为 query，token hidden 作为 key。
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.query_proj = nn.Linear(condition_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.token_gate_bias = nn.Parameter(torch.tensor(float(gate_logit)))
        self.token_gate_dropout = nn.Dropout(dropout)

        # 对齐 Z-Image：condition -> 4 * hidden_dim。
        # chunk 顺序为 scale_msa, gate_msa, scale_mlp, gate_mlp。
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 4 * hidden_dim, bias=True),
        )

        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        ffn_hidden_dim = int(hidden_dim / 3 * 8)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.up_proj = nn.Linear(hidden_dim, embedding_dim)

    def zero_init_output(self, gate_init: float) -> None:
        """初始化为 identity 附近，同时保留 residual 分支的可训练梯度。

        Z-Image 原模型的 modulation 参数来自已训练基座；这里是新 adapter，不能
        直接复制其权重。因此：
        - scale 初始为 0，使 1+scale 初始为 1；
        - branch gate 初始为小正值，避免和 zero-init up_proj 一起完全死锁；
        - up_proj 初始为 0，使整个 adapter 初始输出 delta=0。
        """

        _zero_init_linear(self.up_proj)
        modulation_linear = self.adaLN_modulation[-1]
        nn.init.zeros_(modulation_linear.weight)
        nn.init.zeros_(modulation_linear.bias)
        gate_logit = math.atanh(0.1)
        with torch.no_grad():
            modulation_linear.bias[self.hidden_dim : 2 * self.hidden_dim].fill_(gate_logit)
            modulation_linear.bias[3 * self.hidden_dim : 4 * self.hidden_dim].fill_(gate_logit)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError(f"hidden 期望 shape [T, d]，实际 {tuple(hidden.shape)}")
        if condition.ndim != 2 or condition.shape[0] != 1:
            raise ValueError(f"condition 期望 shape [1, D]，实际 {tuple(condition.shape)}")

        token_features = self.token_norm(hidden)  # [T, D]
        states = self.down_proj(token_features)  # [T, d]
        condition_features = self.condition_norm(condition)  # [1, D]

        # token_gate: [T, 1]。
        query = self.query_proj(condition_features)  # [1, d]
        key = self.key_proj(states)  # [T, d]
        token_gate_logits = key @ query.transpose(0, 1)
        token_gate_logits = token_gate_logits / math.sqrt(float(self.hidden_dim))
        token_gate = torch.sigmoid(token_gate_logits + self.token_gate_bias)
        token_gate = self.token_gate_dropout(token_gate)

        # Z-Image AdaLN modulation: [1, 4d] -> four branch parameters。
        modulation = self.adaLN_modulation(condition_features)
        scale_msa, gate_msa, scale_mlp, gate_mlp = modulation.chunk(4, dim=-1)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)

        # Attention branch。condition 只调制 normalized states，不直接加到 token。
        attn_input = self.attention_norm(states) * scale_msa
        attn_output, _ = self.self_attn(
            attn_input.unsqueeze(0),
            attn_input.unsqueeze(0),
            attn_input.unsqueeze(0),
            need_weights=False,
        )
        states = states + gate_msa * attn_output.squeeze(0)

        # FFN branch，与 Z-Image 的第二个 AdaLN residual branch 对齐。
        ffn_input = self.ffn_norm(states) * scale_mlp
        ffn_output = self.feed_forward(ffn_input)
        states = states + gate_mlp * ffn_output

        delta = self.up_proj(states)  # [T, D]
        return delta, token_gate


class ZImageAdaLNTextConditionAdapter(RiskConditionMixin, nn.Module):
    """使用目标概念文本 embedding 作为 condition 的 Z-Image AdaLN adapter。

    训练开始时由训练脚本用 Z-Image text encoder 编码固定的目标概念文本，例如：

        Snow White
        Doraemon
        Minion
        Elsa
        SpongeBob SquarePants

    经过有效 token mean pooling 后得到 condition table：

        condition_embeddings: [N, D]

    该 table 是 frozen buffer，不参与训练。forward 时根据 risk_id 查表：

        condition_embeddings[risk_id]: [D]
        condition:                  [1, D]

    benign 样本由 trainer 随机分配一个 IP-specific risk_id，因此会随机使用一个
    目标概念 condition，但仍只通过 identity/embedding 约束保护 benign prompt。
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        adapter_depth: int = 1,
        gate_init: float = 0.2,
        residual_scale: float = 0.2,
        dropout: float = 0.0,
        use_risk_condition: bool = True,
        num_risk_types: int = len(RISK_NAME_TO_ID),
        clamp_delta: bool = True,
        zero_init: bool = True,
        condition_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于 0")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if adapter_depth <= 0:
            raise ValueError("adapter_depth 必须大于 0")
        if residual_scale <= 0:
            raise ValueError("residual_scale 必须大于 0")

        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.adapter_depth = int(adapter_depth)
        self.gate_init = float(gate_init)
        self.residual_scale = float(residual_scale)
        self.dropout = float(dropout)
        self.use_risk_condition = bool(use_risk_condition)
        self.clamp_delta = bool(clamp_delta)
        self.zero_init = bool(zero_init)
        self.num_risk_types = int(num_risk_types)

        if condition_embeddings is None:
            initial_conditions = torch.zeros(num_risk_types, embedding_dim, dtype=torch.float32)
        else:
            initial_conditions = torch.as_tensor(condition_embeddings, dtype=torch.float32).detach().clone()
            if initial_conditions.shape != (num_risk_types, embedding_dim):
                raise ValueError(
                    "condition_embeddings 期望 shape "
                    f"[{num_risk_types}, {embedding_dim}]，实际 {tuple(initial_conditions.shape)}"
                )
        self.register_buffer("condition_embeddings", initial_conditions, persistent=True)
        # 兼容 adapter_supports_risk_id 和现有 trainer 的 IP condition 检测。
        self.risk_embedding = None

        self.blocks = nn.ModuleList(
            [
                ZImageAdaLNTextConditionBlock(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    condition_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    gate_init=gate_init,
                )
                for _ in range(adapter_depth)
            ]
        )
        self.delta_activation = nn.Tanh()
        self._last_gate_values: list[float] = []
        self._last_gate_tensors: list[torch.Tensor] = []
        if zero_init:
            for block in self.blocks:
                block.zero_init_output(gate_init)

    @property
    def condition_dim(self) -> int:
        return int(self.condition_embeddings.shape[-1])

    def set_condition_embeddings(self, condition_embeddings: torch.Tensor) -> None:
        """设置训练启动时预计算好的 frozen target concept table。"""

        values = torch.as_tensor(condition_embeddings, dtype=torch.float32, device=self.condition_embeddings.device)
        if values.shape != self.condition_embeddings.shape:
            raise ValueError(
                "condition_embeddings shape 不匹配："
                f"expected={tuple(self.condition_embeddings.shape)} actual={tuple(values.shape)}"
            )
        with torch.no_grad():
            self.condition_embeddings.copy_(values.detach())

    def gate_values(self) -> list[float]:
        return list(self._last_gate_values)

    def gate_sparse_loss(self) -> torch.Tensor | None:
        if not self._last_gate_tensors:
            return None
        return torch.stack([gate.float().mean() for gate in self._last_gate_tensors]).mean()

    def forward(
        self,
        prompt_embeds: Sequence[torch.Tensor],
        risk_ids: Sequence[int | str] | torch.Tensor | None = None,
        token_masks: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        if not prompt_embeds:
            raise ValueError("prompt_embeds 不能为空")

        device = prompt_embeds[0].device
        risk_tensor = self._normalize_risk_ids(risk_ids, len(prompt_embeds), device)
        if risk_tensor.numel() != len(prompt_embeds):
            raise ValueError("risk_ids 长度必须与 prompt_embeds batch 大小一致")
        if bool((risk_tensor < 0).any()) or bool((risk_tensor >= self.condition_embeddings.shape[0]).any()):
            raise ValueError("risk_ids 超出 condition_embeddings 范围")
        token_mask_list = normalize_token_masks(token_masks, len(prompt_embeds))

        safe_embeds: list[torch.Tensor] = []
        gate_values_by_block: list[list[torch.Tensor]] = [[] for _ in range(self.adapter_depth)]
        gate_tensors: list[torch.Tensor] = []
        for sample_index, embed in enumerate(prompt_embeds):
            if embed.ndim != 2:
                raise ValueError(f"每个 prompt embedding 必须是 [T, D]，实际 {tuple(embed.shape)}")
            if embed.shape[-1] != self.embedding_dim:
                raise ValueError(f"embedding dim 不匹配，期望 {self.embedding_dim}，实际 {embed.shape[-1]}")

            original_dtype = embed.dtype
            # token_masks 启用时，hidden 只包含 user content token：
            # [T_content, D]。AdaLN attention 的 Q/K/V 只在 user content 内计算。
            hidden = select_adapter_input_tokens(embed, token_mask_list[sample_index])
            condition = self.condition_embeddings[risk_tensor[sample_index]].view(1, -1).float()
            for block_index, block in enumerate(self.blocks):
                delta, token_gate = block(hidden, condition)
                if self.clamp_delta:
                    delta = self.delta_activation(delta)
                # token_gate 只在最终 residual 上使用一次，保证最终输出只作用于
                # 被 gate 选中的 token，同时不削弱 block 内部的 Attention/FFN 表达。
                hidden = hidden + self.residual_scale * token_gate.to(hidden.dtype) * delta
                gate_values_by_block[block_index].append(token_gate.detach().float().mean())
                gate_tensors.append(token_gate)
            hidden = apply_token_mask_to_safe_embed(embed.float(), hidden, token_mask_list[sample_index])
            safe_embeds.append(hidden.to(dtype=original_dtype))

        self._last_gate_values = [
            float(torch.stack(block_values).mean().cpu())
            for block_values in gate_values_by_block
            if block_values
        ]
        self._last_gate_tensors = gate_tensors
        return safe_embeds
