"""训练配置。

这里全部使用 dataclass，原因是：
1. 训练入口可以直接由 argparse 填充；
2. checkpoint 可以把配置原样保存，方便复现实验；
3. 模块之间传参时不需要到处传一长串零散参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_UNSAFE_PORN_CSV = "/mnt/nas2/zhiwen/SafeGuard/safree_0615/testset_split/porn/level_5.csv"
DEFAULT_BENIGN_CSV = "/mnt/nas2/zhiwen/SafeGuard/safree_0615/testset_split/benign_all.csv"
DEFAULT_ZIMAGE_MODEL_PATH = "/mnt/nas2/zhiwen/SafeGuard/models/Tongyi-MAI/Z-Image-Turbo"


@dataclass
class DatasetConfig:
    """prompt 数据配置。

    当前第一版只做 porn-only：
    - unsafe_csv 使用 level_5 porn prompt；
    - benign_csv 使用 porn risk 为 0 的 benign_all；
    - benign 是否要求全维度 safe 由 filter_benign_label_safe 控制。
    """

    unsafe_csv: str = DEFAULT_UNSAFE_PORN_CSV
    benign_csv: str = DEFAULT_BENIGN_CSV
    # 逗号分隔的额外 benign/related-preservation CSV。读取后会和 benign_csv
    # 合并，训练时作为 is_benign=True 样本，只参与 identity/embedding 等保护项。
    extra_benign_csvs: str | None = None
    # 额外 benign 样本在采样池中的重复次数，用来避免小规模 related set 被
    # 大 benign_all 稀释。1 表示不重复。
    extra_benign_repeat: int = 1
    filter_benign_label_safe: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 20260715
    sample_seed_base: int = 42
    max_unsafe_samples: int | None = None
    max_benign_samples: int | None = None


@dataclass
class AdapterConfig:
    """SafeEmbeddingAdapter 结构配置。

    embedding_dim 由 Z-Image text encoder 的输出维度决定。训练脚本默认先编码
    一个 prompt 自动推断，因此命令行一般不用手动指定。
    """

    adapter_type: str = "mlp"
    embedding_dim: int | None = None
    bottleneck_dim: int | None = None
    attention_dim: int | None = None
    attention_heads: int = 4
    attention_ffn_multiplier: int = 4
    adapter_depth: int = 1
    gate_type: str = "none"
    learnable_gate: bool = False
    gate_init: float = 0.5
    residual_scale: float = 0.1
    dropout: float = 0.0
    use_risk_condition: bool = True
    num_risk_types: int = 9
    # Adapter 只为 5 个待擦除 IP 建立 condition；Z-03 的“其它”类不参与。
    num_classifier_classes: int = 5
    classifier_condition_hidden_dim: int | None = None
    clamp_delta: bool = True
    zero_init_depth2: bool = True


@dataclass
class ProxyLatentConfig:
    """Z-Image latent_x1 proxy 配置。

    target_step 由训练循环随机采样；这里保存的是采样范围和 denoise 总步数。
    height/width 默认用 512 降低显存压力。如果要更贴近最终评估，可以改为 1024。
    Z-03 实际输入是目标 step 的一步 x1 估计：
        latent_x1 = latents - sigma_t * velocity
    而不是 scheduler.step 后的下一步 noisy latent。
    """

    height: int = 512
    width: int = 512
    num_inference_steps: int = 9
    t_min: int = 2
    t_max: int = 9
    guidance_scale: float = 0.0
    cfg_normalization: bool = False
    cfg_truncation: float = 1.0
    max_sequence_length: int = 512


@dataclass
class LossConfig:
    """loss 权重配置。

    第一版默认使用 porn CE。也可以切到 softplus margin 或 hinge margin；
    benign prompt 的保护主要依赖 identity loss，避免 adapter 把安全语义也大幅改写。
    """

    w_porn: float = 1.0
    w_emb: float = 0.05
    w_id: float = 0.1
    risk_loss_type: str = "ce"
    use_margin_loss: bool = False
    margin: float = 0.5
    ip_loss_mode: str = "known_sum"
    risk_loss_on: str = "all"
    # DINO cosine 反馈默认只对 unsafe prompt 生效；benign 主要由 identity loss 保护。
    dino_loss_on: str = "unsafe_only"
    # 风险敏感子空间外 residual 正则权重。>0 时需要提供 TrainConfig.risk_subspace_dir。
    w_risk_subspace_perp: float = 0.0
    # all=对所有样本约束 Delta_perp；unsafe_only=只对 unsafe 样本约束。
    risk_subspace_loss_on: str = "unsafe_only"


@dataclass
class TrainConfig:
    """训练循环配置。"""

    output_dir: str = "outputs/safe_embedding_adapter"
    max_train_steps: int = 1000
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    benign_fraction: float = 0.5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 0
    torch_dtype: str = "bfloat16"
    mixed_precision: str = "no"
    device: str = "cuda"
    num_workers: int = 0
    sampler_seed: int = 20260715
    randomize_train_seeds: bool = True
    train_seed_upper_bound: int = 2_147_483_647
    target_risk: str = "porn"  # porn / gore / ip
    # 反馈模型：z03 使用 latent 分类器；dino 使用 VAE decode 后的图像特征相似度。
    feedback_model: str = "z03"
    # Z-03 模型来源。auto 会根据 checkpoint 路径自动识别 legacy/export。
    z03_model_type: str = "auto"
    z03_model_file: str | None = None
    z03_ckpt: str | None = None
    dino_model_name_or_path: str | None = None
    dino_reference_features: str | None = None
    dino_local_files_only: bool = True
    ip_conditioning: bool = True
    # IP 训练时，普通 benign 默认随机分配 IP condition；打开后，如果 benign
    # 样本 metadata 里能推断出 IP condition，就用该 condition 算 identity。
    # 这主要服务 related concept preservation 数据集。
    benign_ip_condition_from_metadata: bool = False
    restrict_adapter_to_user_content_tokens: bool = False
    # 离线计算得到的 P_risk 目录。当前主要用于 IP 训练，目录内应包含
    # ip_xxx_P_risk_rank*.pt 文件。
    risk_subspace_dir: str | None = None
    # 可选：只取 P_risk 前多少个方向；None 表示使用文件中保存的全部 rank。
    risk_subspace_rank: int | None = None
    # 可选：训练 risk loss 时，先用 Z-03 对 latent_x1 求空间 saliency，
    # 再只让高响应区域的 latent 梯度主要回传给 adapter。forward 数值保持
    # 完整 latent，不遮挡 Z-03 输入。
    risk_spatial_grad_mask: bool = False
    risk_spatial_mask_type: str = "soft"  # soft / hard
    risk_spatial_mask_topk: float = 0.2
    risk_spatial_mask_floor: float = 0.05
    risk_spatial_mask_gamma: float = 1.0
    risk_spatial_mask_score: str = "margin"  # target_logit / margin / prob
    resume_from_checkpoint: str | None = None
    reset_optimizer_on_resume: bool = False
    num_processes: int = 1


def torch_dtype_from_name(name: str) -> torch.dtype:
    """把命令行字符串转换成 torch dtype。"""

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """保存 checkpoint 前统一把 dataclass 转成普通 dict。"""

    data = asdict(obj)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
