"""prompt CSV 读取、划分和平衡采样。

数据读取层只负责“文本样本”的结构化，不参与模型逻辑。这样后续扩展 gore/IP
数据源时，只需要在这里增加新的标签和 risk_heads，不需要改 trainer。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .config import DatasetConfig


@dataclass(frozen=True)
class PromptSample:
    """单条 prompt 样本。

    porn_target 当前固定为 0，因为训练目标不是复现原始 unsafe 标签，而是让修正后
    的 latent 在 Z-03 porn head 上更接近 safe 类。
    """

    sample_id: str
    prompt: str
    source_csv: str
    is_benign: bool
    porn_target: int = 0
    seed: int = 42
    risk_heads: tuple[str, ...] = ("porn",)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBatch:
    """训练循环使用的 batch 容器。

    保留字符串 id/source_csv 是为了日志和排错；真正进入模型的是 prompts、seeds
    和 is_benign。
    """

    samples: list[PromptSample]

    @property
    def prompts(self) -> list[str]:
        return [sample.prompt for sample in self.samples]

    @property
    def seeds(self) -> list[int]:
        return [sample.seed for sample in self.samples]

    @property
    def sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

    @property
    def source_csvs(self) -> list[str]:
        return [sample.source_csv for sample in self.samples]

    def benign_mask(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor([sample.is_benign for sample in self.samples], dtype=torch.bool, device=device)


def _coerce_prompt(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def parse_csv_path_list(value: str | None) -> list[str]:
    """解析逗号分隔的 CSV 路径列表。"""

    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _coerce_bool(value: Any) -> bool:
    """兼容 pandas 读出的 bool、字符串 bool 和 0/1。"""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _read_csv_with_pandas(csv_path: str | Path):
    """读取 CSV。

    这里刻意使用 pandas，而不是按行 split。原因是 safree CSV 里的 llm_raw_output
    可能包含多行 JSON，简单的行级解析会把一条样本拆坏。
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("读取训练 CSV 需要 pandas，请在当前 conda 环境中安装 pandas。") from exc

    return pd.read_csv(csv_path)


def _row_get(row: Any, key: str, default: Any = "") -> Any:
    if key not in row:
        return default
    value = row[key]
    try:
        import pandas as pd

        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _build_samples_from_frame(
    frame: Any,
    csv_path: str | Path,
    *,
    is_benign: bool,
    seed_start: int,
    max_samples: int | None,
) -> list[PromptSample]:
    csv_path = Path(csv_path)
    samples: list[PromptSample] = []

    for row_index, row in frame.iterrows():
        prompt = _coerce_prompt(_row_get(row, "prompt"))
        if not prompt:
            continue

        raw_id = _coerce_prompt(_row_get(row, "id"))
        sample_id = raw_id or f"{csv_path.stem}_{row_index:05d}"

        # metadata 只保留轻量字段，避免把多行 llm_raw_output 放进内存和日志。
        metadata = {
            "csv_row_index": int(row_index),
            "label_porn_risk_level": _row_get(row, "label_porn_risk_level", ""),
            "label_gore_risk_level": _row_get(row, "label_gore_risk_level", ""),
            "label_ip_risk_level": _row_get(row, "label_ip_risk_level", ""),
            "label_safe": _row_get(row, "label_safe", ""),
            "category": _row_get(row, "category", ""),
            "control_category": _row_get(row, "control_category", ""),
            "sub_category": _row_get(row, "sub_category", ""),
            "original_category": _row_get(row, "original_category", ""),
            # related concept preservation CSV 可以用这些字段显式声明：
            # “这条 benign prompt 应该在哪个 IP condition 下保持 identity”。
            "preserve_condition": _row_get(row, "preserve_condition", ""),
            "target_condition": _row_get(row, "target_condition", ""),
            "risk_id": _row_get(row, "risk_id", ""),
            "target_risk_id": _row_get(row, "target_risk_id", ""),
            "related_to_ip": _row_get(row, "related_to_ip", ""),
            "ip_names": _row_get(row, "ip_names", ""),
            "ip_type": _row_get(row, "ip_type", ""),
        }

        samples.append(
            PromptSample(
                sample_id=str(sample_id),
                prompt=prompt,
                source_csv=csv_path.name,
                is_benign=is_benign,
                porn_target=0,
                seed=seed_start + len(samples),
                risk_heads=("porn",),
                metadata=metadata,
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def load_prompt_samples(config: DatasetConfig) -> tuple[list[PromptSample], list[PromptSample]]:
    """读取 unsafe porn 和 benign prompt。

    返回值按语义拆成两个列表，后续划分 train/val/test 时会分别打乱和 split，
    避免某个 split 中正负比例失衡。
    """

    unsafe_frame = _read_csv_with_pandas(config.unsafe_csv)
    benign_frame = _read_csv_with_pandas(config.benign_csv)
    extra_benign_paths = parse_csv_path_list(config.extra_benign_csvs)
    extra_benign_frames = [
        (path, _read_csv_with_pandas(path))
        for path in extra_benign_paths
    ]

    if "prompt" not in unsafe_frame.columns:
        raise ValueError(f"unsafe_csv 缺少 prompt 列: {config.unsafe_csv}")
    if "prompt" not in benign_frame.columns:
        raise ValueError(f"benign_csv 缺少 prompt 列: {config.benign_csv}")
    for extra_path, extra_frame in extra_benign_frames:
        if "prompt" not in extra_frame.columns:
            raise ValueError(f"extra_benign_csv 缺少 prompt 列: {extra_path}")

    if config.filter_benign_label_safe:
        if "label_safe" not in benign_frame.columns:
            raise ValueError("filter_benign_label_safe=True 但 benign_csv 缺少 label_safe 列")
        benign_frame = benign_frame[benign_frame["label_safe"].map(_coerce_bool)]
        filtered_extra_frames = []
        for extra_path, extra_frame in extra_benign_frames:
            if "label_safe" not in extra_frame.columns:
                raise ValueError(
                    "filter_benign_label_safe=True 但 extra_benign_csv 缺少 label_safe 列: "
                    f"{extra_path}"
                )
            filtered_extra_frames.append((extra_path, extra_frame[extra_frame["label_safe"].map(_coerce_bool)]))
        extra_benign_frames = filtered_extra_frames

    unsafe_samples = _build_samples_from_frame(
        unsafe_frame,
        config.unsafe_csv,
        is_benign=False,
        seed_start=config.sample_seed_base,
        max_samples=config.max_unsafe_samples,
    )
    benign_samples = _build_samples_from_frame(
        benign_frame,
        config.benign_csv,
        is_benign=True,
        seed_start=config.sample_seed_base + 1_000_000,
        max_samples=None,
    )
    extra_repeat = int(config.extra_benign_repeat)
    if extra_repeat <= 0:
        raise ValueError("extra_benign_repeat 必须大于等于 1")
    for extra_index, (extra_path, extra_frame) in enumerate(extra_benign_frames):
        extra_samples = _build_samples_from_frame(
            extra_frame,
            extra_path,
            is_benign=True,
            seed_start=config.sample_seed_base + 2_000_000 + extra_index * 1_000_000,
            max_samples=None,
        )
        for _ in range(extra_repeat):
            benign_samples.extend(extra_samples)
    if config.max_benign_samples is not None:
        benign_samples = benign_samples[: config.max_benign_samples]

    if not unsafe_samples:
        raise ValueError(f"没有从 unsafe_csv 读到有效 prompt: {config.unsafe_csv}")
    if not benign_samples:
        raise ValueError(f"没有从 benign_csv 读到有效 prompt: {config.benign_csv}")

    return unsafe_samples, benign_samples


def _split_one_group(
    samples: Sequence[PromptSample],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[PromptSample], list[PromptSample], list[PromptSample]]:
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)

    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return items[:train_end], items[train_end:val_end], items[val_end:]


def split_prompt_samples(
    unsafe_samples: Sequence[PromptSample],
    benign_samples: Sequence[PromptSample],
    config: DatasetConfig,
) -> dict[str, list[PromptSample]]:
    """分别 split unsafe/benign 后再合并。

    这样 train/val/test 都能保留相近的类别比例；如果先混合再 split，小数据 smoke
    test 时很容易某个 split 缺少 benign 或 unsafe。
    """

    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratio 之和必须为 1，当前为 {total_ratio}")

    unsafe_train, unsafe_val, unsafe_test = _split_one_group(
        unsafe_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.split_seed,
    )
    benign_train, benign_val, benign_test = _split_one_group(
        benign_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.split_seed + 17,
    )

    return {
        "train": unsafe_train + benign_train,
        "val": unsafe_val + benign_val,
        "test": unsafe_test + benign_test,
        "train_unsafe": unsafe_train,
        "train_benign": benign_train,
        "val_unsafe": unsafe_val,
        "val_benign": benign_val,
        "test_unsafe": unsafe_test,
        "test_benign": benign_test,
    }


class BalancedPromptBatchSampler:
    """按固定期望比例有放回采样 batch。

    第一版训练通常比较贵，训练步数由 max_train_steps 控制，不一定要完整遍历
    一个 epoch。因此这里使用“有放回采样”比 DataLoader epoch 语义更直接。
    对小 batch，benign 数量按期望比例随机取整。例如 batch_size=1 且
    benign_fraction=0.3 时，长期约 30% step 会抽 benign。
    """

    def __init__(
        self,
        samples: Iterable[PromptSample],
        *,
        batch_size: int,
        benign_fraction: float = 0.5,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if not 0.0 <= benign_fraction <= 1.0:
            raise ValueError("benign_fraction 必须在 [0, 1] 之间")

        self.batch_size = int(batch_size)
        self.benign_fraction = float(benign_fraction)
        self.rng = random.Random(seed)
        self.unsafe_samples = [sample for sample in samples if not sample.is_benign]
        self.benign_samples = [sample for sample in samples if sample.is_benign]

        if not self.unsafe_samples:
            raise ValueError("BalancedPromptBatchSampler 缺少 unsafe samples")
        if not self.benign_samples:
            raise ValueError("BalancedPromptBatchSampler 缺少 benign samples")

    def _sample_benign_count(self) -> int:
        """按 benign_fraction 抽样当前 batch 的 benign 数量。"""

        expected = self.batch_size * self.benign_fraction
        benign_count = int(expected)
        fractional = expected - benign_count
        if fractional > 0.0 and self.rng.random() < fractional:
            benign_count += 1
        return min(max(benign_count, 0), self.batch_size)

    def sample_batch(self) -> PromptBatch:
        benign_count = self._sample_benign_count()
        unsafe_count = self.batch_size - benign_count

        batch = []
        batch.extend(self.rng.choices(self.unsafe_samples, k=unsafe_count))
        batch.extend(self.rng.choices(self.benign_samples, k=benign_count))
        self.rng.shuffle(batch)
        return PromptBatch(samples=batch)
