#!/usr/bin/env python
"""正式的 Z-Image SAFREE baseline 运行脚本。

这个脚本把官方 SAFREE 的文本空间投影方法适配到 Z-Image：

- Z-Image prompt embedding 是变长 list，每条 prompt 的 token 张量形状为 ``[T, D]``。
- SAFREE 不训练模型，也不修改 Z-Image 权重，只在推理时替换 prompt embedding。
- 可以通过 ``--re_attn_t`` 控制 SAFREE embedding 在哪些 denoise step 生效。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipelines.ZImageIPPipeline import ZImagePipeline  # noqa: E402
from safree_text import (  # noqa: E402
    apply_safree_projection,
    encode_concept_vectors,
    encode_prompt_with_content_mask,
    projection_matrix_from_rows,
)


DEFAULT_MODEL_PATH = "/mnt/nas2/zhiwen/SafeGuard/models/Tongyi-MAI/Z-Image-Turbo"
DEFAULT_OUTPUT_DIR = "outputs/baselines/zimage_safree"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "cp936")

DEFAULT_TOXIC_CONCEPTS_0 = [
    # 色情/裸露类。
    "Genitals",
    "Nipples",
    "Exposed breasts",
    "Exposed buttocks",
    "Sexual intercourse",
    "Oral sex",
    "Penetration",
    "Pornography",
    "Full frontal nudity",
    # 暴力/血腥类。
    "Blood",
    "Gore",
    "Severed body part",
    "Corpse",
    "Murder",
    "Torture",
    # IP 类。
    "Elsa",
    "Snow White",
    "Doraemon",
    "Minions",
    "Spongebob Squarepants",
]

UNSAFE_PROMPT_PRESETS = {
    # 正式 baseline 对比默认使用统一 target concept 集合。
    # 这样 porn/gore/IP/benign 跑 SAFREE 时使用同一个危险概念子空间，避免
    # 不同测试集之间因为 target concept 不一致引入额外变量。
    "toxic0": DEFAULT_TOXIC_CONCEPTS_0,
    # 来自官方 SAFREE generate_safree.py 的 nudity 概念词表。
    "porn": [
        "Sexual Acts",
        "Content Meant to Arouse Sexual Excitement",
        "Nudity",
        "Pornography",
        "Erotic Art",
        "Lustful",
        "Seductive",
        "Orgasmic",
        "Libido",
        "Kinky",
        "Sexual Orientation",
        "Sexual Attraction",
        "Sexual Intercourse",
        "Sexual Pleasure",
        "Sexual Fantasy",
        "Carnal Desires",
        "Sexual Gratification",
    ],
    "gore": [
        "Blood",
        "Gore",
        "Severed body part",
        "Corpse",
        "Murder",
        "Torture",
        "Graphic violence",
        "Bloody injury",
        "Violent wound",
    ],
    "ip": [
        "copyrighted character",
        "famous cartoon character",
        "well-known animated character",
        "recognizable franchise character",
    ],
}


@dataclass
class PromptCase:
    """一次生成任务中被选中的一条 prompt。

    ``metadata`` 会保留 CSV 里的原始列，方便后续按 IP 类别、语言等字段分组统计。
    """

    run_index: int
    sample_id: str
    prompt: str
    source_csv: str
    source_stem: str
    csv_row_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 prompt CSV 上运行 Z-Image SAFREE baseline。")
    parser.add_argument(
        "--input_csv",
        required=True,
        help="CSV 文件、包含 CSV 的目录，或用逗号分隔的多个 CSV 路径；必须包含 prompt 列。",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)

    parser.add_argument("--prompt_column", default="prompt")
    parser.add_argument("--id_column", default="id")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--use_all_prompts", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=20260715)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--filter_metadata_key", default=None)
    parser.add_argument("--filter_metadata_value", default=None)
    parser.add_argument("--group_by_metadata_key", default=None)
    parser.add_argument("--group_metadata_values", default=None)
    parser.add_argument("--samples_per_group", type=int, default=None)

    parser.add_argument("--device_ids", default="0")
    parser.add_argument("--torch_dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention_backend", default="", help="可选 transformer attention backend，例如 flash。")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=9)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    parser.add_argument("--unsafe_preset", choices=tuple(UNSAFE_PROMPT_PRESETS), default="toxic0")
    parser.add_argument(
        "--unsafe_prompts",
        default="",
        help="可选危险概念 prompt，用分号分隔；设置后会覆盖 --unsafe_preset。",
    )
    parser.add_argument("--sf_alpha", type=float, default=0.01, help="SAFREE token 选择阈值系数 alpha。")
    parser.add_argument(
        "--safree_scale",
        type=float,
        default=1.0,
        help="SAFREE 投影 embedding 的插值强度；1.0 表示完全使用投影结果。",
    )
    parser.add_argument(
        "--re_attn_t",
        default="-1,100000",
        help="官方 SAFREE 风格的 step 生效范围 start,end；从 0 开始且左右都包含，默认表示所有 step。",
    )
    parser.add_argument("--self_validation_filter", action="store_true", help="使用 SAFREE 的 beta 自适应 step 截断。")
    parser.add_argument("--up_t", type=int, default=10)
    parser.add_argument("--concept_type", default="nudity", help="SAFREE f_beta 调度使用的概念类型字符串。")

    parser.add_argument("--run_base", action="store_true", help="同时用相同 seed 生成 base Z-Image 图片。")
    parser.add_argument("--save_compare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_values(text: str | None, separator: str = ",") -> list[str]:
    """把逗号或指定分隔符连接的参数拆成字符串列表。"""

    if not text:
        return []
    return [item.strip() for item in str(text).split(separator) if item.strip()]


def torch_dtype_from_name(name: str) -> torch.dtype:
    """把命令行中的 dtype 字符串转换成 torch dtype。"""

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def safe_filename(text: str, max_length: int = 96) -> str:
    """把样本 id 转成适合作为图片文件名的字符串。"""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    text = text.strip("._")
    return (text[:max_length] or "sample")


def coerce_text(value: Any) -> str:
    """把 CSV 单元格值转成去掉首尾空白的字符串。"""

    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取 CSV，按常见中文/英文编码依次尝试。"""

    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return [
                    {str(key): coerce_text(value) for key, value in row.items()}
                    for row in reader
                ]
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {csv_path} with encodings={CSV_ENCODINGS}") from last_error


def expand_input_csvs(input_csv: str) -> list[Path]:
    """展开 --input_csv，支持单文件、目录和逗号分隔的多路径。"""

    paths: list[Path] = []
    for item in split_values(input_csv):
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            paths.append(path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input CSV paths do not exist: {missing}")
    if not paths:
        raise ValueError("--input_csv did not resolve to any CSV files.")
    return paths


def load_prompt_cases(args: argparse.Namespace) -> list[PromptCase]:
    """从输入 CSV 中加载 prompt，并统一转换成 PromptCase。"""

    cases: list[PromptCase] = []
    for csv_path in expand_input_csvs(args.input_csv):
        rows = read_csv_rows(csv_path)
        for row_index, row in enumerate(rows):
            prompt = coerce_text(row.get(args.prompt_column, ""))
            if not prompt:
                continue
            raw_id = coerce_text(row.get(args.id_column, ""))
            sample_id = raw_id or f"{csv_path.stem}_{row_index:05d}"
            metadata = dict(row)
            metadata["csv_row_index"] = row_index
            metadata["source_csv"] = csv_path.name
            metadata["source_stem"] = csv_path.stem
            cases.append(
                PromptCase(
                    run_index=len(cases),
                    sample_id=sample_id,
                    prompt=prompt,
                    source_csv=csv_path.name,
                    source_stem=csv_path.stem,
                    csv_row_index=row_index,
                    metadata=metadata,
                )
            )
    if not cases:
        raise ValueError(f"No valid prompts loaded from --input_csv={args.input_csv!r}.")
    return cases


def metadata_value(case: PromptCase, key: str) -> str:
    """读取某条样本的 metadata 字段，并转成字符串。"""

    return coerce_text(case.metadata.get(key, ""))


def apply_filters(cases: list[PromptCase], args: argparse.Namespace) -> list[PromptCase]:
    """根据 metadata key/value 对样本做简单过滤。"""

    if bool(args.filter_metadata_key) != bool(args.filter_metadata_value is not None):
        raise ValueError("--filter_metadata_key and --filter_metadata_value must be set together.")

    filtered = list(cases)
    if args.filter_metadata_key:
        target_value = str(args.filter_metadata_value).strip().lower()
        filtered = [
            case
            for case in filtered
            if metadata_value(case, args.filter_metadata_key).lower() == target_value
        ]
    if not filtered:
        raise ValueError("No prompts remain after metadata filtering.")
    return filtered


def select_cases(cases: list[PromptCase], args: argparse.Namespace) -> list[PromptCase]:
    """根据参数抽样 prompt，支持整体随机抽样和按 metadata 分组抽样。"""

    filtered = apply_filters(cases, args)
    rng = random.Random(args.sample_seed)

    if args.samples_per_group is not None or args.group_by_metadata_key or args.group_metadata_values:
        if not args.group_by_metadata_key:
            raise ValueError("--group_by_metadata_key is required for grouped sampling.")
        if args.samples_per_group is None or args.samples_per_group <= 0:
            raise ValueError("--samples_per_group must be positive for grouped sampling.")
        group_values = split_values(args.group_metadata_values)
        if not group_values:
            raise ValueError("--group_metadata_values cannot be empty for grouped sampling.")

        selected: list[PromptCase] = []
        for group_value in group_values:
            group_pool = [
                case
                for case in filtered
                if metadata_value(case, args.group_by_metadata_key).lower() == group_value.lower()
            ]
            if len(group_pool) < args.samples_per_group:
                raise ValueError(
                    f"Not enough prompts for metadata.{args.group_by_metadata_key}={group_value!r}: "
                    f"available={len(group_pool)}, required={args.samples_per_group}"
                )
            selected.extend(rng.sample(group_pool, k=args.samples_per_group))
    elif args.use_all_prompts or args.num_prompts >= len(filtered):
        selected = list(filtered)
    else:
        if args.num_prompts <= 0:
            raise ValueError("--num_prompts must be positive unless --use_all_prompts is set.")
        selected = rng.sample(filtered, k=args.num_prompts)

    for run_index, case in enumerate(selected):
        case.run_index = run_index
    return selected


def split_round_robin(items: list[PromptCase], num_parts: int) -> list[list[PromptCase]]:
    """把样本 round-robin 分给多个 worker/GPU，避免单个 CSV 顺序导致负载偏斜。"""

    shards: list[list[PromptCase]] = [[] for _ in range(num_parts)]
    for index, item in enumerate(items):
        shards[index % num_parts].append(item)
    return shards


def parse_unsafe_prompts(args: argparse.Namespace) -> list[str]:
    """解析危险概念 prompt；自定义 --unsafe_prompts 优先于 preset。"""

    custom = split_values(args.unsafe_prompts, separator=";")
    if custom:
        return custom
    return list(UNSAFE_PROMPT_PRESETS[args.unsafe_preset])


def parse_re_attn_t(text: str, num_inference_steps: int) -> set[int]:
    """解析 SAFREE 生效的 denoise step 集合。

    ``--re_attn_t 0,3`` 表示第 0、1、2、3 步使用 SAFREE embedding。
    这里的 step index 从 0 开始，和 diffusers callback 的 ``step_index`` 保持一致。
    """

    values = split_values(text)
    if len(values) != 2:
        raise ValueError("--re_attn_t must have format start,end, e.g. -1,100000 or 0,3")
    start, end = int(values[0]), int(values[1])
    return {index for index in range(num_inference_steps) if start <= index <= end}


def make_output_name(case: PromptCase) -> str:
    """为单条样本构造稳定的图片文件名。"""

    return f"{case.run_index:05d}_{safe_filename(case.sample_id)}.png"


def build_pipeline(args: argparse.Namespace, device: torch.device) -> ZImagePipeline:
    """加载 Z-Image pipeline，并把所有模块切到 eval 状态。

    SAFREE 是 training-free baseline，这里不会打开梯度，也不会修改 transformer/text encoder/vae 参数。
    """

    dtype = torch_dtype_from_name(args.torch_dtype)
    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(device)

    # 三个主模块都只做推理；后续函数外层也会用 torch.no_grad()。
    pipe.text_encoder.eval()
    pipe.transformer.eval()
    pipe.vae.eval()
    if args.attention_backend:
        try:
            pipe.transformer.set_attention_backend(args.attention_backend)
            print(f"[{device}] attention backend set to {args.attention_backend}")
        except Exception as exc:
            print(f"[{device}] attention backend {args.attention_backend!r} unavailable: {exc}")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


def save_comparison(base_image: Image.Image, safree_image: Image.Image, output_path: Path) -> None:
    """把 base 和 SAFREE 图片左右拼接，方便肉眼对比。"""

    width, height = base_image.size
    label_height = 32
    canvas = Image.new("RGB", (width * 2, height + label_height), color=(255, 255, 255))
    canvas.paste(base_image.convert("RGB"), (0, label_height))
    canvas.paste(safree_image.convert("RGB"), (width, label_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), "base", fill=(0, 0, 0), font=font)
    draw.text((width + 10, 10), "safree", fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def initial_prompt_embeds_for_steps(
    base_prompt_embeds: list[torch.Tensor],
    safree_prompt_embeds: list[torch.Tensor],
    active_steps: set[int],
) -> list[torch.Tensor]:
    """决定 denoise 第 0 步应该使用 base embedding 还是 SAFREE embedding。

    Args:
        base_prompt_embeds: list 长度通常为 1，内部张量形状 ``[T, D]``。
        safree_prompt_embeds: list 长度通常为 1，内部张量形状 ``[T, D]``。
        active_steps: SAFREE 生效的 step index 集合。
    """

    return safree_prompt_embeds if 0 in active_steps else base_prompt_embeds


def make_step_switch_callback(
    base_prompt_embeds: list[torch.Tensor],
    safree_prompt_embeds: list[torch.Tensor],
    active_steps: set[int],
):
    """为下一步 denoise 切换 prompt embedding。

    Z-Image/diffusers 的 callback 在第 i 步结束时被调用，因此返回的
    ``prompt_embeds`` 会在第 i+1 步使用。

    张量形状：
        base_prompt_embeds: ``list[Tensor[T, D]]``。
        safree_prompt_embeds: ``list[Tensor[T, D]]``。
        callback 返回字典里的 ``prompt_embeds`` 仍保持同样结构。
    """

    def callback(_pipe, step_index: int, _timestep, _callback_kwargs):
        # 当前 step_index 已经结束，选择下一步要用的 embedding。
        next_step = step_index + 1
        return {
            "prompt_embeds": (
                safree_prompt_embeds
                if next_step in active_steps
                else base_prompt_embeds
            )
        }

    return callback


@torch.no_grad()
def generate_image_from_embeds(
    pipe: ZImagePipeline,
    prompt_embeds: list[torch.Tensor],
    negative_prompt_embeds: list[torch.Tensor],
    *,
    args: argparse.Namespace,
    generator: torch.Generator,
    callback_on_step_end=None,
) -> Image.Image:
    """直接用已经构造好的 prompt embedding 调用 Z-Image 生成图片。

    Args:
        prompt_embeds: ``list[Tensor[T, D]]``，第 0 个元素是一条 prompt 的正向 embedding。
        negative_prompt_embeds: ``list[Tensor[T_neg, D]]``，CFG 负向 prompt embedding；
            当 ``guidance_scale=0`` 时可能是空 list 或 pipeline 兼容格式。
        callback_on_step_end: 可选 step callback，用于在指定 step 切换 base/SAFREE embedding。
    """

    output = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
        return_dict=True,
    )
    return output.images[0]


@torch.no_grad()
def build_safree_prompt_embeds(
    pipe: ZImagePipeline,
    case: PromptCase,
    concept_projection: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict[str, Any], set[int]]:
    """编码一条 prompt，并构造 base/SAFREE 两套 prompt embedding。

    Returns:
        base_prompt_embeds: ``list[Tensor[T, D]]``，原始 prompt embedding。
        safree_prompt_embeds: ``list[Tensor[T, D]]``，经过 SAFREE 投影后的 embedding。
        negative_prompt_embeds: ``list[Tensor[T_neg, D]]``，负向 prompt embedding。
        metrics: SAFREE 投影诊断信息，如投影 token 数、beta、active steps。
        active_steps: SAFREE embedding 实际生效的 denoise step 集合。
    """

    encoded = encode_prompt_with_content_mask(
        pipe,
        case.prompt,
        device=device,
        max_sequence_length=args.max_sequence_length,
    )

    # encoded.embeds: [T, D]；pipeline 期望 prompt_embeds 是 list[Tensor[T, D]]。
    base_prompt_embeds = [encoded.embeds.to(device)]

    # 复用 pipeline.encode_prompt 生成 negative prompt embedding。
    # 正向 prompt_embeds 已经传入，因此不会重复编码正向 prompt。
    _, negative_prompt_embeds = pipe.encode_prompt(
        prompt=[case.prompt],
        device=device,
        do_classifier_free_guidance=args.guidance_scale > 0,
        negative_prompt=args.negative_prompt,
        prompt_embeds=base_prompt_embeds,
        max_sequence_length=args.max_sequence_length,
    )

    # negative_prompt_embeds: list[Tensor[T_neg, D]]，放到当前 GPU。
    negative_prompt_embeds = [embed.to(device) for embed in negative_prompt_embeds]

    # projection_result.embeds: [T, D]，只修改 user content token，template token 保持不变。
    projection_result = apply_safree_projection(
        encoded.embeds,
        encoded.content_mask,
        concept_projection,
        alpha=args.sf_alpha,
        scale=args.safree_scale,
        use_self_validation_filter=args.self_validation_filter,
        up_t=args.up_t,
        concept_type=args.concept_type,
    )
    safree_prompt_embeds = [projection_result.embeds.to(device)]

    if args.self_validation_filter and projection_result.adaptive_end_step is not None:
        # 使用 SAFREE 自适应 beta 截断时，从第 0 步到 end_step 使用 SAFREE embedding。
        end_step = min(int(projection_result.adaptive_end_step), args.num_inference_steps - 1)
        active_steps = set(range(0, end_step + 1))
    else:
        # 否则使用命令行 --re_attn_t 指定的 step 范围。
        active_steps = parse_re_attn_t(args.re_attn_t, args.num_inference_steps)

    metrics = {
        "projected_tokens": projection_result.projected_tokens,
        "content_tokens": projection_result.content_tokens,
        "beta": projection_result.beta,
        "adaptive_end_step": projection_result.adaptive_end_step,
        "mean_token_cosine": projection_result.mean_token_cosine,
        "active_steps": sorted(active_steps),
    }
    return base_prompt_embeds, safree_prompt_embeds, negative_prompt_embeds, metrics, active_steps


def case_to_record(case: PromptCase) -> dict[str, Any]:
    """把 PromptCase 转成可写入 jsonl 的基础记录。"""

    return {
        "run_index": case.run_index,
        "sample_id": case.sample_id,
        "prompt": case.prompt,
        "source_csv": case.source_csv,
        "source_stem": case.source_stem,
        "csv_row_index": case.csv_row_index,
        "metadata": case.metadata,
    }


def run_worker(
    rank: int,
    gpu_id: int,
    cases: list[PromptCase],
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """单个 worker/GPU 的主循环。

    每个 worker 会独立加载一份 Z-Image pipeline，构造危险概念子空间，然后处理自己分到的
    prompt shard。这样多卡运行时不同 GPU 之间不共享模型状态，逻辑更简单。
    """

    if not cases:
        print(f"[GPU{gpu_id}] no prompts assigned")
        return

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # 固定当前进程使用的 CUDA 设备，避免默认设备混乱。
        torch.cuda.set_device(device)

    print(f"[GPU{gpu_id}] loading Z-Image for SAFREE")
    pipe = build_pipeline(args, device)

    unsafe_prompts = parse_unsafe_prompts(args)
    print(f"[GPU{gpu_id}] building unsafe concept subspace with {len(unsafe_prompts)} prompts")

    # concept_vectors: [N_concept, D]，每行是一个危险概念 prompt 的平均内容 embedding。
    concept_vectors = encode_concept_vectors(
        pipe,
        unsafe_prompts,
        device=device,
        max_sequence_length=args.max_sequence_length,
    )

    # concept_projection: [D, D]，危险概念子空间投影矩阵 P_risk。
    concept_projection = projection_matrix_from_rows(concept_vectors).to(device)

    root = Path(output_dir)
    shard_dir = root / "records" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"worker_{rank}_gpu{gpu_id}.jsonl"

    base_root = root / "images" / "base"
    safree_root = root / "images" / "safree"
    compare_root = root / "compare"

    with shard_path.open("w", encoding="utf-8") as handle:
        for local_index, case in enumerate(cases, start=1):
            start_time = time.time()
            seed = args.generation_seed + case.run_index
            file_name = make_output_name(case)
            base_path = base_root / case.source_stem / file_name
            safree_path = safree_root / case.source_stem / file_name
            compare_path = compare_root / case.source_stem / file_name

            record = case_to_record(case)
            record.update(
                {
                    "gpu_id": gpu_id,
                    "worker_rank": rank,
                    "seed": seed,
                    "base_image_path": str(base_path) if args.run_base else "",
                    "safree_image_path": str(safree_path),
                    "compare_image_path": str(compare_path) if args.save_compare else "",
                }
            )

            should_skip = safree_path.exists() and not args.overwrite
            if args.run_base:
                should_skip = should_skip and base_path.exists()
            if args.save_compare:
                should_skip = should_skip and compare_path.exists()
            if should_skip:
                record["status"] = "skipped_existing"
                record["elapsed_sec"] = 0.0
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[GPU{gpu_id}] {local_index:04d}/{len(cases):04d} skip {case.sample_id}")
                continue

            try:
                safree_path.parent.mkdir(parents=True, exist_ok=True)
                if args.run_base:
                    base_path.parent.mkdir(parents=True, exist_ok=True)

                # base_prompt_embeds / safree_prompt_embeds: list[Tensor[T, D]]。
                # negative_prompt_embeds: list[Tensor[T_neg, D]]。
                base_prompt_embeds, safree_prompt_embeds, negative_prompt_embeds, safree_metrics, active_steps = (
                    build_safree_prompt_embeds(pipe, case, concept_projection, args, device)
                )
                record["safree_metrics"] = safree_metrics

                base_image: Image.Image | None = None
                if args.run_base:
                    # base 分支完全使用原始 prompt 文本，由 pipeline 自己编码。
                    base_generator = torch.Generator(device=device).manual_seed(seed)
                    base_image = pipe(
                        prompt=case.prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=base_generator,
                        negative_prompt=args.negative_prompt,
                        max_sequence_length=args.max_sequence_length,
                        return_dict=True,
                    ).images[0]
                    base_image.save(base_path)

                # 第 0 步之前需要先决定初始 prompt_embeds；后续 step 由 callback 切换。
                initial_embeds = initial_prompt_embeds_for_steps(
                    base_prompt_embeds,
                    safree_prompt_embeds,
                    active_steps,
                )

                # callback 会在每一步结束时返回下一步要使用的 prompt_embeds。
                step_callback = make_step_switch_callback(
                    base_prompt_embeds,
                    safree_prompt_embeds,
                    active_steps,
                )

                # SAFREE 分支和 base 分支使用相同 seed，便于比较 embedding 修改本身的影响。
                safree_generator = torch.Generator(device=device).manual_seed(seed)
                safree_image = generate_image_from_embeds(
                    pipe,
                    initial_embeds,
                    negative_prompt_embeds,
                    args=args,
                    generator=safree_generator,
                    callback_on_step_end=step_callback,
                )
                safree_image.save(safree_path)

                if args.save_compare:
                    if base_image is None and base_path.exists():
                        with Image.open(base_path) as loaded:
                            base_image = loaded.convert("RGB")
                    if base_image is None:
                        raise RuntimeError("--save_compare requires --run_base or an existing base image")
                    save_comparison(base_image, safree_image, compare_path)

                record["status"] = "ok"
                record["elapsed_sec"] = round(time.time() - start_time, 3)
                print(
                    f"[GPU{gpu_id}] {local_index:04d}/{len(cases):04d} ok "
                    f"seed={seed} id={case.sample_id} projected={safree_metrics['projected_tokens']}/"
                    f"{safree_metrics['content_tokens']}"
                )
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                record["elapsed_sec"] = round(time.time() - start_time, 3)
                print(f"[GPU{gpu_id}] error id={case.sample_id}: {exc}")

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            gc.collect()
            if torch.cuda.is_available():
                # 清理当前 prompt 的临时显存，长批量生成时更稳。
                torch.cuda.empty_cache()

    print(f"[GPU{gpu_id}] done -> {shard_path}")


def merge_shards(output_dir: Path) -> list[dict[str, Any]]:
    """合并所有 worker 写出的 shard jsonl，并按 run_index 排序。"""

    shard_dir = output_dir / "records" / "shards"
    records: list[dict[str, Any]] = []
    for shard_path in sorted(shard_dir.glob("worker_*.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records.sort(key=lambda item: int(item.get("run_index", 0)))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """写 jsonl 文件，每行一条 record。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float | None:
    """安全计算均值；空列表返回 None，便于写入 summary。"""

    if not values:
        return None
    return sum(values) / len(values)


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总本次 SAFREE 运行的配置和基础统计信息。"""

    status_counts = Counter(record.get("status", "unknown") for record in records)
    source_counts = Counter(record.get("source_stem", "unknown") for record in records)
    projected_counts = [
        float(record["safree_metrics"]["projected_tokens"])
        for record in records
        if "safree_metrics" in record
    ]
    content_counts = [
        float(record["safree_metrics"]["content_tokens"])
        for record in records
        if "safree_metrics" in record
    ]
    betas = [
        float(record["safree_metrics"]["beta"])
        for record in records
        if "safree_metrics" in record
    ]
    elapsed = [
        float(record["elapsed_sec"])
        for record in records
        if record.get("status") == "ok"
    ]
    return {
        "runner": "zimage_safree_baseline",
        "input_csv": args.input_csv,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "num_records": len(records),
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "mean_elapsed_sec": mean(elapsed),
        "mean_projected_tokens": mean(projected_counts),
        "mean_content_tokens": mean(content_counts),
        "mean_beta": mean(betas),
        "config": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "max_sequence_length": args.max_sequence_length,
            "unsafe_preset": args.unsafe_preset,
            "unsafe_prompts": parse_unsafe_prompts(args),
            "sf_alpha": args.sf_alpha,
            "safree_scale": args.safree_scale,
            "re_attn_t": args.re_attn_t,
            "self_validation_filter": args.self_validation_filter,
            "up_t": args.up_t,
            "concept_type": args.concept_type,
            "run_base": args.run_base,
            "save_compare": args.save_compare,
            "sample_seed": args.sample_seed,
            "generation_seed": args.generation_seed,
            "device_ids": args.device_ids,
            "torch_dtype": args.torch_dtype,
            "attention_backend": args.attention_backend,
        },
    }


def write_run_config(args: argparse.Namespace, selected_cases: list[PromptCase], output_dir: Path) -> None:
    """保存运行配置和抽样到的 prompt，保证实验可追溯。"""

    data = {
        "args": vars(args),
        "num_selected_cases": len(selected_cases),
        "selected_cases": [
            {
                "run_index": case.run_index,
                "sample_id": case.sample_id,
                "source_csv": case.source_csv,
                "csv_row_index": case.csv_row_index,
                "prompt": case.prompt,
                "metadata": case.metadata,
            }
            for case in selected_cases
        ],
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    """命令行入口：抽样、分片运行、合并结果。"""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = load_prompt_cases(args)
    selected_cases = select_cases(all_cases, args)
    write_run_config(args, selected_cases, output_dir)

    device_ids = [int(value) for value in split_values(args.device_ids)]
    if not device_ids:
        raise ValueError("--device_ids must contain at least one device id.")

    print(f"[zimage-safree] loaded {len(all_cases)} prompts")
    print(f"[zimage-safree] selected {len(selected_cases)} prompts")
    print(f"[zimage-safree] devices={device_ids}")
    print(f"[zimage-safree] output_dir={output_dir}")
    print(f"[zimage-safree] unsafe_prompt_count={len(parse_unsafe_prompts(args))}")

    shards = split_round_robin(selected_cases, len(device_ids))
    for rank, shard in enumerate(shards):
        print(f"[zimage-safree] shard {rank}: {len(shard)} prompts")

    if len(device_ids) == 1:
        # 单卡时直接在当前进程跑，调试更方便。
        run_worker(0, device_ids[0], shards[0], args, str(output_dir))
    else:
        # 多卡时使用 spawn，为每张 GPU 启动一个独立进程。
        ctx = get_context("spawn")
        processes = []
        for rank, gpu_id in enumerate(device_ids):
            proc = ctx.Process(
                target=run_worker,
                args=(rank, gpu_id, shards[rank], args, str(output_dir)),
            )
            proc.start()
            processes.append(proc)

        exit_codes = []
        for proc in processes:
            proc.join()
            exit_codes.append(proc.exitcode)
        if any(code != 0 for code in exit_codes):
            raise SystemExit(f"One or more SAFREE workers failed: {exit_codes}")

    records = merge_shards(output_dir)
    write_jsonl(output_dir / "records" / "results.jsonl", records)
    summary = build_summary(args, records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[zimage-safree] results: {output_dir / 'records' / 'results.jsonl'}")
    print(f"[zimage-safree] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
