#!/usr/bin/env python
"""Z-Image base model 的数据集级别生成脚本。

这个 runner 只负责原始 Z-Image 生成，不做 SAFREE/STG/adapter 等任何安全修改。
接口和当前 SAFREE/STG runner 尽量保持一致，方便同一批 CSV 用相同 seed 和生成参数
跑 baseline 对比。
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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.ZImageIPPipeline import ZImagePipeline  # noqa: E402


DEFAULT_MODEL_PATH = "/mnt/nas2/zhiwen/SafeGuard/models/Tongyi-MAI/Z-Image-Turbo"
DEFAULT_OUTPUT_DIR = "outputs/baselines/zimage_base"
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "cp936")


@dataclass
class PromptCase:
    """一次生成任务中的一条 prompt。"""

    run_index: int
    sample_id: str
    prompt: str
    source_csv: str
    source_stem: str
    csv_row_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base Z-Image generation on prompt CSV files.")
    parser.add_argument("--input_csv", required=True, help="CSV 文件、CSV 目录，或逗号分隔的多个 CSV 路径。")
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
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_values(text: str | None, separator: str = ",") -> list[str]:
    if not text:
        return []
    return [item.strip() for item in str(text).split(separator) if item.strip()]


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def safe_filename(text: str, max_length: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    text = text.strip("._")
    return text[:max_length] or "sample"


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return [{str(key): coerce_text(value) for key, value in row.items()} for row in reader]
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {csv_path} with encodings={CSV_ENCODINGS}") from last_error


def expand_input_csvs(input_csv: str) -> list[Path]:
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
    cases: list[PromptCase] = []
    for csv_path in expand_input_csvs(args.input_csv):
        rows = read_csv_rows(csv_path)
        for row_index, row in enumerate(rows):
            prompt = coerce_text(row.get(args.prompt_column, ""))
            if not prompt:
                continue
            sample_id = coerce_text(row.get(args.id_column, "")) or f"{csv_path.stem}_{row_index:05d}"
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
    return coerce_text(case.metadata.get(key, ""))


def apply_filters(cases: list[PromptCase], args: argparse.Namespace) -> list[PromptCase]:
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
    shards: list[list[PromptCase]] = [[] for _ in range(num_parts)]
    for index, item in enumerate(items):
        shards[index % num_parts].append(item)
    return shards


def make_output_name(case: PromptCase) -> str:
    return f"{case.run_index:05d}_{safe_filename(case.sample_id)}.png"


def build_pipeline(args: argparse.Namespace, device: torch.device) -> ZImagePipeline:
    dtype = torch_dtype_from_name(args.torch_dtype)
    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(device)
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


def case_to_record(case: PromptCase) -> dict[str, Any]:
    return {
        "run_index": case.run_index,
        "sample_id": case.sample_id,
        "prompt": case.prompt,
        "source_csv": case.source_csv,
        "source_stem": case.source_stem,
        "csv_row_index": case.csv_row_index,
        "metadata": case.metadata,
    }


@torch.no_grad()
def generate_base_image(
    pipe: ZImagePipeline,
    case: PromptCase,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipe(
        prompt=case.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
        return_dict=True,
    )
    return output.images[0]


def run_worker(
    rank: int,
    gpu_id: int,
    cases: list[PromptCase],
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    if not cases:
        print(f"[GPU{gpu_id}] no prompts assigned")
        return

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"[GPU{gpu_id}] loading Z-Image base model")
    pipe = build_pipeline(args, device)

    root = Path(output_dir)
    shard_dir = root / "records" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"worker_{rank}_gpu{gpu_id}.jsonl"
    image_root = root / "images" / "base"

    with shard_path.open("w", encoding="utf-8") as handle:
        for local_index, case in enumerate(cases, start=1):
            start_time = time.time()
            seed = args.generation_seed + case.run_index
            image_path = image_root / case.source_stem / make_output_name(case)

            record = case_to_record(case)
            record.update(
                {
                    "gpu_id": gpu_id,
                    "worker_rank": rank,
                    "seed": seed,
                    "base_image_path": str(image_path),
                }
            )

            if image_path.exists() and not args.overwrite:
                record["status"] = "skipped_existing"
                record["elapsed_sec"] = 0.0
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[GPU{gpu_id}] {local_index:04d}/{len(cases):04d} skip {case.sample_id}")
                continue

            try:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = generate_base_image(pipe, case, args, device, seed)
                image.save(image_path)
                record["status"] = "ok"
                record["elapsed_sec"] = round(time.time() - start_time, 3)
                print(f"[GPU{gpu_id}] {local_index:04d}/{len(cases):04d} ok seed={seed} id={case.sample_id}")
            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                record["elapsed_sec"] = round(time.time() - start_time, 3)
                print(f"[GPU{gpu_id}] error id={case.sample_id}: {exc}")

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[GPU{gpu_id}] done -> {shard_path}")


def merge_shards(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard_path in sorted((output_dir / "records" / "shards").glob("worker_*.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records.sort(key=lambda item: int(item.get("run_index", 0)))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(record.get("status", "unknown") for record in records)
    source_counts = Counter(record.get("source_stem", "unknown") for record in records)
    elapsed = [float(record["elapsed_sec"]) for record in records if record.get("status") == "ok"]
    return {
        "runner": "zimage_base_baseline",
        "input_csv": args.input_csv,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "num_records": len(records),
        "status_counts": dict(status_counts),
        "source_counts": dict(source_counts),
        "mean_elapsed_sec": mean(elapsed),
        "config": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
            "sample_seed": args.sample_seed,
            "generation_seed": args.generation_seed,
            "device_ids": args.device_ids,
            "torch_dtype": args.torch_dtype,
            "attention_backend": args.attention_backend,
        },
    }


def write_run_config(args: argparse.Namespace, selected_cases: list[PromptCase], output_dir: Path) -> None:
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
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = load_prompt_cases(args)
    selected_cases = select_cases(all_cases, args)
    write_run_config(args, selected_cases, output_dir)

    device_ids = [int(value) for value in split_values(args.device_ids)]
    if not device_ids:
        raise ValueError("--device_ids must contain at least one device id.")

    print(f"[zimage-base] loaded {len(all_cases)} prompts")
    print(f"[zimage-base] selected {len(selected_cases)} prompts")
    print(f"[zimage-base] devices={device_ids}")
    print(f"[zimage-base] output_dir={output_dir}")

    shards = split_round_robin(selected_cases, len(device_ids))
    for rank, shard in enumerate(shards):
        print(f"[zimage-base] shard {rank}: {len(shard)} prompts")

    if len(device_ids) == 1:
        run_worker(0, device_ids[0], shards[0], args, str(output_dir))
    else:
        ctx = get_context("spawn")
        processes = []
        for rank, gpu_id in enumerate(device_ids):
            proc = ctx.Process(target=run_worker, args=(rank, gpu_id, shards[rank], args, str(output_dir)))
            proc.start()
            processes.append(proc)

        exit_codes = []
        for proc in processes:
            proc.join()
            exit_codes.append(proc.exitcode)
        if any(code != 0 for code in exit_codes):
            raise SystemExit(f"One or more base workers failed: {exit_codes}")

    records = merge_shards(output_dir)
    write_jsonl(output_dir / "records" / "results.jsonl", records)
    summary = build_summary(args, records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[zimage-base] results: {output_dir / 'records' / 'results.jsonl'}")
    print(f"[zimage-base] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
