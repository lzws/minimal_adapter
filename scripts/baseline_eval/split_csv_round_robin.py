#!/usr/bin/env python
"""把评测 CSV 按 round-robin 切成多份。

用于单 GPU adapter 测试脚本的多卡并行包装：先把一个输入 CSV 切成 N 个 shard，
再为每张 GPU 启动一个测试进程。这样不需要改动 adapter 测试逻辑。
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "cp936")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a prompt CSV into round-robin shards.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=20260715)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {path}")
                return list(reader.fieldnames), rows
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to decode {path} with encodings={CSV_ENCODINGS}") from last_error


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = read_csv(input_csv)
    if args.num_prompts is not None:
        if args.num_prompts <= 0:
            raise ValueError("--num_prompts must be positive when set")
        rng = random.Random(args.sample_seed)
        if args.num_prompts < len(rows):
            rows = rng.sample(rows, k=args.num_prompts)

    shards: list[list[dict[str, str]]] = [[] for _ in range(args.num_shards)]
    for index, row in enumerate(rows):
        shards[index % args.num_shards].append(row)

    manifest_path = output_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for shard_index, shard_rows in enumerate(shards):
            shard_path = output_dir / f"shard_{shard_index:03d}.csv"
            with shard_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(shard_rows)
            manifest.write(f"{shard_index}\t{shard_path}\t{len(shard_rows)}\n")
            print(f"{shard_index}\t{shard_path}\t{len(shard_rows)}")


if __name__ == "__main__":
    main()
