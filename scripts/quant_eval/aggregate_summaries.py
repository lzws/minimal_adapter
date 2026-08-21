#!/usr/bin/env python
"""把多个图片审核 summary.json 合并成一张实验对比表。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate quant-eval summary files.")
    parser.add_argument("--summary_root", default="outputs/quant_eval")
    parser.add_argument("--output_csv", default="outputs/quant_eval/comparison.csv")
    parser.add_argument("--output_json", default="outputs/quant_eval/comparison.json")
    return parser.parse_args()


def flatten_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """提取最常用的总体指标，缺失指标保留为空。"""
    overall = dict(payload.get("overall", {}))
    row: dict[str, Any] = {
        "experiment": payload.get("experiment", ""),
        "task": payload.get("task", ""),
        "target_ip": payload.get("target_ip", ""),
        "num_scanned": overall.get("num_scanned", ""),
        "num_success": overall.get("num_success", ""),
        "num_error": overall.get("num_error", ""),
        "metadata_match_rate_percent": overall.get("metadata_match_rate_percent", ""),
        "porn_strict_hit_rate_percent": overall.get("porn_strict_hit_rate_percent", ""),
        "porn_any_hit_rate_percent": overall.get("porn_any_hit_rate_percent", ""),
        "porn_strict_suppression_rate_percent": overall.get(
            "porn_strict_suppression_rate_percent", ""
        ),
        "gore_hit_rate_percent": overall.get("gore_hit_rate_percent", ""),
        "violence_or_gore_hit_rate_percent": overall.get(
            "violence_or_gore_hit_rate_percent", ""
        ),
        "gore_suppression_rate_percent": overall.get("gore_suppression_rate_percent", ""),
        "target_ip_hit_rate_percent": overall.get("target_ip_hit_rate_percent", ""),
        "target_ip_suppression_rate_percent": overall.get(
            "target_ip_suppression_rate_percent", ""
        ),
        "related_ip_hit_rate_percent": overall.get("related_ip_hit_rate_percent", ""),
        "any_ip_hit_rate_percent": overall.get("any_ip_hit_rate_percent", ""),
        "benign_false_positive_rate_percent": overall.get(
            "benign_false_positive_rate_percent", ""
        ),
        "benign_clean_rate_percent": overall.get("benign_clean_rate_percent", ""),
    }
    return row


def main() -> None:
    args = parse_args()
    root = Path(args.summary_root)
    summary_paths = sorted(
        path for path in root.glob("*/summary.json") if path.resolve() != Path(args.output_json).resolve()
    )
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in summary_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payloads.append(payload)
        rows.append(flatten_summary(payload))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "experiment",
        "task",
        "target_ip",
        "num_scanned",
        "num_success",
        "num_error",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payloads, handle, ensure_ascii=False, indent=2)

    print(f"[quant-eval] aggregated {len(rows)} summaries")
    print(f"[quant-eval] comparison csv: {output_csv}")
    print(f"[quant-eval] comparison json: {output_json}")


if __name__ == "__main__":
    main()
