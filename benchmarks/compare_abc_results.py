#!/usr/bin/env python3
"""Validate and compare one completed A/B/C benchmark experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from benchmark_session_affinity import compare_summaries
except ModuleNotFoundError:  # Imported as benchmarks.compare_abc_results in tests.
    from benchmarks.benchmark_session_affinity import compare_summaries


GROUPS = ("baseline", "candidate-off", "candidate-on")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_comparison(experiment_dir: Path) -> dict[str, Any]:
    summaries = {group: load_json(experiment_dir / group / "summary.json") for group in GROUPS}
    configs = {group: load_json(experiment_dir / group / "config.json") for group in GROUPS}
    raw_hashes = [config.get("workload_sha256") for config in configs.values()]
    workload_hashes = {value for value in raw_hashes if isinstance(value, str) and value}
    if len(workload_hashes) != 1 or len(workload_hashes) != len(set(raw_hashes)):
        raise ValueError(f"A/B/C workload SHA-256 values differ or are missing: {workload_hashes}")
    group_validity: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        path = experiment_dir / group / "validity.json"
        group_validity[group] = (
            load_json(path) if path.is_file() else {"valid": False, "checks": {"validity_artifact_present": False}}
        )
    valid = all(value.get("valid") is True for value in group_validity.values())
    return {
        "valid": valid,
        "validity": {"groups": group_validity},
        "workload_sha256": workload_hashes.pop(),
        "A_vs_B": compare_summaries(summaries["baseline"], summaries["candidate-off"]),
        "B_vs_C": compare_summaries(summaries["candidate-off"], summaries["candidate-on"]),
        "A_vs_C": compare_summaries(summaries["baseline"], summaries["candidate-on"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison(args.experiment_dir)
    output = args.experiment_dir / "comparison.json"
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
