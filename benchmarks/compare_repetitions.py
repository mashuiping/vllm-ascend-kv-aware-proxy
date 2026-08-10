#!/usr/bin/env python3
"""Aggregate paired A/B/C comparisons across valid experiment repetitions."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

DEFAULT_METRICS = (
    "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_queue_time_seconds",
    "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_prefill_time_seconds",
    "measurement_metrics_delta_by_role.decode.derived:mean:vllm:request_queue_time_seconds",
    "warm_turns.ttft_ms.p95",
    "warm_turns.e2e_ms.p95",
    "warm_turns.request_throughput_per_second",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_mean_ci(values: list[float], *, samples: int = 10_000, seed: int = 20260810) -> list[float]:
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def aggregate_repetitions(
    experiment_dirs: list[Path],
    comparison_name: str,
    metrics: list[str],
) -> dict[str, Any]:
    comparisons: list[tuple[Path, dict[str, Any]]] = []
    modes: set[str] = set()
    design_rows: list[dict[str, Any]] = []
    for directory in experiment_dirs:
        comparison = load_json(directory / "comparison.json")
        if comparison.get("valid") is not True:
            raise ValueError(f"invalid experiment cannot be aggregated: {directory}")
        baseline_config = load_json(directory / "baseline" / "config.json")
        identity = baseline_config.get("proxy_identity") or {}
        mode = identity.get("comparison_mode")
        if mode:
            modes.add(str(mode))
        metadata_path = directory / "metadata.json"
        manifest_path = directory / "workload.jsonl.manifest.json"
        metadata = load_json(metadata_path) if metadata_path.is_file() else {}
        manifest = load_json(manifest_path) if manifest_path.is_file() else {}
        group_order = metadata.get("group_order")
        design_rows.append(
            {
                "experiment": directory.name,
                "group_order": group_order if isinstance(group_order, list) else None,
                "workload_seed": metadata.get("workload_seed", manifest.get("seed")),
                "workload_sha256": manifest.get("sha256"),
            }
        )
        comparisons.append((directory, comparison))
    if len(modes) > 1:
        raise ValueError(f"comparison modes differ across repetitions: {sorted(modes)}")

    result_metrics: dict[str, Any] = {}
    for metric in metrics:
        rows = []
        for directory, comparison in comparisons:
            value = (comparison.get(comparison_name) or {}).get(metric)
            if not isinstance(value, dict) or not isinstance(value.get("improvement"), (int, float)):
                raise ValueError(f"missing paired metric {comparison_name}.{metric}: {directory}")
            rows.append(
                {
                    "experiment": directory.name,
                    "baseline": value.get("baseline"),
                    "treatment": value.get("treatment"),
                    "improvement": float(value["improvement"]),
                }
            )
        improvements = [row["improvement"] for row in rows]
        result_metrics[metric] = {
            "repetitions": len(rows),
            "mean_improvement": statistics.fmean(improvements),
            "median_improvement": statistics.median(improvements),
            "mean_improvement_95pct_bootstrap_ci": bootstrap_mean_ci(improvements),
            "treatment_wins": sum(value > 0 for value in improvements),
            "ties": sum(value == 0 for value in improvements),
            "rows": rows,
        }
    observed_orders = {
        tuple(row["group_order"])
        for row in design_rows
        if isinstance(row["group_order"], list) and len(row["group_order"]) == 3
    }
    observed_seeds = {row["workload_seed"] for row in design_rows if row["workload_seed"] is not None}
    return {
        "valid": True,
        "comparison": comparison_name,
        "comparison_mode": next(iter(modes), None),
        "experiments": [directory.name for directory in experiment_dirs],
        "experimental_design": {
            "all_six_group_orders_covered": len(observed_orders) == 6,
            "distinct_group_orders": len(observed_orders),
            "distinct_workload_seeds": len(observed_seeds),
            "seed_varied_every_repetition": len(observed_seeds) == len(experiment_dirs),
            "recommended_six_run_design_complete": (
                len(experiment_dirs) >= 6 and len(observed_orders) == 6 and len(observed_seeds) == len(experiment_dirs)
            ),
            "rows": design_rows,
        },
        "metrics": result_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--comparison", choices=("A_vs_B", "B_vs_C", "A_vs_C"), default="B_vs_C")
    parser.add_argument("--metric", action="append", dest="metrics")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = aggregate_repetitions(args.experiment_dirs, args.comparison, args.metrics or list(DEFAULT_METRICS))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
