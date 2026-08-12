#!/usr/bin/env python3
"""Apply the same-policy noise-gate verdict to A/B/C repetitions.

For each metric, the per-run |A vs B| relative delta defines the noise band
(B is decision-identical to A). An effect is claimed only when the mean
|A vs C| delta exceeds the band and every repetition moves in the same
direction. This encodes the judgment standard from
docs/analysis-active-token-weight.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "warm_turns.ttft_ms.p95",
    "warm_turns.e2e_ms.p95",
    "warm_turns.request_throughput_per_second",
    "warm_turns.cached_token_ratio",
    "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_queue_time_seconds",
    "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_prefill_time_seconds",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def relative_delta(comparison: dict[str, Any], metric: str) -> float | None:
    entry = comparison.get(metric)
    if not isinstance(entry, dict):
        return None
    value = entry.get("relative_change")
    return float(value) if isinstance(value, (int, float)) else None


def judge(experiment_dirs: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for directory in experiment_dirs:
        comparison = load_json(directory / "comparison.json")
        rows.append(
            {
                "run": directory.name,
                "valid": comparison.get("valid") is True,
                "A_vs_B": comparison.get("A_vs_B") or {},
                "A_vs_C": comparison.get("A_vs_C") or {},
            }
        )

    # The judgment standard requires valid repetitions; a run that failed the
    # proxy-identity or reset checks must not contribute to the noise band.
    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        raise ValueError("no valid repetitions to judge; every comparison.json has valid=false")

    verdicts: dict[str, Any] = {}
    for metric in METRICS:
        ab = [relative_delta(row["A_vs_B"], metric) for row in valid_rows]
        ac = [relative_delta(row["A_vs_C"], metric) for row in valid_rows]
        ab_values = [value for value in ab if value is not None]
        ac_values = [value for value in ac if value is not None]
        if not ab_values or not ac_values:
            verdicts[metric] = {"verdict": "no-data"}
            continue
        noise_band = max(abs(value) for value in ab_values)
        mean_effect = statistics.fmean(ac_values)
        consistent = all(value < 0 for value in ac_values) or all(value > 0 for value in ac_values)
        exceeds = abs(mean_effect) > noise_band
        verdicts[metric] = {
            "noise_band_abs_max_A_vs_B": noise_band,
            "per_run_A_vs_C": ac_values,
            "mean_A_vs_C": mean_effect,
            "direction_consistent": consistent,
            "exceeds_noise_band": exceeds,
            "verdict": "effect" if (consistent and exceeds) else "noise",
        }

    return {
        "runs": [{"run": row["run"], "valid": row["valid"]} for row in rows],
        "all_valid": all(row["valid"] for row in rows),
        "excluded_invalid_runs": [row["run"] for row in rows if not row["valid"]],
        "judged_run_count": len(valid_rows),
        "metrics": verdicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    result = judge(args.experiment_dirs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
