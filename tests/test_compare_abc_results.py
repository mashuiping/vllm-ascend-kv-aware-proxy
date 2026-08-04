import json
from pathlib import Path

import pytest

from scripts.compare_abc_results import build_comparison


def write_group(root: Path, group: str, workload_hash: str, ttft: float) -> None:
    directory = root / group
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"workload_sha256": workload_hash}),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "warm_turns": {
                    "ttft_ms": {"p50": ttft, "p95": ttft, "p99": ttft},
                    "e2e_ms": {"p95": ttft},
                },
                "overall": {"success_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )


def test_build_comparison_uses_candidate_off_as_affinity_baseline(tmp_path: Path):
    write_group(tmp_path, "baseline", "same-hash", 100)
    write_group(tmp_path, "candidate-off", "same-hash", 80)
    write_group(tmp_path, "candidate-on", "same-hash", 40)

    comparison = build_comparison(tmp_path)

    assert comparison["workload_sha256"] == "same-hash"
    assert comparison["B_vs_C"]["warm_turns.ttft_ms.p50"]["improvement"] == pytest.approx(0.5)
    assert comparison["A_vs_B"]["warm_turns.ttft_ms.p50"]["improvement"] == pytest.approx(0.2)


def test_build_comparison_rejects_different_workloads(tmp_path: Path):
    write_group(tmp_path, "baseline", "hash-a", 100)
    write_group(tmp_path, "candidate-off", "hash-a", 80)
    write_group(tmp_path, "candidate-on", "hash-b", 40)

    with pytest.raises(ValueError, match="SHA-256"):
        build_comparison(tmp_path)
