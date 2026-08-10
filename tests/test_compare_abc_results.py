import json
from pathlib import Path

import pytest

from benchmarks.compare_abc_results import build_comparison


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
    (directory / "validity.json").write_text(
        json.dumps({"valid": True, "checks": {"reset_verified": True}}),
        encoding="utf-8",
    )


def add_identity(root: Path, group: str, source_hash: str, *, mode: str = "affinity") -> None:
    config_path = root / group / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variant = "baseline" if group == "baseline" else "candidate"
    kv_aware = "true" if mode == "affinity" and group == "candidate-on" else "false"
    active_weight = "0" if mode == "active-token" and group == "candidate-off" else "1.0"
    config["proxy_identity"] = {
        "group": group,
        "comparison_mode": mode,
        "expected_variant": variant,
        "actual_variant": variant,
        "expected_kv_aware": kv_aware,
        "actual_kv_aware": kv_aware,
        "expected_active_token_weight": active_weight,
        "actual_active_token_weight": active_weight,
        "expected_source_sha256": source_hash,
        "actual_source_sha256": source_hash,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_build_comparison_uses_candidate_off_as_affinity_baseline(tmp_path: Path):
    write_group(tmp_path, "baseline", "same-hash", 100)
    write_group(tmp_path, "candidate-off", "same-hash", 80)
    write_group(tmp_path, "candidate-on", "same-hash", 40)

    comparison = build_comparison(tmp_path)

    assert comparison["workload_sha256"] == "same-hash"
    assert comparison["valid"] is True
    assert comparison["B_vs_C"]["warm_turns.ttft_ms.p50"]["improvement"] == pytest.approx(0.5)
    assert comparison["A_vs_B"]["warm_turns.ttft_ms.p50"]["improvement"] == pytest.approx(0.2)


def test_build_comparison_rejects_different_workloads(tmp_path: Path):
    write_group(tmp_path, "baseline", "hash-a", 100)
    write_group(tmp_path, "candidate-off", "hash-a", 80)
    write_group(tmp_path, "candidate-on", "hash-b", 40)

    with pytest.raises(ValueError, match="SHA-256"):
        build_comparison(tmp_path)


def test_build_comparison_marks_failed_group_invalid(tmp_path: Path):
    for group in ("baseline", "candidate-off", "candidate-on"):
        write_group(tmp_path, group, "same-hash", 100)
    (tmp_path / "candidate-off" / "validity.json").write_text(
        json.dumps({"valid": False, "checks": {"reset_verified": False}}),
        encoding="utf-8",
    )

    comparison = build_comparison(tmp_path)

    assert comparison["valid"] is False
    assert comparison["validity"]["groups"]["candidate-off"]["checks"]["reset_verified"] is False


def test_build_comparison_validates_distinct_baseline_proxy_identity(tmp_path: Path):
    for group in ("baseline", "candidate-off", "candidate-on"):
        write_group(tmp_path, group, "same-hash", 100)
    add_identity(tmp_path, "baseline", "baseline-source")
    add_identity(tmp_path, "candidate-off", "candidate-source")
    add_identity(tmp_path, "candidate-on", "candidate-source")

    comparison = build_comparison(tmp_path)

    assert comparison["valid"] is True
    assert comparison["validity"]["proxy_identity"]["baseline_differs_from_candidate"] is True


def test_build_comparison_rejects_same_source_for_baseline_and_candidate(tmp_path: Path):
    for group in ("baseline", "candidate-off", "candidate-on"):
        write_group(tmp_path, group, "same-hash", 100)
        add_identity(tmp_path, group, "candidate-source")

    comparison = build_comparison(tmp_path)

    assert comparison["valid"] is False
    assert comparison["validity"]["proxy_identity"]["baseline_differs_from_candidate"] is False


def test_build_comparison_accepts_active_token_isolation_semantics(tmp_path: Path):
    for group in ("baseline", "candidate-off", "candidate-on"):
        write_group(tmp_path, group, "same-hash", 100)
        source_hash = "baseline-source" if group == "baseline" else "candidate-source"
        add_identity(tmp_path, group, source_hash, mode="active-token")

    comparison = build_comparison(tmp_path)

    assert comparison["valid"] is True
    assert comparison["validity"]["proxy_identity"]["comparison_mode_semantics_match"] is True
