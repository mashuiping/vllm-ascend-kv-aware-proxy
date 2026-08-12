import json
from pathlib import Path

import pytest

from benchmarks.judge_affinity_repetitions import judge

METRIC = "warm_turns.ttft_ms.p95"


def write_run(root: Path, name: str, *, valid: bool, ab: float, ac: float) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "comparison.json").write_text(
        json.dumps(
            {
                "valid": valid,
                "A_vs_B": {METRIC: {"relative_change": ab}},
                "A_vs_C": {METRIC: {"relative_change": ac}},
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_consistent_effect_beyond_noise_band_is_reported(tmp_path: Path):
    dirs = [
        write_run(tmp_path, "run0", valid=True, ab=0.01, ac=0.08),
        write_run(tmp_path, "run1", valid=True, ab=-0.02, ac=0.06),
        write_run(tmp_path, "run2", valid=True, ab=0.015, ac=0.09),
    ]

    result = judge(dirs)
    verdict = result["metrics"][METRIC]
    assert verdict["verdict"] == "effect"
    assert verdict["noise_band_abs_max_A_vs_B"] == pytest.approx(0.02)
    assert verdict["direction_consistent"] is True
    assert result["all_valid"] is True
    assert result["judged_run_count"] == 3


def test_effect_within_noise_band_or_inconsistent_direction_is_noise(tmp_path: Path):
    within_band = [
        write_run(tmp_path, "band0", valid=True, ab=0.05, ac=0.03),
        write_run(tmp_path, "band1", valid=True, ab=-0.06, ac=0.02),
    ]
    assert judge(within_band)["metrics"][METRIC]["verdict"] == "noise"

    inconsistent = [
        write_run(tmp_path, "dir0", valid=True, ab=0.01, ac=0.08),
        write_run(tmp_path, "dir1", valid=True, ab=0.01, ac=-0.07),
    ]
    assert judge(inconsistent)["metrics"][METRIC]["verdict"] == "noise"


def test_invalid_runs_are_excluded_from_judgment(tmp_path: Path):
    dirs = [
        write_run(tmp_path, "run0", valid=True, ab=0.01, ac=0.08),
        write_run(tmp_path, "run1", valid=True, ab=0.015, ac=0.09),
        # This run would blow the noise band open and flip the direction.
        write_run(tmp_path, "run2", valid=False, ab=0.5, ac=-0.5),
    ]

    result = judge(dirs)
    assert result["all_valid"] is False
    assert result["excluded_invalid_runs"] == ["run2"]
    assert result["judged_run_count"] == 2
    verdict = result["metrics"][METRIC]
    assert verdict["noise_band_abs_max_A_vs_B"] == pytest.approx(0.015)
    assert verdict["verdict"] == "effect"


def test_all_invalid_runs_raise(tmp_path: Path):
    dirs = [write_run(tmp_path, "run0", valid=False, ab=0.0, ac=0.0)]
    with pytest.raises(ValueError, match="no valid repetitions"):
        judge(dirs)


def test_missing_metric_reports_no_data(tmp_path: Path):
    directory = tmp_path / "run0"
    directory.mkdir()
    (directory / "comparison.json").write_text(
        json.dumps({"valid": True, "A_vs_B": {}, "A_vs_C": {}}),
        encoding="utf-8",
    )
    result = judge([directory])
    assert result["metrics"][METRIC] == {"verdict": "no-data"}
