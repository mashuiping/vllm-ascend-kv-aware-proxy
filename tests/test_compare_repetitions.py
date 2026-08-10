import json

import pytest

from benchmarks.compare_repetitions import aggregate_repetitions


def write_experiment(
    root,
    name,
    improvement,
    *,
    valid=True,
    mode="active-token",
    group_order=None,
    seed=None,
):
    directory = root / name
    (directory / "baseline").mkdir(parents=True)
    (directory / "baseline" / "config.json").write_text(
        json.dumps({"proxy_identity": {"comparison_mode": mode}}), encoding="utf-8"
    )
    (directory / "comparison.json").write_text(
        json.dumps(
            {
                "valid": valid,
                "B_vs_C": {
                    "metric": {
                        "baseline": 10,
                        "treatment": 10 * (1 - improvement),
                        "improvement": improvement,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    if group_order is not None or seed is not None:
        (directory / "metadata.json").write_text(
            json.dumps({"group_order": group_order, "workload_seed": seed}), encoding="utf-8"
        )
    return directory


def test_aggregate_repetitions_reports_paired_signs_and_bootstrap_ci(tmp_path):
    directories = [
        write_experiment(tmp_path, "run-1", 0.1),
        write_experiment(tmp_path, "run-2", 0.2),
        write_experiment(tmp_path, "run-3", -0.1),
    ]

    result = aggregate_repetitions(directories, "B_vs_C", ["metric"])

    metric = result["metrics"]["metric"]
    assert result["comparison_mode"] == "active-token"
    assert metric["mean_improvement"] == pytest.approx(0.0666666667)
    assert metric["median_improvement"] == pytest.approx(0.1)
    assert metric["treatment_wins"] == 2
    assert len(metric["mean_improvement_95pct_bootstrap_ci"]) == 2


def test_aggregate_repetitions_rejects_invalid_input(tmp_path):
    directory = write_experiment(tmp_path, "invalid", 0.1, valid=False)

    with pytest.raises(ValueError, match="invalid experiment"):
        aggregate_repetitions([directory], "B_vs_C", ["metric"])


def test_aggregate_repetitions_reports_complete_six_run_design(tmp_path):
    orders = [
        ["baseline", "candidate-off", "candidate-on"],
        ["baseline", "candidate-on", "candidate-off"],
        ["candidate-off", "baseline", "candidate-on"],
        ["candidate-off", "candidate-on", "baseline"],
        ["candidate-on", "baseline", "candidate-off"],
        ["candidate-on", "candidate-off", "baseline"],
    ]
    directories = [
        write_experiment(
            tmp_path,
            f"run-{index}",
            0.1,
            group_order=order,
            seed=1000 + index,
        )
        for index, order in enumerate(orders)
    ]

    result = aggregate_repetitions(directories, "B_vs_C", ["metric"])

    design = result["experimental_design"]
    assert design["all_six_group_orders_covered"] is True
    assert design["seed_varied_every_repetition"] is True
    assert design["recommended_six_run_design_complete"] is True
