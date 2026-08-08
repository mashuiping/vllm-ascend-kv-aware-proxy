import json

from benchmarks.render_benchmark_report import render


def test_render_benchmark_report(tmp_path):
    summary = {
        "overall": {"requests": 2, "success_rate": 1.0},
        "warm_turns": {
            "ttft_ms": {"p95": 100.0},
            "e2e_ms": {"p95": 200.0},
            "cached_token_ratio": 0.5,
            "request_throughput_per_second": 2.0,
        },
    }
    for group in ("baseline", "candidate-off", "candidate-on"):
        group_dir = tmp_path / group
        group_dir.mkdir()
        (group_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (group_dir / "config.json").write_text("{}", encoding="utf-8")
        (group_dir / "validity.json").write_text('{"valid": true}', encoding="utf-8")

    comparison = {
        "valid": True,
        "workload_sha256": "abc123",
        "A_vs_B": {},
        "B_vs_C": {
            "warm_turns.ttft_ms.p95": {
                "baseline": 100.0,
                "treatment": 80.0,
                "relative_change": -0.2,
                "improvement": 0.2,
            }
        },
        "A_vs_C": {},
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    (tmp_path / "workload.jsonl.manifest.json").write_text(
        json.dumps({"token_count_verified": True, "scenario": "session-long", "requests": 2}),
        encoding="utf-8",
    )

    output = tmp_path / "report.html"
    render(tmp_path, output)
    page = output.read_text(encoding="utf-8")
    assert "Benchmark report" in page
    assert "B → C" in page
    assert "20.0%" in page
    assert "positive" in page
    assert "Experiment validity checks passed" in page


def test_render_benchmark_report_marks_invalid_comparison_as_diagnostic(tmp_path):
    summary = {
        "overall": {"requests": 2, "success_rate": 1.0},
        "warm_turns": {
            "ttft_ms": {"p95": 100.0},
            "e2e_ms": {"p95": 200.0},
            "cached_token_ratio": 0.5,
            "request_throughput_per_second": 2.0,
        },
    }
    for group in ("baseline", "candidate-off", "candidate-on"):
        group_dir = tmp_path / group
        group_dir.mkdir()
        (group_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (group_dir / "config.json").write_text("{}", encoding="utf-8")

    comparison = {
        "valid": False,
        "workload_sha256": "abc123",
        "A_vs_B": {},
        "B_vs_C": {
            "warm_turns.ttft_ms.p95": {
                "baseline": 100.0,
                "treatment": 80.0,
                "relative_change": -0.2,
                "improvement": 0.2,
            }
        },
        "A_vs_C": {},
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")

    output = tmp_path / "report.html"
    render(tmp_path, output)
    page = output.read_text(encoding="utf-8")

    assert "Experiment is invalid" in page
    assert '<td class="neutral">+20.0%</td>' in page
