import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.benchmark_workload import TokenTextFactory, generate_workload, write_workload_jsonl

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_session_affinity.py"
SPEC = importlib.util.spec_from_file_location("benchmark_session_affinity", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def make_result(*, cached_tokens, ttft_ms, ok=True):
    return benchmark.RequestResult(
        session_index=0,
        session_id="session",
        turn=1,
        scenario="session-long",
        phase="measure",
        stage="turn-1",
        status_code=200 if ok else 500,
        ok=ok,
        started_at_s=0.0,
        ended_at_s=(ttft_ms + 10) / 1000,
        ttfb_ms=ttft_ms,
        ttft_ms=ttft_ms,
        e2e_ms=ttft_ms + 10,
        prompt_tokens=100,
        completion_tokens=8,
        client_cached_tokens=cached_tokens,
        output_chars=8,
        error=None if ok else "failed",
    )


def test_latency_summary_separates_errors_and_reports_percentiles():
    summary = benchmark.latency_summary(
        [
            make_result(cached_tokens=0, ttft_ms=100.0),
            make_result(cached_tokens=80, ttft_ms=50.0),
            make_result(cached_tokens=None, ttft_ms=25.0, ok=False),
        ]
    )

    assert summary["requests"] == 3
    assert summary["successes"] == 2
    assert summary["errors"] == 1
    assert summary["ttft_ms"]["p50"] == 75.0
    assert summary["client_cached_tokens"]["mean"] == 40.0
    assert summary["cached_token_request_rate"] == 0.5
    assert summary["cached_token_ratio"] == 0.4
    assert summary["cached_tokens_field_rate"] == 1.0


def test_latency_summary_treats_missing_cached_tokens_as_cache_miss():
    summary = benchmark.latency_summary(
        [
            make_result(cached_tokens=None, ttft_ms=100.0),
            make_result(cached_tokens=None, ttft_ms=90.0),
            make_result(cached_tokens=80, ttft_ms=50.0),
            make_result(cached_tokens=None, ttft_ms=40.0),
        ]
    )

    assert summary["cached_token_request_rate"] == 0.25
    assert summary["cached_token_ratio"] == 0.2
    assert summary["client_cached_tokens"]["mean"] == 20.0
    assert summary["client_computed_tokens"]["mean"] == 80.0
    assert summary["cached_tokens_field_rate"] == 0.25


def test_parse_prometheus_aggregates_backends_and_ignores_unrelated_metrics():
    parsed = benchmark.parse_prometheus(
        """
# HELP vllm:prefix_cache_hits Prefix cache hits
vllm:prefix_cache_hits{instance="p0"} 10
vllm:prefix_cache_hits{instance="p1"} 20
vllm:request_queue_time_seconds_sum 4.5
python_gc_objects_collected_total 999
"""
    )

    assert parsed == {
        "vllm:prefix_cache_hits": 30.0,
        "vllm:request_queue_time_seconds_sum": 4.5,
    }


def test_health_summary_can_exclude_cache_fill_samples(tmp_path: Path):
    samples = tmp_path / "samples.jsonl"
    records = [
        {
            "timestamp": 1.0,
            "kind": "proxy_health",
            "value": {"prefix_affinity_stats": {"lookups": 10, "hits": 2}},
        },
        {
            "timestamp": 2.0,
            "kind": "proxy_health",
            "value": {"prefix_affinity_stats": {"lookups": 100, "hits": 50}},
        },
        {
            "timestamp": 3.0,
            "kind": "proxy_health",
            "value": {"prefix_affinity_stats": {"lookups": 112, "hits": 59}},
        },
    ]
    samples.write_text("".join(f"{json.dumps(record)}\n" for record in records))

    summary = benchmark.summarize_health_samples(samples, start_timestamp=2.0)

    assert summary["prefix_affinity_stats_delta"]["lookups"] == 12
    assert summary["prefix_affinity_stats_delta"]["hits"] == 9
    assert summary["prefix_affinity_stats_delta"]["derived_prefix_hit_rate"] == 0.75


def test_verify_reset_across_prefills_primes_checks_and_cleans(tmp_path: Path, monkeypatch):
    base_urls = ["http://p0:7100", "http://p1:7100"]
    calls = {url: 0 for url in base_urls}
    reset_calls = []

    def fake_probe(base_url, *_args, **_kwargs):
        calls[base_url] += 1
        cached = 256 if calls[base_url] == 2 else None
        return {
            "base_url": base_url,
            "status_code": 200,
            "prompt_tokens": 300,
            "cached_tokens": cached,
        }

    monkeypatch.setattr(benchmark, "send_reset_probe", fake_probe)
    monkeypatch.setattr(
        benchmark,
        "wait_for_prefills_idle",
        lambda *args, **kwargs: {"wait_seconds": 0.0, "backends": {}},
    )

    def fake_reset(*args, **kwargs):
        reset_calls.append(args[0])
        return {
            "url": args[0],
            "status_code": 200,
            "body": {
                "success": True,
                "backends": [{"target": base_url, "status_code": 200} for base_url in base_urls],
            },
        }

    monkeypatch.setattr(benchmark, "reset_cache", fake_reset)

    result = benchmark.verify_reset_across_prefills(
        base_urls=base_urls,
        reset_url="http://proxy/reset_prefix_cache",
        model="model",
        label="baseline",
        api_key=None,
        timeout=10,
        drain_timeout=5,
        verify=True,
        output_dir=tmp_path,
    )

    assert result["verified"] is True
    assert reset_calls == ["http://proxy/reset_prefix_cache"] * 2
    assert [probe["phase"] for probe in result["probes"]].count("post-reset-miss") == 2
    assert json.loads((tmp_path / "reset-validation.json").read_text())["verified"] is True
    assert len((tmp_path / "reset-probe-requests.jsonl").read_text().splitlines()) == 6


def test_validate_reset_fanout_rejects_missing_prefill():
    result = {
        "body": {
            "success": True,
            "backends": [{"target": "http://p0:7100", "status_code": 200}],
        }
    }

    with pytest.raises(RuntimeError, match="reported 1 backends"):
        benchmark.validate_reset_fanout(result, ["http://p0:7100", "http://p1:7100"])


def test_main_runs_frozen_workload_with_separate_warmup_and_cache_fill(
    tmp_path: Path,
    monkeypatch,
):
    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, **kwargs):
            chunks = [
                {"choices": [{"delta": {"content": "ok"}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 80},
                    },
                },
            ]
            return [*(f"data: {json.dumps(chunk)}" for chunk in chunks), "data: [DONE]"]

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

        def close(self):
            return None

    profile = {
        "version": 1,
        "name": "integration",
        "scenario": "session-long",
        "seed": 5,
        "data": {
            "sessions": 2,
            "turns": 2,
            "system_prompt_tokens": 8,
            "turn_input_tokens": 4,
            "fixed_assistant_text": "Fixed.",
        },
        "load": {"type": "turn-barrier", "concurrency": 2},
        "request": {"max_tokens": 1, "temperature": 0.0},
    }
    workload = tmp_path / "workload.jsonl"
    write_workload_jsonl(workload, generate_workload(profile, TokenTextFactory()))
    output = tmp_path / "result"
    monkeypatch.setattr(benchmark, "make_http_session", lambda concurrency: FakeSession())
    monkeypatch.setattr(
        benchmark,
        "reset_cache",
        lambda *args, **kwargs: {"url": "http://benchmark.invalid/reset_prefix_cache", "status_code": 200},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_session_affinity.py",
            "--base-url",
            "http://benchmark.invalid",
            "--model",
            "test-model",
            "--workload-file",
            str(workload),
            "--output-dir",
            str(output),
            "--concurrency",
            "2",
            "--system-warmup-requests",
            "1",
            "--reset-before",
            "--no-health-sampling",
        ],
    )

    assert benchmark.main() == 0

    summary = json.loads((output / "summary.json").read_text())
    assert summary["cache_fill"]["requests"] == 2
    assert summary["measured"]["requests"] == 2
    assert summary["warm_turns"]["client_cached_tokens"]["mean"] == 80.0
    assert len((output / "system-warmup.jsonl").read_text().splitlines()) == 1
    assert len((output / "requests.jsonl").read_text().splitlines()) == 4
