import importlib.util
import sys
from pathlib import Path

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
        status_code=200 if ok else 500,
        ok=ok,
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
