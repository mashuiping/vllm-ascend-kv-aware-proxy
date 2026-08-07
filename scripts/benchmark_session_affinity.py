#!/usr/bin/env python3
"""Benchmark session-affinity routing against an OpenAI-compatible PD proxy.

The workload advances all sessions one turn at a time. This preserves
conversation order within a session while allowing independent sessions to run
concurrently. Besides per-request latency, the script can sample the proxy
health endpoint and every P/D vLLM /metrics endpoint.

Example:
  OPENAI_API_KEY=... python scripts/benchmark_session_affinity.py \
    --base-url http://proxy:9000 \
    --model glm5.1 \
    --scenario session-long \
    --sessions 256 --turns 6 --concurrency 64 \
    --prefix-words 1024 --max-tokens 16 \
    --prefill-metrics-url http://p0:7100/metrics \
    --prefill-metrics-url http://p1:7100/metrics \
    --prefill-metrics-url http://p2:7100/metrics \
    --prefill-metrics-url http://p3:7100/metrics \
    --decode-metrics-url http://d0:7200/metrics \
    --output-dir results/affinity-on
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter

try:
    from benchmark_workload import PlannedRequest, load_workload_jsonl
except ModuleNotFoundError:  # Imported as scripts.benchmark_session_affinity in tests.
    from scripts.benchmark_workload import PlannedRequest, load_workload_jsonl

PROMETHEUS_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]Inf)"
)

INTERESTING_METRIC_PREFIXES = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prompt_tokens",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_prefill_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:request_success",
)


@dataclass
class RequestResult:
    session_index: int
    session_id: str
    turn: int
    scenario: str
    phase: str
    stage: str
    status_code: int | None
    ok: bool
    started_at_s: float
    ended_at_s: float
    ttfb_ms: float | None
    ttft_ms: float | None
    e2e_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    client_cached_tokens: int | None
    output_chars: int
    error: str | None


class JsonlWriter:
    def __init__(self, path: Path):
        self._file = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, value: Any) -> None:
        with self._lock:
            self._file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def latency_summary(results: list[RequestResult]) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    ttft = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    ttfb = [result.ttfb_ms for result in successful if result.ttfb_ms is not None]
    e2e = [result.e2e_ms for result in successful]
    prompt_tokens = [float(result.prompt_tokens) for result in successful if result.prompt_tokens is not None]
    # vLLM omits prompt_tokens_details.cached_tokens on a cache miss in the
    # deployment used by this benchmark. Preserve that distinction in the raw
    # JSONL, but normalize the missing value to zero for aggregate cache
    # metrics. Excluding it would make the denominator contain hits only and
    # report cached_token_request_rate=1.0 whenever any request hit.
    cache_eligible = [result for result in successful if result.prompt_tokens is not None]
    cached_tokens = [float(result.client_cached_tokens or 0) for result in cache_eligible]
    cached_tokens_field_count = sum(result.client_cached_tokens is not None for result in cache_eligible)
    cache_pairs = [(float(result.client_cached_tokens or 0), float(result.prompt_tokens)) for result in cache_eligible]
    completion_tokens = [
        float(result.completion_tokens) for result in successful if result.completion_tokens is not None
    ]
    computed_tokens = [float(result.prompt_tokens - (result.client_cached_tokens or 0)) for result in cache_eligible]
    started_at = [result.started_at_s for result in successful]
    ended_at = [result.ended_at_s for result in successful]
    window_seconds = max(ended_at) - min(started_at) if started_at and ended_at else 0.0

    def describe(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    return {
        "requests": len(results),
        "successes": len(successful),
        "errors": len(results) - len(successful),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "ttfb_ms": describe(ttfb),
        "ttft_ms": describe(ttft),
        "e2e_ms": describe(e2e),
        "prompt_tokens": describe(prompt_tokens),
        "client_cached_tokens": describe(cached_tokens),
        "client_computed_tokens": describe(computed_tokens),
        "completion_tokens": describe(completion_tokens),
        "cached_tokens_field_rate": (cached_tokens_field_count / len(cache_eligible) if cache_eligible else None),
        "cached_token_request_rate": (
            sum(value > 0 for value in cached_tokens) / len(cached_tokens) if cached_tokens else None
        ),
        "cached_token_ratio": (
            sum(cached for cached, _ in cache_pairs) / sum(prompt for _, prompt in cache_pairs)
            if cache_pairs and sum(prompt for _, prompt in cache_pairs) > 0
            else None
        ),
        "measurement_window_seconds": window_seconds,
        "request_throughput_per_second": (len(successful) / window_seconds if window_seconds > 0 else None),
        "input_token_throughput_per_second": (sum(prompt_tokens) / window_seconds if window_seconds > 0 else None),
        "output_token_throughput_per_second": (sum(completion_tokens) / window_seconds if window_seconds > 0 else None),
    }


def parse_prometheus(text: str) -> dict[str, float]:
    """Aggregate each metric name across labels.

    Histograms remain split into their standard _bucket, _sum, and _count
    series. Per-backend attribution is preserved by storing every URL
    separately in the result files.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if not name.startswith(INTERESTING_METRIC_PREFIXES):
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] = totals.get(name, 0.0) + value
    return totals


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in sorted(set(before) | set(after)):
        if name.endswith(("_bucket", "_sum", "_count")) or name in (
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_hits",
            "vllm:prompt_tokens",
            "vllm:prompt_tokens_cached",
            "vllm:request_success",
        ):
            result[name] = after.get(name, 0.0) - before.get(name, 0.0)
    queries = result.get("vllm:prefix_cache_queries", 0.0)
    hits = result.get("vllm:prefix_cache_hits", 0.0)
    if queries > 0:
        result["derived:prefix_cache_hit_rate"] = hits / queries
    computed_sum = result.get("vllm:request_prefill_kv_computed_tokens_sum", 0.0)
    computed_count = result.get("vllm:request_prefill_kv_computed_tokens_count", 0.0)
    if computed_count > 0:
        result["derived:mean_prefill_computed_tokens"] = computed_sum / computed_count
    for metric in (
        "vllm:request_prefill_time_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
    ):
        count = result.get(f"{metric}_count", 0.0)
        if count > 0:
            result[f"derived:mean:{metric}"] = result.get(f"{metric}_sum", 0.0) / count
    return result


def aggregate_metric_snapshots(
    urls: list[str],
    snapshots: dict[str, dict[str, float]],
) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for url in urls:
        for name, value in snapshots.get(url, {}).items():
            aggregate[name] = aggregate.get(name, 0.0) + value
    return aggregate


class EndpointSampler:
    def __init__(
        self,
        health_url: str | None,
        metrics_urls: list[str],
        interval: float,
        timeout: float,
        verify: bool,
        writer: JsonlWriter,
    ):
        self.health_url = health_url
        self.metrics_urls = metrics_urls
        self.interval = interval
        self.timeout = timeout
        self.verify = verify
        self.writer = writer
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.before: dict[str, dict[str, float]] = {}
        self.after: dict[str, dict[str, float]] = {}

    def _get_health(self) -> dict[str, Any] | None:
        if not self.health_url:
            return None
        response = requests.get(
            self.health_url,
            timeout=self.timeout,
            verify=self.verify,
        )
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, dict) else {"value": value}

    def _get_metric(self, url: str) -> tuple[str, dict[str, float], str]:
        response = requests.get(url, timeout=self.timeout, verify=self.verify)
        response.raise_for_status()
        return url, parse_prometheus(response.text), response.text

    def snapshot_metrics(self, raw_dir: Path, suffix: str) -> dict[str, dict[str, float]]:
        snapshots: dict[str, dict[str, float]] = {}
        for index, url in enumerate(self.metrics_urls):
            try:
                metric_url, parsed, raw = self._get_metric(url)
                snapshots[metric_url] = parsed
                (raw_dir / f"metrics-{index:02d}-{suffix}.prom").write_text(
                    raw,
                    encoding="utf-8",
                )
            except Exception as exc:
                self.writer.write(
                    {
                        "timestamp": time.time(),
                        "kind": "metrics_error",
                        "url": url,
                        "error": repr(exc),
                    }
                )
        return snapshots

    def snapshot_health(self, phase: str) -> float | None:
        """Persist a health snapshot and return its timestamp for phase filtering."""
        if not self.health_url:
            return None
        timestamp = time.time()
        try:
            self.writer.write(
                {
                    "timestamp": timestamp,
                    "kind": "proxy_health",
                    "phase": phase,
                    "url": self.health_url,
                    "value": self._get_health(),
                }
            )
        except Exception as exc:
            self.writer.write(
                {
                    "timestamp": timestamp,
                    "kind": "health_error",
                    "phase": phase,
                    "url": self.health_url,
                    "error": repr(exc),
                }
            )
        return timestamp

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            timestamp = time.time()
            if self.health_url:
                try:
                    self.writer.write(
                        {
                            "timestamp": timestamp,
                            "kind": "proxy_health",
                            "url": self.health_url,
                            "value": self._get_health(),
                        }
                    )
                except Exception as exc:
                    self.writer.write(
                        {
                            "timestamp": timestamp,
                            "kind": "health_error",
                            "url": self.health_url,
                            "error": repr(exc),
                        }
                    )
            for url in self.metrics_urls:
                try:
                    metric_url, parsed, _ = self._get_metric(url)
                    gauges = {
                        name: value
                        for name, value in parsed.items()
                        if name
                        in (
                            "vllm:kv_cache_usage_perc",
                            "vllm:num_requests_running",
                            "vllm:num_requests_waiting",
                        )
                    }
                    self.writer.write(
                        {
                            "timestamp": timestamp,
                            "kind": "vllm_gauges",
                            "url": metric_url,
                            "value": gauges,
                        }
                    )
                except Exception as exc:
                    self.writer.write(
                        {
                            "timestamp": timestamp,
                            "kind": "metrics_error",
                            "url": url,
                            "error": repr(exc),
                        }
                    )

    def start(self, raw_dir: Path) -> None:
        self.before = self.snapshot_metrics(raw_dir, "before")
        if self.health_url:
            try:
                self.writer.write(
                    {
                        "timestamp": time.time(),
                        "kind": "proxy_health",
                        "url": self.health_url,
                        "value": self._get_health(),
                    }
                )
            except Exception as exc:
                self.writer.write(
                    {
                        "timestamp": time.time(),
                        "kind": "health_error",
                        "url": self.health_url,
                        "error": repr(exc),
                    }
                )
        if self.health_url or self.metrics_urls:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self, raw_dir: Path) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(self.interval * 2, 5.0))
        if self.health_url:
            try:
                self.writer.write(
                    {
                        "timestamp": time.time(),
                        "kind": "proxy_health",
                        "url": self.health_url,
                        "value": self._get_health(),
                    }
                )
            except Exception as exc:
                self.writer.write(
                    {
                        "timestamp": time.time(),
                        "kind": "health_error",
                        "url": self.health_url,
                        "error": repr(exc),
                    }
                )
        self.after = self.snapshot_metrics(raw_dir, "after")


def make_http_session(concurrency: int) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=max(concurrency, 16),
        pool_maxsize=max(concurrency, 16),
        max_retries=0,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def read_prefix(args: argparse.Namespace) -> str:
    if args.prefix_file:
        return Path(args.prefix_file).read_text(encoding="utf-8")
    words = [f"cacheword{i % 97:02d}" for i in range(args.prefix_words)]
    return " ".join(words)


def build_initial_messages(
    scenario: str,
    session_index: int,
    common_prefix: str,
) -> list[dict[str, str]]:
    marker = f"session-marker-{session_index:08d}"
    if scenario in ("shared-prefix", "hot-key", "hot-prefix"):
        system = f"{common_prefix}\nFollow the instructions carefully."
        first_user = f"{marker}\nSummarize the stable document in one short sentence."
    elif scenario == "short":
        system = f"{marker}\nYou are a concise assistant."
        first_user = "Reply with one short sentence."
    elif scenario == "one-shot":
        system = f"{marker}\n{common_prefix}"
        first_user = "Give one short observation about this document."
    else:
        # Put the marker before the long text so different sessions do not
        # accidentally share the benchmarked prefix with each other.
        system = f"{marker}\n{common_prefix}"
        first_user = "Summarize the stable document in one short sentence."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": first_user},
    ]


def extract_stream_text(delta: dict[str, Any]) -> str:
    parts = [
        delta.get("reasoning"),
        delta.get("reasoning_content"),
        delta.get("reasoning_text"),
        delta.get("content"),
    ]
    return "".join(part for part in parts if isinstance(part, str))


def send_request(
    *,
    http: requests.Session,
    api_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    session_id: str,
    session_index: int,
    turn: int,
    scenario: str,
    phase: str,
    stage: str,
    session_header: str,
    send_session_key: bool,
    max_tokens: int,
    temperature: float,
    timeout: float,
    verify: bool,
) -> tuple[RequestResult, str]:
    headers = {
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if send_session_key:
        headers[session_header] = session_id

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    started_at_s = time.time()
    started = time.perf_counter()
    ttfb_ms: float | None = None
    ttft_ms: float | None = None
    status_code: int | None = None
    usage: dict[str, Any] = {}
    output_parts: list[str] = []
    error: str | None = None
    try:
        with http.post(
            api_url,
            json=payload,
            headers=headers,
            stream=True,
            timeout=(30.0, timeout),
            verify=verify,
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not raw_line:
                    continue
                now = time.perf_counter()
                if ttfb_ms is None:
                    ttfb_ms = (now - started) * 1000
                line = str(raw_line)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                text = extract_stream_text(delta)
                if text:
                    if ttft_ms is None:
                        ttft_ms = (now - started) * 1000
                    output_parts.append(text)
    except Exception as exc:
        error = repr(exc)

    ended = time.perf_counter()
    ended_at_s = time.time()
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    output = "".join(output_parts)
    result = RequestResult(
        session_index=session_index,
        session_id=session_id,
        turn=turn,
        scenario=scenario,
        phase=phase,
        stage=stage,
        status_code=status_code,
        ok=error is None and status_code is not None and status_code < 400,
        started_at_s=started_at_s,
        ended_at_s=ended_at_s,
        ttfb_ms=ttfb_ms,
        ttft_ms=ttft_ms,
        e2e_ms=(ended - started) * 1000,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        client_cached_tokens=details.get("cached_tokens"),
        output_chars=len(output),
        error=error,
    )
    return result, output


def derive_urls(base_url: str) -> tuple[str, str]:
    normalized = base_url.rstrip("/") + "/"
    if normalized.endswith("/v1/"):
        root = normalized[: -len("v1/")]
        api_url = urljoin(normalized, "chat/completions")
    else:
        root = normalized
        api_url = urljoin(normalized, "v1/chat/completions")
    return api_url, urljoin(root, "healthcheck")


def reset_cache(url: str, api_key: str | None, timeout: float, verify: bool) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.post(url, headers=headers, timeout=timeout, verify=verify)
    response.raise_for_status()
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:2048]
    return {"url": url, "status_code": response.status_code, "body": body}


def validate_reset_fanout(reset_result: dict[str, Any], base_urls: list[str]) -> None:
    body = reset_result.get("body")
    if not isinstance(body, dict) or body.get("success") is not True:
        raise RuntimeError(f"proxy reset did not return a successful structured result: {body!r}")
    backends = body.get("backends")
    if not isinstance(backends, list) or len(backends) != len(base_urls):
        raise RuntimeError(
            f"proxy reset reported {len(backends) if isinstance(backends, list) else 0} backends; "
            f"expected {len(base_urls)}: {body!r}"
        )
    expected = {base_url.rstrip("/") for base_url in base_urls}
    reported = {
        str(backend.get("target", "")).rstrip("/")
        for backend in backends
        if isinstance(backend, dict) and not backend.get("error")
    }
    if reported != expected:
        raise RuntimeError(f"proxy reset targets differ: expected {sorted(expected)}, got {sorted(reported)}")


def metrics_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "metrics")


def chat_completions_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "v1/chat/completions")


def wait_for_prefills_idle(
    base_urls: list[str],
    timeout: float,
    verify: bool,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, dict[str, float]] = {}
    while True:
        idle = True
        current: dict[str, dict[str, float]] = {}
        for base_url in base_urls:
            response = requests.get(metrics_url(base_url), timeout=min(timeout, 30.0), verify=verify)
            response.raise_for_status()
            parsed = parse_prometheus(response.text)
            missing = {
                name for name in ("vllm:num_requests_running", "vllm:num_requests_waiting") if name not in parsed
            }
            if missing:
                raise RuntimeError(f"Prefill metrics from {base_url} are missing {sorted(missing)}")
            running = parsed.get("vllm:num_requests_running", 0.0)
            waiting = parsed.get("vllm:num_requests_waiting", 0.0)
            current[base_url] = {"running": running, "waiting": waiting}
            idle = idle and running <= 0 and waiting <= 0
        last = current
        if idle:
            return {"wait_seconds": time.monotonic() - started, "backends": last}
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f"Prefill drain timed out after {timeout}s: {last}")
        time.sleep(poll_interval)


def send_reset_probe(
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    verify: bool,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.monotonic()
    response = requests.post(
        chat_completions_url(base_url),
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=(30.0, timeout),
        verify=verify,
    )
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    cached_tokens = details.get("cached_tokens")
    return {
        "base_url": base_url,
        "status_code": response.status_code,
        "elapsed_seconds": time.monotonic() - started,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tokens if isinstance(cached_tokens, int) else None,
    }


def verify_reset_across_prefills(
    *,
    base_urls: list[str],
    reset_url: str,
    model: str,
    label: str,
    api_key: str | None,
    timeout: float,
    drain_timeout: float,
    verify: bool,
    output_dir: Path,
) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "verified": False,
        "prefill_count": len(base_urls),
        "prefill_base_urls": base_urls,
        "drains": [],
        "probes": [],
        "resets": [],
    }
    probe_path = output_dir / "reset-probe-requests.jsonl"
    probe_writer = JsonlWriter(probe_path)
    probe_nonce = uuid.uuid4().hex

    def record_probe(phase: str, index: int, base_url: str, prompt: str) -> dict[str, Any]:
        result = send_reset_probe(base_url, model, prompt, api_key, timeout, verify)
        result.update({"phase": phase, "prefill_index": index})
        validation["probes"].append(result)
        probe_writer.write(result)
        return result

    try:
        validation["drains"].append(
            {"phase": "before-prime", **wait_for_prefills_idle(base_urls, drain_timeout, verify)}
        )
        prompts = [
            (
                f"reset-verification-{probe_nonce}-{label}-{index}\n"
                + " ".join(f"resetprobe{index:02d}word{word:04d}" for word in range(384))
            )
            for index in range(len(base_urls))
        ]
        for index, (base_url, prompt) in enumerate(zip(base_urls, prompts, strict=True)):
            first = record_probe("prime-miss", index, base_url, prompt)
            if not isinstance(first.get("prompt_tokens"), int) or first["prompt_tokens"] <= 128:
                raise RuntimeError(f"reset probe for {base_url} did not exceed one 128-token cache block: {first}")
            second = record_probe("prime-hit", index, base_url, prompt)
            if not isinstance(second.get("cached_tokens"), int) or second["cached_tokens"] <= 0:
                raise RuntimeError(f"reset probe did not establish a cache hit on {base_url}: {second}")

        validation["drains"].append(
            {"phase": "before-verified-reset", **wait_for_prefills_idle(base_urls, drain_timeout, verify)}
        )
        verified_reset = reset_cache(reset_url, api_key, timeout, verify)
        validation["resets"].append({"phase": "verified-reset", **verified_reset})
        validate_reset_fanout(verified_reset, base_urls)
        validation["drains"].append(
            {"phase": "after-verified-reset", **wait_for_prefills_idle(base_urls, drain_timeout, verify)}
        )

        for index, (base_url, prompt) in enumerate(zip(base_urls, prompts, strict=True)):
            cold = record_probe("post-reset-miss", index, base_url, prompt)
            if cold.get("cached_tokens") not in (None, 0):
                raise RuntimeError(f"reset verification still hit cached tokens on {base_url}: {cold}")

        validation["drains"].append(
            {"phase": "before-final-reset", **wait_for_prefills_idle(base_urls, drain_timeout, verify)}
        )
        final_reset = reset_cache(reset_url, api_key, timeout, verify)
        validation["resets"].append({"phase": "final-reset", **final_reset})
        validate_reset_fanout(final_reset, base_urls)
        validation["drains"].append(
            {"phase": "after-final-reset", **wait_for_prefills_idle(base_urls, drain_timeout, verify)}
        )
        validation["verified"] = True
        return validation
    except Exception as exc:
        validation["error"] = repr(exc)
        raise
    finally:
        probe_writer.close()
        (output_dir / "reset-validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def summarize_health_samples(
    samples_path: Path,
    start_timestamp: float | None = None,
) -> dict[str, Any]:
    spreads: list[float] = []
    cvs: list[float] = []
    max_priorities: list[float] = []
    first_stats_by_field: dict[str, dict[str, int | float]] = {}
    last_stats_by_field: dict[str, dict[str, int | float]] = {}
    first_source_stats: dict[str, dict[str, int | float]] = {}
    last_source_stats: dict[str, dict[str, int | float]] = {}
    affinity_config: dict[str, int | float] = {}

    def numeric_stats(value: Any) -> dict[str, int | float]:
        if not isinstance(value, dict):
            return {}
        return {name: metric for name, metric in value.items() if isinstance(metric, (int, float))}

    def stats_delta(
        first: dict[str, int | float],
        last: dict[str, int | float],
    ) -> dict[str, int | float]:
        return {name: last.get(name, 0) - first.get(name, 0) for name in sorted(set(first) | set(last))}

    with samples_path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample.get("kind") != "proxy_health":
                continue
            if start_timestamp is not None and float(sample.get("timestamp", 0)) < start_timestamp:
                continue
            health = sample.get("value") or {}
            for field in ("session_affinity_stats", "prefix_affinity_stats"):
                current = numeric_stats(health.get(field))
                if current:
                    first_stats_by_field.setdefault(field, current)
                    last_stats_by_field[field] = current
            by_source = health.get("prefill_cache_stats_by_source")
            if isinstance(by_source, dict):
                for source, value in by_source.items():
                    current = numeric_stats(value)
                    if current:
                        first_source_stats.setdefault(source, current)
                        last_source_stats[source] = current
            for field in (
                "prefix_hash_chars",
                "prefix_cache_capacity",
                "prefix_spill_max_nodes",
                "prefill_kv_weight",
            ):
                value = health.get(field)
                if isinstance(value, (int, float)):
                    affinity_config[field] = value
            loads = health.get("prefill_loads") or {}
            priorities = [
                float(value["priority"])
                for value in loads.values()
                if isinstance(value, dict) and isinstance(value.get("priority"), (int, float))
            ]
            if not priorities:
                continue
            spreads.append(max(priorities) - min(priorities))
            max_priorities.append(max(priorities))
            mean = statistics.fmean(priorities)
            cvs.append(statistics.pstdev(priorities) / mean if mean > 0 else 0.0)

    session_delta = stats_delta(
        first_stats_by_field.get("session_affinity_stats", {}),
        last_stats_by_field.get("session_affinity_stats", {}),
    )
    if session_delta:
        lookups = session_delta.get("lookups", 0)
        if lookups:
            session_delta["derived_affinity_hit_rate"] = session_delta.get("hits", 0) / lookups
            session_delta["derived_overload_fallback_rate"] = session_delta.get("overload_fallbacks", 0) / lookups
        prompt_tokens = session_delta.get("prompt_tokens", 0)
        if prompt_tokens:
            session_delta["derived_cached_token_rate"] = session_delta.get("cached_tokens", 0) / prompt_tokens

    prefix_delta = stats_delta(
        first_stats_by_field.get("prefix_affinity_stats", {}),
        last_stats_by_field.get("prefix_affinity_stats", {}),
    )
    if prefix_delta:
        lookups = prefix_delta.get("lookups", 0)
        if lookups:
            prefix_delta["derived_prefix_hit_rate"] = prefix_delta.get("hits", 0) / lookups
            prefix_delta["derived_spillover_rate"] = prefix_delta.get("spillover_routes", 0) / lookups

    source_deltas = {
        source: stats_delta(first_source_stats.get(source, {}), last) for source, last in last_source_stats.items()
    }
    for delta in source_deltas.values():
        prompt_tokens = delta.get("prompt_tokens", 0)
        if prompt_tokens:
            delta["derived_cached_token_rate"] = delta.get("cached_tokens", 0) / prompt_tokens

    return {
        "samples": len(spreads),
        "priority_spread_p50": percentile(spreads, 0.50),
        "priority_spread_p95": percentile(spreads, 0.95),
        "priority_cv_p50": percentile(cvs, 0.50),
        "priority_cv_p95": percentile(cvs, 0.95),
        "max_priority_p95": percentile(max_priorities, 0.95),
        "affinity_config": affinity_config,
        "session_affinity_stats_delta": session_delta,
        "prefix_affinity_stats_delta": prefix_delta,
        "prefill_cache_stats_by_source_delta": source_deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Proxy root URL or its /v1 URL.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--scenario",
        choices=(
            "session-long",
            "short",
            "one-shot",
            "shared-prefix",
            "hot-key",
            "hot-prefix",
            "churn",
        ),
        default="session-long",
    )
    parser.add_argument("--sessions", type=int, default=256)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--prefix-words", type=int, default=1024)
    parser.add_argument("--prefix-file", help="Use a real stable prefix instead of generated words.")
    parser.add_argument(
        "--workload-file",
        type=Path,
        help="Immutable JSONL workload generated once and shared by all A/B/C groups.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-time", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--session-header", default="x-session-id")
    parser.add_argument("--no-session-key", action="store_true")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key; never written to results.",
    )
    parser.add_argument("--health-url", help="Proxy /healthcheck; derived from base URL by default.")
    parser.add_argument("--no-health-sampling", action="store_true")
    parser.add_argument("--prefill-metrics-url", action="append", default=[])
    parser.add_argument("--decode-metrics-url", action="append", default=[])
    parser.add_argument(
        "--prefill-base-url",
        action="append",
        default=[],
        help="Direct Prefill root used to drain and verify reset; repeat once per Prefill.",
    )
    parser.add_argument(
        "--metrics-url",
        action="append",
        default=[],
        help="Additional unclassified vLLM metrics endpoint.",
    )
    parser.add_argument("--reset-before", action="store_true")
    parser.add_argument("--reset-url", help="Proxy reset endpoint; derived by default.")
    parser.add_argument(
        "--verify-reset",
        action="store_true",
        help="Prime, reset, and probe every --prefill-base-url before measurement.",
    )
    parser.add_argument("--expected-prefill-count", type=int, default=0)
    parser.add_argument("--reset-drain-timeout", type=float, default=120.0)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--system-warmup-requests",
        type=int,
        default=0,
        help="Sacrificial requests sent before cache reset and metric snapshots.",
    )
    parser.add_argument("--label", default="session-affinity")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--compare-with",
        help="Existing baseline summary.json; writes comparison.json for this run.",
    )
    args = parser.parse_args()
    for name in ("prefill_base_url", "prefill_metrics_url", "decode_metrics_url", "metrics_url"):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    if args.sessions <= 0 or args.turns <= 0 or args.concurrency <= 0:
        parser.error("sessions, turns, and concurrency must be positive")
    if args.system_warmup_requests < 0:
        parser.error("system-warmup-requests must be non-negative")
    if args.expected_prefill_count < 0:
        parser.error("expected-prefill-count must be non-negative")
    if args.reset_drain_timeout <= 0:
        parser.error("reset-drain-timeout must be positive")
    if args.verify_reset and not args.reset_before:
        parser.error("--verify-reset requires --reset-before")
    if args.verify_reset and not args.prefill_base_url:
        parser.error("--verify-reset requires at least one --prefill-base-url")
    if args.expected_prefill_count and len(args.prefill_base_url) != args.expected_prefill_count:
        parser.error(f"expected {args.expected_prefill_count} Prefill URLs, got {len(args.prefill_base_url)}")
    if args.scenario == "one-shot":
        args.turns = 1
    return args


def nested_number(value: dict[str, Any], path: str) -> float | None:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if isinstance(current, (int, float)) else None


def compare_summaries(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    paths = [
        "warm_turns.ttft_ms.p50",
        "warm_turns.ttft_ms.p95",
        "warm_turns.ttft_ms.p99",
        "warm_turns.e2e_ms.p95",
        "warm_turns.cached_token_request_rate",
        "warm_turns.cached_token_ratio",
        "warm_turns.client_computed_tokens.mean",
        "warm_turns.request_throughput_per_second",
        "warm_turns.output_token_throughput_per_second",
        "overall.success_rate",
        "measurement_metrics_delta_by_role.prefill.derived:prefix_cache_hit_rate",
        "measurement_metrics_delta_by_role.prefill.derived:mean_prefill_computed_tokens",
        "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_prefill_time_seconds",
        "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_queue_time_seconds",
        "measurement_metrics_delta_by_role.decode.derived:mean:vllm:request_queue_time_seconds",
        "proxy_prefill_load_balance_measurement.priority_cv_p95",
        ("proxy_prefill_load_balance_measurement.session_affinity_stats_delta." "derived_overload_fallback_rate"),
        ("proxy_prefill_load_balance_measurement.prefix_affinity_stats_delta." "derived_prefix_hit_rate"),
        ("proxy_prefill_load_balance_measurement.prefix_affinity_stats_delta." "derived_spillover_rate"),
    ]
    baseline_stages = set((baseline.get("per_stage") or {}).keys())
    treatment_stages = set((treatment.get("per_stage") or {}).keys())
    for stage in sorted(baseline_stages & treatment_stages):
        paths.extend(
            [
                f"per_stage.{stage}.ttft_ms.p95",
                f"per_stage.{stage}.e2e_ms.p95",
                f"per_stage.{stage}.cached_token_request_rate",
                f"per_stage.{stage}.cached_token_ratio",
                f"per_stage.{stage}.client_computed_tokens.mean",
                f"per_stage.{stage}.request_throughput_per_second",
            ]
        )
    comparison: dict[str, Any] = {}
    lower_is_better = (
        "ttft",
        "e2e",
        "computed_tokens",
        "prefill_time",
        "queue_time",
        "priority_cv",
        "fallback_rate",
        "spillover_rate",
    )
    for path in paths:
        before = nested_number(baseline, path)
        after = nested_number(treatment, path)
        if before is None or after is None:
            continue
        if before == 0:
            relative_change = None
            improvement = None
        else:
            relative_change = (after - before) / before
            improvement = (
                (before - after) / before
                if any(name in path for name in lower_is_better)
                else (after - before) / before
            )
        comparison[path] = {
            "baseline": before,
            "treatment": after,
            "relative_change": relative_change,
            "improvement": improvement,
        }
    return comparison


def workload_stages(records: list[PlannedRequest]) -> list[list[PlannedRequest]]:
    """Return contiguous phase/stage batches without silently merging repeats."""
    batches: list[list[PlannedRequest]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.phase, record.stage)
        if not batches or (batches[-1][0].phase, batches[-1][0].stage) != key:
            if key in seen:
                raise ValueError(f"workload phase/stage is not contiguous: {key}")
            batches.append([])
            seen.add(key)
        batches[-1].append(record)
    return batches


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    planned_records = load_workload_jsonl(args.workload_file) if args.workload_file else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    api_url, derived_health_url = derive_urls(args.base_url)
    health_url = None if args.no_health_sampling else (args.health_url or derived_health_url)
    root_url = args.base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3].rstrip("/")
    reset_url = args.reset_url or f"{root_url}/reset_prefix_cache"
    api_key = os.environ.get(args.api_key_env)
    verify = not args.insecure

    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config["api_url"] = api_url
    config["health_url"] = health_url
    config["reset_url"] = reset_url
    config["api_key_present"] = bool(api_key)
    if args.workload_file:
        config["workload_sha256"] = hashlib.sha256(args.workload_file.read_bytes()).hexdigest()
        config["workload_requests"] = len(planned_records or [])
        config["workload_scenarios"] = sorted({record.scenario for record in planned_records or []})
        config["workload_stages"] = list(dict.fromkeys(record.stage for record in planned_records or []))
        config["workload_max_tokens"] = sorted({record.max_tokens for record in planned_records or []})
    config.pop("api_key_env", None)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"[benchmark] label={args.label} requests={len(planned_records or []) or 'generated'} " f"output={output_dir}",
        file=sys.stderr,
        flush=True,
    )
    if args.system_warmup_requests:
        print(
            f"[benchmark] system warmup: {args.system_warmup_requests} requests",
            file=sys.stderr,
            flush=True,
        )
        warmup_writer = JsonlWriter(output_dir / "system-warmup.jsonl")
        warmup_http = make_http_session(1)
        try:
            for index in range(args.system_warmup_requests):
                result, _ = send_request(
                    http=warmup_http,
                    api_url=api_url,
                    api_key=api_key,
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"System warm-up request {args.seed}-{index}; reply briefly.",
                        }
                    ],
                    session_id=f"system-warmup-{args.seed}-{index}",
                    session_index=-1,
                    turn=-1,
                    scenario="system-warmup",
                    phase="system-warmup",
                    stage="system-warmup",
                    session_header=args.session_header,
                    send_session_key=False,
                    max_tokens=1,
                    temperature=0.0,
                    timeout=args.timeout,
                    verify=verify,
                )
                warmup_writer.write(asdict(result))
                if not result.ok:
                    raise RuntimeError(f"system warm-up failed: {result.error}")
        finally:
            warmup_http.close()
            warmup_writer.close()

    # Infrastructure warm-up deliberately happens before this reset. The
    # measured cache-fill phase therefore starts with empty backend KV and,
    # for the candidate, empty routing-affinity LRUs.
    reset_validation: dict[str, Any] | None = None
    if args.reset_before and args.verify_reset:
        print(
            f"[benchmark] verifying reset across {len(args.prefill_base_url)} Prefill backends",
            file=sys.stderr,
            flush=True,
        )
        reset_validation = verify_reset_across_prefills(
            base_urls=args.prefill_base_url,
            reset_url=reset_url,
            model=args.model,
            label=args.label,
            api_key=api_key,
            timeout=args.timeout,
            drain_timeout=args.reset_drain_timeout,
            verify=verify,
            output_dir=output_dir,
        )
    elif args.reset_before:
        print("[benchmark] resetting backend prefix/affinity caches", file=sys.stderr, flush=True)
        reset_result = reset_cache(reset_url, api_key, args.timeout, verify)
        reset_validation = {"verified": False, "resets": [{"phase": "unverified-reset", **reset_result}]}
        (output_dir / "reset-validation.json").write_text(
            json.dumps(reset_validation, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    requests_writer = JsonlWriter(output_dir / "requests.jsonl")
    samples_writer = JsonlWriter(samples_path)
    all_metrics_urls = list(dict.fromkeys(args.prefill_metrics_url + args.decode_metrics_url + args.metrics_url))
    sampler = EndpointSampler(
        health_url=health_url,
        metrics_urls=all_metrics_urls,
        interval=args.sample_interval,
        timeout=min(args.timeout, 30.0),
        verify=verify,
        writer=samples_writer,
    )
    sampler.start(output_dir)
    print("[benchmark] metrics sampling started; running workload", file=sys.stderr, flush=True)

    common_prefix = read_prefix(args) if planned_records is None else ""
    histories = (
        [build_initial_messages(args.scenario, index, common_prefix) for index in range(args.sessions)]
        if planned_records is None
        else []
    )
    session_ids = [f"bench-{args.seed}-{index:08d}" for index in range(args.sessions)]
    if args.scenario == "hot-key" and planned_records is None:
        session_ids = [f"bench-hot-key-{args.seed}"] * args.sessions

    all_results: list[RequestResult] = []
    thread_local = threading.local()
    measurement_metrics_before: dict[str, dict[str, float]] | None = None
    measurement_health_started_at: float | None = None

    def warm_hot_prefix() -> None:
        warmup_http = make_http_session(1)
        try:
            warmup_result, _ = send_request(
                http=warmup_http,
                api_url=api_url,
                api_key=api_key,
                model=args.model,
                messages=build_initial_messages("hot-prefix", -1, common_prefix),
                session_id=f"bench-hot-prefix-warmup-{args.seed}",
                session_index=-1,
                turn=-1,
                scenario="hot-prefix",
                phase="cache-fill",
                stage="hot-prefix-prime",
                session_header=args.session_header,
                send_session_key=not args.no_session_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                verify=verify,
            )
        finally:
            warmup_http.close()
        if not warmup_result.ok:
            raise RuntimeError(f"hot-prefix warm-up failed: {warmup_result.error}")

    def invoke(session_index: int, turn: int) -> tuple[int, RequestResult, str]:
        if not hasattr(thread_local, "http"):
            thread_local.http = make_http_session(args.concurrency)
        session_id = session_ids[session_index]
        if args.scenario == "hot-prefix":
            session_id = f"{session_id}-turn-{turn}"
        result, output = send_request(
            http=thread_local.http,
            api_url=api_url,
            api_key=api_key,
            model=args.model,
            messages=histories[session_index],
            session_id=session_id,
            session_index=session_index,
            turn=turn,
            scenario=args.scenario,
            phase="cache-fill" if turn == 0 else "measure",
            stage=f"turn-{turn}",
            session_header=args.session_header,
            send_session_key=not args.no_session_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            verify=verify,
        )
        return session_index, result, output

    def invoke_planned(record: PlannedRequest) -> tuple[int, RequestResult, str]:
        if not hasattr(thread_local, "http"):
            thread_local.http = make_http_session(args.concurrency)
        result, output = send_request(
            http=thread_local.http,
            api_url=api_url,
            api_key=api_key,
            model=args.model,
            messages=record.messages,
            session_id=record.session_id,
            session_index=record.session_index,
            turn=record.turn,
            scenario=record.scenario,
            phase=record.phase,
            stage=record.stage,
            session_header=args.session_header,
            send_session_key=record.send_session_key,
            max_tokens=record.max_tokens,
            temperature=record.temperature,
            timeout=args.timeout,
            verify=verify,
        )
        return record.session_index, result, output

    def record_completed(future: concurrent.futures.Future[tuple[int, RequestResult, str]]) -> None:
        _, result, _ = future.result()
        all_results.append(result)
        requests_writer.write(asdict(result))

    def print_stage(stage: str, phase: str) -> None:
        current = latency_summary([result for result in all_results if result.stage == stage])
        print(
            json.dumps(
                {
                    "phase": phase,
                    "stage": stage,
                    "requests": current["requests"],
                    "success_rate": current["success_rate"],
                    "ttft_p50_ms": current["ttft_ms"]["p50"],
                    "ttft_p95_ms": current["ttft_ms"]["p95"],
                    "e2e_p95_ms": current["e2e_ms"]["p95"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    started = time.time()
    try:
        if args.scenario == "hot-prefix" and planned_records is None:
            warm_hot_prefix()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            if planned_records is not None:
                batches = workload_stages(planned_records)
                for batch_index, batch in enumerate(batches):
                    offsets = [record.scheduled_offset_s for record in batch]
                    if offsets != sorted(offsets):
                        raise ValueError(f"scheduled offsets are not sorted in stage {batch[0].stage}")
                    stage_started = time.perf_counter()
                    pending: set[concurrent.futures.Future[tuple[int, RequestResult, str]]] = set()
                    for record in batch:
                        delay = stage_started + record.scheduled_offset_s - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        while len(pending) >= args.concurrency:
                            done, pending = concurrent.futures.wait(
                                pending,
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            for future in done:
                                record_completed(future)
                        pending.add(executor.submit(invoke_planned, record))
                    for future in concurrent.futures.as_completed(pending):
                        record_completed(future)
                    print_stage(batch[0].stage, batch[0].phase)
                    next_phase = batches[batch_index + 1][0].phase if batch_index + 1 < len(batches) else None
                    if batch[0].phase == "cache-fill" and next_phase != "cache-fill":
                        measurement_metrics_before = sampler.snapshot_metrics(
                            output_dir,
                            "after-cache-fill",
                        )
                        measurement_health_started_at = sampler.snapshot_health("measure-start")
            else:
                for turn in range(args.turns):
                    futures = [executor.submit(invoke, index, turn) for index in range(args.sessions)]
                    for future in concurrent.futures.as_completed(futures):
                        session_index, result, output = future.result()
                        all_results.append(result)
                        requests_writer.write(asdict(result))
                        if result.ok:
                            histories[session_index].append({"role": "assistant", "content": output or "Acknowledged."})
                            histories[session_index].append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"Turn {turn + 2}: refine the previous answer, "
                                        "keeping it to one short sentence."
                                    ),
                                }
                            )
                    print_stage(f"turn-{turn}", "cache-fill" if turn == 0 else "measure")
                    if turn == 0:
                        measurement_metrics_before = sampler.snapshot_metrics(
                            output_dir,
                            "after-cache-fill",
                        )
                        measurement_health_started_at = sampler.snapshot_health("measure-start")
                    if args.think_time > 0 and turn + 1 < args.turns:
                        time.sleep(args.think_time)
    finally:
        sampler.stop(output_dir)
        requests_writer.close()
        samples_writer.close()

    metric_deltas = {
        url: counter_delta(sampler.before.get(url, {}), sampler.after.get(url, {}))
        for url in sorted(set(sampler.before) | set(sampler.after))
    }
    metric_deltas_by_role = {}
    measurement_metric_deltas_by_role = {}
    for role, urls in (
        ("prefill", args.prefill_metrics_url),
        ("decode", args.decode_metrics_url),
        ("unclassified", args.metrics_url),
    ):
        if urls:
            metric_deltas_by_role[role] = counter_delta(
                aggregate_metric_snapshots(urls, sampler.before),
                aggregate_metric_snapshots(urls, sampler.after),
            )
            if measurement_metrics_before is not None:
                measurement_metric_deltas_by_role[role] = counter_delta(
                    aggregate_metric_snapshots(urls, measurement_metrics_before),
                    aggregate_metric_snapshots(urls, sampler.after),
                )
    scenario = planned_records[0].scenario if planned_records else args.scenario
    turns = sorted({result.turn for result in all_results if result.turn >= 0})
    stages = list(dict.fromkeys(result.stage for result in all_results))
    summary = {
        "label": args.label,
        "scenario": scenario,
        "wall_time_seconds": time.time() - started,
        "overall": latency_summary(all_results),
        "cache_fill": latency_summary([result for result in all_results if result.phase == "cache-fill"]),
        "measured": latency_summary([result for result in all_results if result.phase == "measure"]),
        "cold_turn": latency_summary([result for result in all_results if result.phase == "cache-fill"]),
        "warm_turns": latency_summary([result for result in all_results if result.phase == "measure"]),
        "per_turn": {
            str(turn): latency_summary([result for result in all_results if result.turn == turn]) for turn in turns
        },
        "per_stage": {
            stage: latency_summary([result for result in all_results if result.stage == stage]) for stage in stages
        },
        "proxy_prefill_load_balance": summarize_health_samples(samples_path),
        "proxy_prefill_load_balance_measurement": summarize_health_samples(
            samples_path,
            measurement_health_started_at,
        ),
        "metrics_delta_by_role": metric_deltas_by_role,
        "measurement_metrics_delta_by_role": measurement_metric_deltas_by_role,
        "metrics_delta_by_url": metric_deltas,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    required_checks = {
        "requests_succeeded": summary["overall"]["errors"] == 0,
        "reset_verified": bool(reset_validation and reset_validation.get("verified")),
        "prefill_count_matches": (
            not args.expected_prefill_count or len(args.prefill_base_url) == args.expected_prefill_count
        ),
        "prefill_metrics_complete": (
            not args.prefill_metrics_url
            or all(url in sampler.before and url in sampler.after for url in args.prefill_metrics_url)
        ),
        "decode_metrics_complete": (
            not args.decode_metrics_url
            or all(url in sampler.before and url in sampler.after for url in args.decode_metrics_url)
        ),
    }
    if scenario == "session-long" and args.reset_before:
        required_checks["cold_turn_has_no_cache_hits"] = summary["cold_turn"]["cached_token_request_rate"] == 0
    if scenario == "shared-prefix" and args.reset_before:
        required_checks["shared_prefix_prime_has_no_cache_hits"] = (
            summary["cache_fill"]["cached_token_request_rate"] == 0
        )
        required_checks["shared_prefix_probe_present"] = (
            summary["per_stage"].get("prefix-probe", {}).get("requests", 0) > 0
        )
    validity = {
        "valid": all(required_checks.values()),
        "checks": required_checks,
        "reset_validation_file": "reset-validation.json" if reset_validation is not None else None,
    }
    (output_dir / "validity.json").write_text(
        json.dumps(validity, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.compare_with:
        baseline = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
        comparison = compare_summaries(baseline, summary)
        (output_dir / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["overall"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
