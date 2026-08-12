# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 --workers 2 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - Without affinity, the proxy selects the lowest-priority live instance from
#   its load-tracking heap.
# - For production, ensure your backend servers are robust and secure.
# - Pass --enable-kv-cache-aware-routing to route Prefill requests by stable
#   session IDs, then by a prefix hash when the session has no existing binding.
#   Without the flag, requests keep the normal load-based routing behavior.
# - Multimodal and tool requests skip prefix affinity because this proxy cannot
#   reproduce their model-specific token prefix safely.
# - Affinity applies only to Prefillers. Decoders remain load-balanced.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import base64
import functools
import hashlib
import heapq
import ipaddress
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

try:
    import uvloop  # type: ignore[import-not-found]

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_key: str
    prefiller_score: float
    decoder_key: str
    decoder_score: float
    decoder_host: str
    decoder_port: int
    prefiller_cached_tokens: int | None = None


TAINT_PRIORITY = 1e15
SESSION_KEY_VERSION = "pd-session-v2"
MAX_AFFINITY_ID_BYTES = 256
# Priority is deliberate: a trusted gateway may explicitly override affinity;
# a Codex thread is more conversation-specific than its client session; native
# tool identifiers then win over generic compatibility aliases.
SESSION_HEADER_NAMES: tuple[str, ...] = (
    "x-session-affinity",
    "thread-id",
    "x-opencode-session",
    "x-task-id",
    "x-roo-task-id",
    "x-session-id",
    "session-id",
    "session_id",
)
PREFIX_HASH_VERSION = "pd-prefix-v1"
# Absolute slack so near-zero priorities do not false-trigger overload escape.
AFFINITY_OVERLOAD_MARGIN = 1.0
# A cache-discounted reservation never drops below this fraction of the full
# score: even a 100%-cached prompt still costs scheduling, attention over the
# cached prefix and KV-transfer setup.
AFFINITY_DISCOUNT_MIN_FRACTION = 0.1


def extract_cached_tokens(response_json: dict) -> int | None:
    usage = response_json.get("usage") or {}
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_tokens_details.get("cached_tokens")
    return cached_tokens if isinstance(cached_tokens, int) else None


def extract_reusable_prefix_tokens(response_json: dict) -> int | None:
    """Sum cached and created cache tokens from the Prefill response.

    Returns None when the details are missing or incomplete. The caller decides
    how to react (warn + bind, or skip binding).
    """
    usage = response_json.get("usage") or {}
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_tokens_details, dict):
        return None
    cached_tokens = prompt_tokens_details.get("cached_tokens")
    created_cache_tokens = prompt_tokens_details.get("created_cache_tokens")
    if not isinstance(cached_tokens, int) or not isinstance(created_cache_tokens, int):
        return None
    return cached_tokens + created_cache_tokens


def update_cached_tokens_in_chunk(chunk_json: dict, cached_tokens: int | None) -> bool:
    if cached_tokens is None:
        return False
    usage = chunk_json.get("usage")
    if not isinstance(usage, dict):
        return False
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_tokens_details, dict):
        prompt_tokens_details = {}
    usage["prompt_tokens_details"] = prompt_tokens_details
    prompt_tokens_details["cached_tokens"] = cached_tokens
    return True


def encode_response_chunk(chunk_json: dict, is_sse: bool) -> bytes:
    chunk = json.dumps(chunk_json, ensure_ascii=False).encode("utf-8")
    return b"data: " + chunk + b"\n\n" if is_sse else chunk


def _affinity_value(value: Any, *, max_bytes: int | None = None) -> str | None:
    # Container string representations are not a stable cross-client contract
    # and can retain unexpectedly large request fragments in the shared LRU.
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text or (max_bytes is not None and len(text.encode("utf-8")) > max_bytes):
        return None
    return text


def _session_value(value: Any) -> str | None:
    return _affinity_value(value, max_bytes=MAX_AFFINITY_ID_BYTES)


def _normalized_session_key(*parts: str) -> str:
    """Build a bounded key without retaining a client-provided identifier."""
    payload = json.dumps(
        [SESSION_KEY_VERSION, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"session:{SESSION_KEY_VERSION}:{digest}"


def _session_candidates(
    headers: Mapping[str, str] | None,
    body: Mapping[str, Any] | None,
) -> list[tuple[str, tuple[str, ...]]]:
    candidates: list[tuple[str, tuple[str, ...]]] = []
    lowered = {str(name).lower(): value for name, value in (headers or {}).items()}

    explicit_affinity = _session_value(lowered.get("x-session-affinity"))
    if explicit_affinity is not None:
        candidates.append(("header:x-session-affinity", (explicit_affinity,)))

    thread_id = _session_value(lowered.get("thread-id"))
    if thread_id is not None:
        candidates.append(("header:thread-id", (thread_id,)))

    claude_session_id = _session_value(lowered.get("x-claude-code-session-id"))
    if claude_session_id is not None:
        claude_agent_id = _session_value(lowered.get("x-claude-code-agent-id"))
        parts = (claude_session_id, claude_agent_id) if claude_agent_id is not None else (claude_session_id,)
        candidates.append(("header:x-claude-code-session-id", parts))

    for name in SESSION_HEADER_NAMES[2:]:
        value = _session_value(lowered.get(name))
        if value is not None:
            candidates.append((f"header:{name}", (value,)))

    if not body:
        return candidates

    for name in ("session_id", "sessionId", "task_id", "taskId"):
        value = _session_value(body.get(name))
        if value is not None:
            candidates.append((f"body:{name}", (value,)))

    session_params = body.get("session_params")
    if isinstance(session_params, Mapping):
        value = _session_value(session_params.get("session_id"))
        if value is not None:
            candidates.append(("body:session_params.session_id", (value,)))

    # Gemini CLI's Code Assist schema nests its stable session UUID here.
    request = body.get("request")
    if isinstance(request, Mapping):
        value = _session_value(request.get("session_id"))
        if value is not None:
            candidates.append(("body:request.session_id", (value,)))

    return candidates


def extract_session_key(headers: Mapping[str, str] | None, body: Mapping[str, Any] | None) -> str | None:
    """Extract an explicitly session-scoped Prefiller affinity key."""
    candidates = _session_candidates(headers, body)
    if not candidates:
        return None
    source, parts = candidates[0]
    selected_key = _normalized_session_key(*parts)
    conflicts = [candidate_source for candidate_source, candidate_parts in candidates[1:] if candidate_parts != parts]
    if conflicts:
        logger.debug(
            "Conflicting affinity identifiers; selected %s over %s",
            source,
            ", ".join(conflicts),
        )
    return selected_key


def _canonical_prefix(body: Mapping[str, Any], prefix_chars: int) -> tuple[str, str] | None:
    # Tool definitions and calls are rendered by model-specific templates. A
    # text-only canonicalization can therefore differ from the actual token
    # prefix and falsely route to a node that owns no reusable KV blocks.
    if "tools" in body or "tool_choice" in body:
        return None

    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return "completions", prompt[:prefix_chars]

    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    canonical_messages: list[list[str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            return None
        role = message.get("role")
        content = message.get("content")
        # Structured content covers multimodal inputs and provider-specific
        # blocks. Skip it until canonicalization can reproduce the exact chat
        # template input rather than guessing from a partial representation.
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        if role == "tool" or any(name in message for name in ("tool_calls", "tool_call_id", "function_call")):
            return None
        canonical_messages.append([role, content])

    canonical = json.dumps(canonical_messages, ensure_ascii=False, separators=(",", ":"))
    return "chat", canonical[:prefix_chars]


def extract_prefix_key(body: Mapping[str, Any] | None, prefix_chars: int) -> str | None:
    """Hash a text-only request prefix without retaining prompt content."""
    if not body or prefix_chars <= 0:
        return None
    canonical = _canonical_prefix(body, prefix_chars)
    if canonical is None:
        return None
    endpoint_kind, prefix = canonical
    # Short requests do not contain enough reusable KV to justify overriding
    # normal load balancing, even though they could technically be hashed.
    if len(prefix) < prefix_chars:
        return None

    # Namespace by canonicalization version, endpoint, model, and the explicit
    # OpenAI prompt-cache bucket. Without these fields, identical text rendered
    # by different templates or cache namespaces could create false affinity.
    # Prompt cache key reference:
    # https://platform.openai.com/docs/guides/prompt-caching
    hash_input = json.dumps(
        [
            PREFIX_HASH_VERSION,
            endpoint_kind,
            _affinity_value(body.get("model")),
            _affinity_value(body.get("prompt_cache_key")),
            prefix,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # Store only a fixed-size digest in the cross-worker scheduler so prompts
    # are neither retained as routing keys nor allowed to inflate LRU memory.
    return hashlib.blake2s(hash_input.encode("utf-8"), digest_size=16).hexdigest()


def extract_affinity_keys(
    headers: Mapping[str, str] | None,
    body: Mapping[str, Any] | None,
    *,
    enabled: bool,
    prefix_chars: int,
) -> tuple[str | None, str | None]:
    # Make cache-aware routing opt-in so existing deployments can update this
    # example without silently changing where long text requests are routed.
    if not enabled:
        return None, None
    return extract_session_key(headers, body), extract_prefix_key(body, prefix_chars)


global_args: argparse.Namespace | None = None
shared_scheduler: "SharedProxyScheduler | None" = None
runtime: "WorkerRuntime | None" = None


@dataclass
class BackendServer:
    host: str
    port: int
    ordinal: int
    active_tokens: float = 0.0
    active_kv_cache: float = 0.0
    selections: int = 0
    heap_seq: int = 0


@dataclass
class RolePools:
    """Per-role scheduling state: live servers, priority heap, and drain-isolated keys."""

    servers: dict[str, BackendServer] = field(default_factory=dict)
    heap: list[tuple[float, int, int, str]] = field(default_factory=list)
    tainted: set[str] = field(default_factory=set)


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logger.setLevel(getattr(logging, log_level.upper()))


def next_req_id() -> str:
    return str(uuid.uuid4())


def calculate_prefill_score(request_length: int) -> float:
    length_score = request_length / 4.0
    return length_score * 0.0345 + 120.0745


def calculate_decode_score(request_length: int) -> float:
    return request_length


def normalize_host(host: str) -> str:
    return host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")


def server_key(host: str, port: int) -> str:
    return f"{normalize_host(host)}:{int(port)}"


def build_server_url(host: str, port: int) -> str:
    url = f"http://{host}:{port}"
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address):
            url = f"http://[{host}]:{port}"
    except Exception:
        pass
    return url


def build_base_url(host: str, port: int) -> str:
    return f"{build_server_url(host, port)}/v1"


class SharedProxyScheduler:
    """Centralized mutable scheduling state shared by all uvicorn workers.

    Uses lazy-deletion min-heap: on priority change, push a new entry and
    bump the server's ``heap_seq`` counter; stale entries (whose seq does
    not match) are skipped on pop.
    """

    def __init__(
        self,
        prefiller_instances,
        decoder_instances,
        *,
        session_lru_size: int = 4096,
        prefix_lru_size: int = 1024,
        prefill_active_token_weight: float = 1.0,
        affinity_overload_factor: float = 0.0,
        affinity_miss_unbind_threshold: int = 0,
        affinity_cache_discount_alpha: float = 0.0,
    ):
        self._lock = threading.RLock()
        self.request_num = 0
        self.waiting_nodes: dict[str, tuple[str, tuple[str, int], int]] = {}
        self._pools: dict[ServerRole, RolePools] = {
            ServerRole.PREFILL: RolePools(),
            ServerRole.DECODE: RolePools(),
        }
        self._ordinal = 0
        self.session_lru_size = max(0, int(session_lru_size))
        self.prefix_lru_size = max(0, int(prefix_lru_size))
        self.prefill_active_token_weight = max(0.0, float(prefill_active_token_weight))
        self.affinity_overload_factor = max(0.0, float(affinity_overload_factor))
        self.affinity_miss_unbind_threshold = max(0, int(affinity_miss_unbind_threshold))
        self.affinity_cache_discount_alpha = min(1.0, max(0.0, float(affinity_cache_discount_alpha)))
        self.session_lru: OrderedDict[str, str] = OrderedDict()
        self.prefix_lru: OrderedDict[str, str] = OrderedDict()
        self.session_miss_streak: dict[str, int] = {}
        self.prefix_miss_streak: dict[str, int] = {}
        # Per-binding EMA of cached_tokens/prompt_tokens observed on affinity
        # hits; used to discount reservations toward true compute cost.
        self.session_cache_ema: dict[str, float] = {}
        self.prefix_cache_ema: dict[str, float] = {}
        self.prefill_routing_stats = {
            "heap_decisions": 0,
            "shadow_baseline_active_divergences": 0,
            "actual_differs_from_baseline": 0,
            "actual_differs_from_active": 0,
        }
        self.session_affinity_stats = self._empty_affinity_stats()
        self.prefix_affinity_stats = self._empty_affinity_stats()
        self.prefill_cache_stats_by_source: dict[str, dict[str, int]] = {}

        for host, port in prefiller_instances:
            self._add_server_no_lock(ServerRole.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(ServerRole.DECODE, host, port)

    @staticmethod
    def _empty_affinity_stats() -> dict[str, int]:
        return {
            "hits": 0,
            "binds": 0,
            "overflows": 0,
            "unbinds_lru": 0,
            "unbinds_taint": 0,
            "unbinds_miss": 0,
        }

    def _pool(self, role: ServerRole) -> RolePools:
        return self._pools[role]

    @property
    def prefillers(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.PREFILL).servers

    @property
    def decoders(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.DECODE).servers

    def _next_ordinal(self) -> int:
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    def _priority(self, role: ServerRole, entry: BackendServer, key: str) -> float:
        if key in self._pool(role).tainted:
            return TAINT_PRIORITY
        if role is ServerRole.PREFILL:
            # active_tokens is released when Prefill finishes, while KV pressure
            # lasts until Decoder starts consuming the transfer. This restores
            # the two-phase accounting used before the shared-scheduler refactor.
            return self.prefill_active_token_weight * entry.active_tokens + entry.active_kv_cache * 0.3
        return entry.active_tokens

    def _push_heap(self, role: ServerRole, key: str) -> None:
        pool = self._pool(role)
        entry = pool.servers[key]
        entry.heap_seq += 1
        heapq.heappush(pool.heap, (self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        if len(pool.heap) > 2 * len(pool.servers):
            self._reset_heap(role)

    def _pop_valid(self, role: ServerRole) -> str:
        pool = self._pool(role)
        while pool.heap:
            _, _, seq, key = heapq.heappop(pool.heap)
            if key not in pool.servers:
                continue
            entry = pool.servers[key]
            if entry.heap_seq == seq:
                return key
        raise RuntimeError(f"No available {role.value} servers")

    def _reset_heap(self, role: ServerRole, *, bump_seq: bool = False) -> None:
        pool = self._pool(role)
        heap = []
        for key, entry in pool.servers.items():
            if bump_seq:
                entry.heap_seq += 1
            heap.append((self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        heapq.heapify(heap)
        pool.heap = heap

    def _add_server_no_lock(self, role: ServerRole, host: str, port: int) -> bool:
        key = server_key(host, port)
        pool = self._pool(role)
        if key in pool.servers:
            return False
        pool.servers[key] = BackendServer(host, int(port), self._next_ordinal())
        self._push_heap(role, key)
        return True

    def get_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "prefill_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(self.prefillers.items(), key=lambda item: item[1].ordinal)
                ],
                "decode_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(self.decoders.items(), key=lambda item: item[1].ordinal)
                ],
            }

    def log_status(self, msg: str) -> None:
        snapshot = self.get_snapshot()
        logger.info(
            "%s prefill=%s decode=%s",
            msg,
            [f"{s['host']}:{s['port']}" for s in snapshot["prefill_instances"]],
            [f"{s['host']}:{s['port']}" for s in snapshot["decode_instances"]],
        )

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "prefill_instances": len(self.prefillers),
                "decode_instances": len(self.decoders),
                "request_num": self.request_num,
                "prefill_active_token_weight": self.prefill_active_token_weight,
                "affinity_overload_factor": self.affinity_overload_factor,
                "affinity_miss_unbind_threshold": self.affinity_miss_unbind_threshold,
                "affinity_cache_discount_alpha": self.affinity_cache_discount_alpha,
                "prefill_routing_stats": dict(self.prefill_routing_stats),
                "session_affinity_stats": dict(self.session_affinity_stats),
                "prefix_affinity_stats": dict(self.prefix_affinity_stats),
                "prefill_cache_stats_by_source": {
                    source: dict(stats) for source, stats in self.prefill_cache_stats_by_source.items()
                },
                "prefill_loads": {
                    key: {
                        "host": entry.host,
                        "port": entry.port,
                        "ordinal": entry.ordinal,
                        "active_tokens": entry.active_tokens,
                        "active_kv_cache": entry.active_kv_cache,
                        "priority": self._priority(ServerRole.PREFILL, entry, key),
                        "selections": entry.selections,
                        "tainted": key in self._pool(ServerRole.PREFILL).tainted,
                    }
                    for key, entry in sorted(self.prefillers.items(), key=lambda item: item[1].ordinal)
                },
            }

    def _pick_server(
        self,
        role: ServerRole,
        *,
        key: str | None = None,
        active_tokens_load: float = 0.0,
        kv_cache_load: float = 0.0,
    ) -> dict[str, Any]:
        # Affinity may have already selected a Prefiller. Accepting that key
        # here keeps node selection and both load reservations on the same
        # scheduler primitive instead of duplicating heap/accounting updates.
        if key is None:
            key = self._pop_valid(role)
        entry = self._pool(role).servers[key]
        entry.selections += 1
        entry.active_tokens += active_tokens_load
        entry.active_kv_cache += kv_cache_load
        self._push_heap(role, key)
        return {"key": key, "host": entry.host, "port": entry.port}

    def _release_load(
        self,
        role: ServerRole,
        key: str | None,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> None:
        if not key or key not in self._pool(role).servers:
            return
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens -= load
        if kv_cache:
            entry.active_kv_cache = max(0.0, entry.active_kv_cache - load)
        self._push_heap(role, key)

    def _affinity_stats_for(self, kind: str) -> dict[str, int]:
        return self.session_affinity_stats if kind == "session" else self.prefix_affinity_stats

    def _affinity_miss_streak_for(self, kind: str) -> dict[str, int]:
        return self.session_miss_streak if kind == "session" else self.prefix_miss_streak

    def _affinity_mapping_for(self, kind: str) -> OrderedDict[str, str]:
        return self.session_lru if kind == "session" else self.prefix_lru

    def _affinity_cache_ema_for(self, kind: str) -> dict[str, float]:
        return self.session_cache_ema if kind == "session" else self.prefix_cache_ema

    def _affinity_reserve_fraction_no_lock(self, kind: str, affinity_key: str | None) -> float:
        """Fraction of the full score an affinity-hit reservation should cost.

        A bound node's cached prefix means the real prefill compute is roughly
        (1 - hit_ratio) of the full prompt. Reserving the full score makes the
        node look busier than it is, which biases the overload guard against
        exactly the nodes affinity is helping.
        """
        if self.affinity_cache_discount_alpha <= 0 or not affinity_key:
            return 1.0
        ema = self._affinity_cache_ema_for(kind).get(affinity_key)
        if ema is None:
            return 1.0
        return max(AFFINITY_DISCOUNT_MIN_FRACTION, 1.0 - ema)

    def _min_live_prefill_priority_no_lock(self) -> float | None:
        pool = self._pool(ServerRole.PREFILL)
        live = [
            self._priority(ServerRole.PREFILL, entry, key)
            for key, entry in self.prefillers.items()
            if key not in pool.tainted
        ]
        return min(live) if live else None

    def _affinity_overloaded_no_lock(self, prefiller_key: str) -> bool:
        if self.affinity_overload_factor <= 0:
            return False
        pool = self._pool(ServerRole.PREFILL)
        entry = pool.servers.get(prefiller_key)
        if entry is None or prefiller_key in pool.tainted:
            return True
        bound_priority = self._priority(ServerRole.PREFILL, entry, prefiller_key)
        min_priority = self._min_live_prefill_priority_no_lock()
        if min_priority is None:
            return False
        return bound_priority > min_priority * self.affinity_overload_factor + AFFINITY_OVERLOAD_MARGIN

    def _resolve_affinity_no_lock(
        self,
        mapping: OrderedDict[str, str],
        affinity_key: str | None,
        *,
        kind: str,
    ) -> str | None:
        if not affinity_key:
            return None
        prefiller_key = mapping.get(affinity_key)
        if prefiller_key is None:
            return None
        pool = self._pool(ServerRole.PREFILL)
        stats = self._affinity_stats_for(kind)
        miss_streak = self._affinity_miss_streak_for(kind)
        # A draining node must stop receiving affinity traffic immediately;
        # lazily deleting its entry also makes stale mappings self-healing.
        if prefiller_key not in pool.servers or prefiller_key in pool.tainted:
            mapping.pop(affinity_key, None)
            miss_streak.pop(affinity_key, None)
            self._affinity_cache_ema_for(kind).pop(affinity_key, None)
            stats["unbinds_taint"] += 1
            return None
        mapping.move_to_end(affinity_key)
        return prefiller_key

    def _record_heap_shadow_stats_no_lock(self, picked_key: str) -> None:
        live_prefillers = [
            (key, entry)
            for key, entry in self.prefillers.items()
            if key not in self._pool(ServerRole.PREFILL).tainted
        ]
        if not live_prefillers:
            return
        baseline_shadow_key = min(
            live_prefillers,
            key=lambda item: (0.3 * item[1].active_kv_cache, item[1].ordinal),
        )[0]
        active_shadow_key = min(
            live_prefillers,
            key=lambda item: (item[1].active_tokens + 0.3 * item[1].active_kv_cache, item[1].ordinal),
        )[0]
        self.prefill_routing_stats["heap_decisions"] += 1
        if baseline_shadow_key != active_shadow_key:
            self.prefill_routing_stats["shadow_baseline_active_divergences"] += 1
        if picked_key != baseline_shadow_key:
            self.prefill_routing_stats["actual_differs_from_baseline"] += 1
        if picked_key != active_shadow_key:
            self.prefill_routing_stats["actual_differs_from_active"] += 1

    def _reserve_prefiller_no_lock(
        self,
        compute_load: float,
        kv_load: float,
        session_key: str | None,
        prefix_key: str | None,
    ) -> dict[str, Any]:
        # A session is an explicit locality contract, while a prefix match is
        # only inferred locality. Fall back to the existing load heap only when
        # neither source identifies a live Prefiller.
        # Use case: https://github.com/vllm-project/vllm-ascend/issues/12196
        route_source = "session"
        prefiller_key = self._resolve_affinity_no_lock(self.session_lru, session_key, kind="session")
        if prefiller_key is not None:
            self.session_affinity_stats["hits"] += 1
            if self._affinity_overloaded_no_lock(prefiller_key):
                self.session_affinity_stats["overflows"] += 1
                route_source = "session-overflow"
                prefiller_key = None
        if prefiller_key is None and route_source != "session-overflow":
            route_source = "prefix"
            prefiller_key = self._resolve_affinity_no_lock(self.prefix_lru, prefix_key, kind="prefix")
            if prefiller_key is not None:
                self.prefix_affinity_stats["hits"] += 1
                if self._affinity_overloaded_no_lock(prefiller_key):
                    self.prefix_affinity_stats["overflows"] += 1
                    route_source = "prefix-overflow"
                    prefiller_key = None
        if prefiller_key is None and route_source not in ("session-overflow", "prefix-overflow"):
            route_source = "heap"
        # Affinity hits reserve a cache-discounted compute cost so accounting
        # tracks real prefill work. KV is deliberately NOT discounted: a prefix
        # cache hit skips computation, but the decoder still needs the full
        # prompt KV, so transfer/residency pressure is unchanged. The caller
        # must release exactly what was reserved, so the effective loads are
        # returned alongside the pick.
        if route_source == "session":
            fraction = self._affinity_reserve_fraction_no_lock("session", session_key)
        elif route_source == "prefix":
            fraction = self._affinity_reserve_fraction_no_lock("prefix", prefix_key)
        else:
            fraction = 1.0
        effective_compute_load = compute_load * fraction
        effective_kv_load = kv_load
        # Reserve both phases through the original scheduler primitive so
        # another request sees the complete cost before making its decision.
        picked = self._pick_server(
            ServerRole.PREFILL,
            key=prefiller_key,
            active_tokens_load=effective_compute_load,
            kv_cache_load=effective_kv_load,
        )
        picked["route_source"] = route_source
        picked["compute_load"] = effective_compute_load
        picked["kv_load"] = effective_kv_load
        if route_source in ("heap", "session-overflow", "prefix-overflow"):
            self._record_heap_shadow_stats_no_lock(picked["key"])
        return picked

    def _bind_affinity_no_lock(
        self,
        mapping: OrderedDict[str, str],
        affinity_key: str,
        prefiller_key: str,
        capacity: int,
        *,
        kind: str,
    ) -> None:
        # Affinity keys are supplied by clients. A bounded LRU prevents their
        # cardinality from turning shared scheduler state into an unbounded map.
        if capacity <= 0:
            return
        pool = self._pool(ServerRole.PREFILL)
        # Do not create new ownership claims for a node that is being drained.
        if prefiller_key not in pool.servers or prefiller_key in pool.tainted:
            return
        stats = self._affinity_stats_for(kind)
        miss_streak = self._affinity_miss_streak_for(kind)
        cache_ema = self._affinity_cache_ema_for(kind)
        previous = mapping.get(affinity_key)
        mapping[affinity_key] = prefiller_key
        mapping.move_to_end(affinity_key)
        # Refreshing the same owner must preserve the miss streak so consecutive
        # zero-cache outcomes can still trip the unbind threshold. A different
        # owner starts cold: its hit-ratio history belongs to the old node.
        if previous != prefiller_key:
            miss_streak[affinity_key] = 0
            cache_ema.pop(affinity_key, None)
        else:
            miss_streak.setdefault(affinity_key, 0)
        stats["binds"] += 1
        while len(mapping) > capacity:
            evicted_key, _ = mapping.popitem(last=False)
            miss_streak.pop(evicted_key, None)
            cache_ema.pop(evicted_key, None)
            stats["unbinds_lru"] += 1

    def begin_request(
        self,
        compute_load: float,
        kv_load: float,
        session_key: str | None = None,
        prefix_key: str | None = None,
    ) -> dict[str, Any]:
        """Pick a prefiller and reserve compute plus KV-transfer pressure."""
        with self._lock:
            picked = self._reserve_prefiller_no_lock(compute_load, kv_load, session_key, prefix_key)
            self.request_num += 1
            return picked

    def reserve_prefill_kv(
        self,
        compute_load: float,
        kv_load: float,
        session_key: str | None = None,
        prefix_key: str | None = None,
    ) -> dict[str, Any]:
        """Reserve a prefiller for recompute without bumping the request count."""
        with self._lock:
            return self._reserve_prefiller_no_lock(compute_load, kv_load, session_key, prefix_key)

    def complete_prefill(
        self,
        key: str,
        compute_load: float,
        session_key: str | None,
        prefix_key: str | None,
        route_source: str = "heap",
        allow_affinity: bool = True,
    ) -> None:
        """Release compute pressure and commit affinity after a successful Prefill."""
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, compute_load, active_tokens=True)
            if not allow_affinity:
                # Gate decided this Prefill left no reusable prefix worth pinning.
                # Compute load is still released; session/prefix maps are touched
                # only by callers that opt in.
                return
            # A failed Prefill cannot own reusable KV, so mappings are committed
            # here rather than when the node is initially selected.
            if session_key:
                self._bind_affinity_no_lock(
                    self.session_lru,
                    session_key,
                    key,
                    self.session_lru_size,
                    kind="session",
                )
            # When an existing session wins over a conflicting prefix mapping,
            # keep the prefix owner stable. Rebinding it on every session hit
            # would make the one-node prefix LRU flap between Prefillers.
            # Overflow and heap paths rebind so the abandoned hot node loses
            # ownership.
            if prefix_key and route_source != "session":
                self._bind_affinity_no_lock(
                    self.prefix_lru,
                    prefix_key,
                    key,
                    self.prefix_lru_size,
                    kind="prefix",
                )

    def observe_affinity_outcome(
        self,
        route_source: str,
        affinity_key: str | None,
        cached_tokens: int | None,
        prompt_tokens: int | None = None,
        prefiller_key: str | None = None,
    ) -> None:
        """Record cache outcome and optionally unbind a stale affinity mapping."""
        with self._lock:
            source_stats = self.prefill_cache_stats_by_source.setdefault(
                route_source,
                {"cached_tokens": 0, "prompt_tokens": 0, "requests": 0},
            )
            source_stats["requests"] += 1
            if isinstance(cached_tokens, int):
                source_stats["cached_tokens"] += max(0, cached_tokens)
            if isinstance(prompt_tokens, int):
                source_stats["prompt_tokens"] += max(0, prompt_tokens)

            if route_source not in ("session", "prefix") or not affinity_key:
                return
            mapping = self._affinity_mapping_for(route_source)
            if affinity_key not in mapping:
                return
            # An outcome observed on a node the mapping no longer points at
            # (e.g. an overflow rebound the key mid-flight) must not count
            # against — or unbind — the fresh binding.
            if prefiller_key is not None and mapping.get(affinity_key) != prefiller_key:
                return

            if (
                self.affinity_cache_discount_alpha > 0
                and isinstance(cached_tokens, int)
                and isinstance(prompt_tokens, int)
                and prompt_tokens > 0
            ):
                cache_ema = self._affinity_cache_ema_for(route_source)
                ratio = min(1.0, max(0.0, cached_tokens / prompt_tokens))
                previous_ema = cache_ema.get(affinity_key)
                if previous_ema is None:
                    cache_ema[affinity_key] = ratio
                else:
                    alpha = self.affinity_cache_discount_alpha
                    cache_ema[affinity_key] = alpha * ratio + (1.0 - alpha) * previous_ema

            if self.affinity_miss_unbind_threshold <= 0 or not isinstance(cached_tokens, int):
                return
            miss_streak = self._affinity_miss_streak_for(route_source)
            stats = self._affinity_stats_for(route_source)
            if cached_tokens > 0:
                miss_streak[affinity_key] = 0
                return
            streak = miss_streak.get(affinity_key, 0) + 1
            miss_streak[affinity_key] = streak
            if streak < self.affinity_miss_unbind_threshold:
                return
            mapping.pop(affinity_key, None)
            miss_streak.pop(affinity_key, None)
            self._affinity_cache_ema_for(route_source).pop(affinity_key, None)
            stats["unbinds_miss"] += 1

    def abort_prefill_reservation(
        self,
        key: str,
        compute_load: float,
        kv_load: float,
        count_request: bool,
    ) -> None:
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, compute_load, active_tokens=True)
            self._release_load(ServerRole.PREFILL, key, kv_load, kv_cache=True)
            if count_request:
                self.request_num = max(0, self.request_num - 1)

    def pick_decoder(self, load: float) -> dict[str, Any]:
        with self._lock:
            return self._pick_server(ServerRole.DECODE, active_tokens_load=load)

    def release_prefill_kv(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, load, kv_cache=True)

    def clear_affinity_caches(self) -> None:
        with self._lock:
            self.session_lru.clear()
            self.prefix_lru.clear()
            self.session_miss_streak.clear()
            self.prefix_miss_streak.clear()
            self.session_cache_ema.clear()
            self.prefix_cache_ema.clear()

    def release_decoder(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.DECODE, key, load, active_tokens=True)

    def finish_request(
        self,
        prefiller_key: str | None,
        prefiller_load: float,
        decoder_key: str | None,
        decoder_load: float,
        release_prefill_kv: bool,
    ) -> None:
        with self._lock:
            if release_prefill_kv:
                self._release_load(ServerRole.PREFILL, prefiller_key, prefiller_load, kv_cache=True)
            self._release_load(ServerRole.DECODE, decoder_key, decoder_load, active_tokens=True)
            self.request_num = max(0, self.request_num - 1)

    def get_waiting_nodes(self) -> dict[str, tuple[str, tuple[str, int], int]]:
        with self._lock:
            return dict(self.waiting_nodes)

    def add_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> list[str]:
        waiting_nodes: list[str] = []
        with self._lock:
            servers = self._pool(role).servers
            for host, port in instances:
                key = server_key(host, port)
                if key in servers or key in self.waiting_nodes:
                    continue
                self.waiting_nodes[key] = (role.value, (host, int(port)), 0)
                waiting_nodes.append(f"{host}:{port}")
        return waiting_nodes

    def mark_waiting_retry(self, key: str, retry_count: int) -> None:
        with self._lock:
            if key not in self.waiting_nodes:
                return
            instance_type, server, _ = self.waiting_nodes[key]
            self.waiting_nodes[key] = (instance_type, server, retry_count)

    def activate_waiting_instance(self, role: ServerRole, host: str, port: int) -> None:
        with self._lock:
            key = server_key(host, port)
            self.waiting_nodes.pop(key, None)
            pool = self._pool(role)
            if key in pool.tainted:
                pool.tainted.discard(key)
                self._push_heap(role, key)
                return
            if self._add_server_no_lock(role, host, port):
                self.log_status(f"Add {role.value} instance: {host}:{port}.")

    def drop_waiting_instance(self, key: str) -> None:
        with self._lock:
            self.waiting_nodes.pop(key, None)

    def _prune_affinity_no_lock(self, prefiller_keys: set[str]) -> None:
        for kind, mapping in (("session", self.session_lru), ("prefix", self.prefix_lru)):
            miss_streak = self._affinity_miss_streak_for(kind)
            cache_ema = self._affinity_cache_ema_for(kind)
            stats = self._affinity_stats_for(kind)
            stale_keys = [affinity_key for affinity_key, key in mapping.items() if key in prefiller_keys]
            for affinity_key in stale_keys:
                mapping.pop(affinity_key, None)
                miss_streak.pop(affinity_key, None)
                cache_ema.pop(affinity_key, None)
                stats["unbinds_taint"] += 1

    def remove_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> bool:
        if not instances:
            return False
        keys = {server_key(host, port) for host, port in instances}
        with self._lock:
            pool = self._pool(role)
            if self.request_num > 0:
                pool.tainted.update(keys)
                self._reset_heap(role, bump_seq=True)
                logger.warning("Start to taint %s instances %s.", role.value, sorted(keys))
                return True

            removed = False
            for key in keys:
                removed = pool.servers.pop(key, None) is not None or removed
                self.waiting_nodes.pop(key, None)
            pool.tainted.difference_update(keys)
            if removed:
                if role is ServerRole.PREFILL:
                    self._prune_affinity_no_lock(keys)
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances: {sorted(keys)}.")
            return False

    def finalize_tainted_instances(self) -> None:
        with self._lock:
            if self.request_num != 0:
                return
            for role in ServerRole:
                pool = self._pool(role)
                if not pool.tainted:
                    continue
                keys = list(pool.tainted)
                for key in keys:
                    pool.servers.pop(key, None)
                pool.tainted.clear()
                if role is ServerRole.PREFILL:
                    self._prune_affinity_no_lock(set(keys))
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances after drain: {keys}.")


class SchedulerManager(BaseManager):
    """Multiprocessing RPC bridge; body is empty but required by BaseManager."""


def _shared_scheduler_proxy() -> "SharedProxyScheduler":
    if shared_scheduler is None:
        raise RuntimeError("shared scheduler is not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=_shared_scheduler_proxy)


class WorkerRuntime:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        self._clients: dict[ServerRole, dict[str, httpx.AsyncClient]] = {
            ServerRole.PREFILL: {},
            ServerRole.DECODE: {},
        }
        self._async_lock = asyncio.Lock()

    async def schedule(self, method: str, /, *args, **kwargs) -> Any:
        async with self._async_lock:
            return getattr(self.scheduler, method)(*args, **kwargs)

    async def get_client(self, role: ServerRole, key: str) -> httpx.AsyncClient:
        clients = self._clients[role]
        if key not in clients:
            await self.sync_clients()
        return clients[key]

    async def sync_clients(self) -> None:
        snapshot = self.scheduler.get_snapshot()
        role_targets = {
            ServerRole.PREFILL: {
                server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["prefill_instances"]
            },
            ServerRole.DECODE: {
                server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["decode_instances"]
            },
        }
        for role, targets in role_targets.items():
            await self._sync_clients(role, targets)

    async def _sync_clients(self, role: ServerRole, targets: dict[str, tuple[str, int]]) -> None:
        clients = self._clients[role]
        for key in [key for key in clients if key not in targets]:
            await clients.pop(key).aclose()
        for key, (host, port) in targets.items():
            if key in clients:
                continue
            clients[key] = httpx.AsyncClient(
                timeout=None,
                base_url=build_base_url(host, port),
                limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
            )

    async def close(self) -> None:
        for role in ServerRole:
            for client in list(self._clients[role].values()):
                await client.aclose()
            self._clients[role].clear()


def get_runtime() -> WorkerRuntime:
    if runtime is None:
        raise RuntimeError("worker runtime is not initialized")
    return runtime


class NodeListener:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            args = get_global_args()
            for key, (instance_type, server, retries) in list(self.scheduler.get_waiting_nodes().items()):
                host, port = server
                is_valid = asyncio.run(self.check_instance_status(host, port))
                print(f"Checking instance {key}...")
                retries += 1
                if is_valid:
                    self.scheduler.activate_waiting_instance(ServerRole(instance_type), host, port)
                elif retries >= args.max_waiting_retries:
                    print(f"Instance {key} was not added to the proxy.")
                    self.scheduler.drop_waiting_instance(key)
                else:
                    self.scheduler.mark_waiting_retry(key, retries)

            self.scheduler.finalize_tainted_instances()
            time.sleep(args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(host: str, port: int) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            async with httpx.AsyncClient(timeout=5.0, base_url=build_base_url(host, port)) as client:
                response = await client.get(endpoint, headers=headers)
                response.raise_for_status()
                return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def manager_config_path(proxy_port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"vllm_lb_proxy_manager_{proxy_port}.json"


def write_manager_config(proxy_port: int, host: str, manager_port: int, authkey: bytes) -> None:
    manager_config_path(proxy_port).write_text(
        json.dumps(
            {
                "host": host,
                "port": manager_port,
                "authkey": base64.b64encode(authkey).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )


def read_manager_config(proxy_port: int) -> dict[str, Any]:
    path = manager_config_path(proxy_port)
    if not path.is_file():
        raise RuntimeError(
            f"Manager config not found at {path}. "
            "Start the proxy from __main__ with --workers > 1 before worker processes connect."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_manager_config(proxy_port: int) -> None:
    manager_config_path(proxy_port).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes. Scheduling state is shared across workers.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the proxy server.",
    )
    parser.add_argument(
        "--prefill-active-token-weight",
        type=float,
        default=1.0,
        help=(
            "Weight applied to in-flight Prefill compute load. Set to 0 to retain "
            "candidate instrumentation while matching the upstream KV-only heap priority."
        ),
    )
    parser.add_argument(
        "--enable-kv-cache-aware-routing",
        action="store_true",
        help="Enable session and text-prefix affinity for Prefiller routing.",
    )
    parser.add_argument(
        "--enable-reusable-prefix-affinity-gate",
        action="store_true",
        help=(
            "Gate session/prefix affinity commit on the Prefill response's "
            "reusable_prefix_tokens (cached_tokens + created_cache_tokens). "
            "No-op without --enable-kv-cache-aware-routing. Prefillers should run "
            "with --enable-prompt-tokens-details."
        ),
    )
    parser.add_argument(
        "--session-lru-size",
        type=int,
        default=4096,
        help=(
            "Maximum session-to-prefiller entries when KV-cache-aware routing is enabled; 0 disables session affinity."
        ),
    )
    parser.add_argument(
        "--prefix-hash-chars",
        type=int,
        default=1024,
        help=("Text prefix characters used when KV-cache-aware routing is enabled; 0 disables prefix affinity."),
    )
    parser.add_argument(
        "--prefix-lru-size",
        type=int,
        default=1024,
        help=(
            "Maximum prefix-to-prefiller entries when KV-cache-aware routing is enabled; 0 disables prefix affinity."
        ),
    )
    parser.add_argument(
        "--affinity-overload-factor",
        type=float,
        default=0.0,
        help=(
            "Escape affinity when the bound Prefiller priority exceeds "
            "min_live_priority * factor + margin. 0 disables the guard; "
            "enabled values must be >= 1 or the guard would flag the "
            "least-loaded node as overloaded."
        ),
    )
    parser.add_argument(
        "--affinity-miss-unbind-threshold",
        type=int,
        default=0,
        help=(
            "Unbind an affinity key after this many consecutive zero-cached_tokens "
            "outcomes on affinity hits. 0 disables miss-based unbind."
        ),
    )
    parser.add_argument(
        "--affinity-cache-discount-alpha",
        type=float,
        default=0.0,
        help=(
            "EMA smoothing factor for per-binding cache hit ratio. When > 0, "
            "affinity-hit reservations are discounted by the observed hit ratio "
            "so load accounting tracks real prefill compute instead of raw "
            "prompt size. 0 disables discounting."
        ),
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.session_lru_size = max(0, args.session_lru_size)
    args.prefix_hash_chars = max(0, args.prefix_hash_chars)
    args.prefix_lru_size = max(0, args.prefix_lru_size)
    if not math.isfinite(args.prefill_active_token_weight) or args.prefill_active_token_weight < 0:
        raise ValueError("prefill active-token weight must be finite and non-negative")
    if not math.isfinite(args.affinity_overload_factor) or args.affinity_overload_factor < 0:
        raise ValueError("affinity overload factor must be finite and non-negative")
    if 0 < args.affinity_overload_factor < 1:
        raise ValueError("affinity overload factor must be 0 (disabled) or >= 1")
    if (
        not math.isfinite(args.affinity_cache_discount_alpha)
        or not 0.0 <= args.affinity_cache_discount_alpha <= 1.0
    ):
        raise ValueError("affinity cache discount alpha must be within [0, 1]")
    if args.affinity_miss_unbind_threshold < 0:
        raise ValueError("affinity miss-unbind threshold must be non-negative")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def _scheduler_affinity_kwargs(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        "session_lru_size": args.session_lru_size,
        "prefix_lru_size": args.prefix_lru_size,
        "prefill_active_token_weight": args.prefill_active_token_weight,
        "affinity_overload_factor": args.affinity_overload_factor,
        "affinity_miss_unbind_threshold": args.affinity_miss_unbind_threshold,
        "affinity_cache_discount_alpha": args.affinity_cache_discount_alpha,
    }


def get_global_args() -> argparse.Namespace:
    global global_args
    if global_args is None:
        global_args = parse_args()
    return global_args


def connect_shared_scheduler(proxy_port: int):
    manager_cfg = read_manager_config(proxy_port)
    manager = SchedulerManager(
        address=(manager_cfg["host"], manager_cfg["port"]),
        authkey=base64.b64decode(manager_cfg["authkey"]),
    )
    manager.connect()
    return manager.get_scheduler()  # type: ignore[attr-defined]


def bootstrap_parent_process(args: argparse.Namespace) -> None:
    """Initialize cross-worker shared state in the parent process before uvicorn spawns workers."""
    global shared_scheduler
    if args.workers <= 1:
        return

    shared_scheduler = SharedProxyScheduler(
        args.prefiller_instances,
        args.decoder_instances,
        **_scheduler_affinity_kwargs(args),
    )
    NodeListener(shared_scheduler)

    authkey = os.urandom(16)
    manager = SchedulerManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.address)
    write_manager_config(args.port, host, port, authkey)


def _ensure_scheduler(args) -> SharedProxyScheduler:
    global shared_scheduler
    if shared_scheduler is not None:
        return shared_scheduler
    shared_scheduler = SharedProxyScheduler(
        args.prefiller_instances,
        args.decoder_instances,
        **_scheduler_affinity_kwargs(args),
    )
    NodeListener(shared_scheduler)
    return shared_scheduler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    args = get_global_args()
    if args.workers > 1:
        scheduler = connect_shared_scheduler(args.port)
    else:
        scheduler = _ensure_scheduler(args)
    runtime = WorkerRuntime(scheduler)
    await runtime.sync_clients()
    snapshot = scheduler.get_snapshot()
    logger.info(
        "Initialized %s prefill clients and %s decode clients in worker %s.",
        len(snapshot["prefill_instances"]),
        len(snapshot["decode_instances"]),
        os.getpid(),
    )
    yield
    await runtime.close()
    runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    setup_logging(get_global_args().log_level)
    return app


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def auth_headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


def build_prefill_request(req_data: dict) -> dict:
    payload = req_data.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


async def send_request_to_service(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    req_data = build_prefill_request(req_data)
    headers = auth_headers(request_id)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = auth_headers(request_id)
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise exc
        except Exception as exc:
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", exc)
                return
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise exc


async def _abort_prefill_selection(
    runtime: WorkerRuntime,
    prefiller_key: str,
    prefiller_compute_score: float,
    prefiller_kv_score: float,
    *,
    is_initial_request: bool,
) -> None:
    """Roll back both pressures when Prefill has not completed successfully."""
    await runtime.schedule(
        "abort_prefill_reservation",
        prefiller_key,
        prefiller_compute_score,
        prefiller_kv_score,
        is_initial_request,
    )


async def _abort_after_prefill(
    runtime: WorkerRuntime,
    prefiller_key: str,
    prefiller_kv_score: float,
    *,
    is_initial_request: bool,
) -> None:
    """Release only KV/request state after compute was already completed."""
    if is_initial_request:
        await runtime.schedule("finish_request", prefiller_key, prefiller_kv_score, None, 0.0, release_prefill_kv=True)
    else:
        await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_kv_score)


async def _finish_instance(runtime: WorkerRuntime, info: InstanceInfo, *, release_prefill_kv: bool) -> None:
    await runtime.schedule(
        "finish_request",
        info.prefiller_key,
        info.prefiller_score,
        info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
    )


async def assign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
    session_key: str | None = None,
    prefix_key: str | None = None,
) -> InstanceInfo:
    runtime = get_runtime()
    args = get_global_args()
    prefiller_compute_score = prefiller_kv_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    prefiller = await runtime.schedule(
        pick_prefill,
        prefiller_compute_score,
        prefiller_kv_score,
        session_key,
        prefix_key,
    )
    prefiller_key = prefiller["key"]
    # Cache-discounted reservations must be released at exactly the reserved
    # amount, so all downstream complete/abort/release calls use the effective
    # loads the scheduler actually booked.
    prefiller_compute_score = float(prefiller.get("compute_load", prefiller_compute_score))
    prefiller_kv_score = float(prefiller.get("kv_load", prefiller_kv_score))

    try:
        response = await send_request_to_service(
            await runtime.get_client(ServerRole.PREFILL, prefiller_key),
            api,
            req_data,
            request_id,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
        )
        response_json = response.json()
        # Validate before complete_prefill commits affinity. This keeps malformed
        # 2xx responses on the same rollback path as transport/HTTP failures.
        if not isinstance(response_json, dict):
            raise ValueError("Prefill response must be a JSON object")
    except Exception:
        await _abort_prefill_selection(
            runtime,
            prefiller_key,
            prefiller_compute_score,
            prefiller_kv_score,
            is_initial_request=is_initial_request,
        )
        raise

    reusable_prefix_tokens = extract_reusable_prefix_tokens(response_json)
    allow_affinity = True
    if getattr(args, "enable_reusable_prefix_affinity_gate", False):
        if reusable_prefix_tokens is None:
            # Demoted to DEBUG because every request would log this when Prefillers
            # do not expose created_cache_tokens (e.g. the /v1/completions path).
            logger.debug(
                "Reusable-prefix affinity gate enabled but Prefill response is missing "
                "complete prompt_tokens_details (reusable_prefix_tokens=None); falling back to "
                "optimistic bind for request %s session=%s prefix=%s",
                request_id,
                session_key,
                prefix_key,
            )
        else:
            allow_affinity = reusable_prefix_tokens > 0

    try:
        await runtime.schedule(
            "complete_prefill",
            prefiller_key,
            prefiller_compute_score,
            session_key,
            prefix_key,
            prefiller["route_source"],
            allow_affinity,
        )
    except Exception:
        await _abort_prefill_selection(
            runtime,
            prefiller_key,
            prefiller_compute_score,
            prefiller_kv_score,
            is_initial_request=is_initial_request,
        )
        raise

    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params
    prefiller_cached_tokens = extract_cached_tokens(response_json)
    usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    route_source = prefiller.get("route_source", "heap")
    if route_source == "session":
        outcome_affinity_key = session_key
    elif route_source == "prefix":
        outcome_affinity_key = prefix_key
    else:
        outcome_affinity_key = None
    try:
        await runtime.schedule(
            "observe_affinity_outcome",
            route_source,
            outcome_affinity_key,
            prefiller_cached_tokens,
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            prefiller_key,
        )
    except Exception:
        logger.debug("Failed to record affinity outcome for request %s", request_id, exc_info=True)

    try:
        decoder = await runtime.schedule("pick_decoder", decoder_score)
    except Exception:
        await _abort_after_prefill(
            runtime,
            prefiller_key,
            prefiller_kv_score,
            is_initial_request=is_initial_request,
        )
        raise

    prefiller_client = await runtime.get_client(ServerRole.PREFILL, prefiller_key)
    decoder_client = await runtime.get_client(ServerRole.DECODE, decoder["key"])
    if route_source == "session":
        affinity_key = session_key
    elif route_source == "prefix":
        affinity_key = prefix_key
    else:
        affinity_key = None
    extra_route_fields: dict[str, Any] = {}
    if getattr(args, "enable_reusable_prefix_affinity_gate", False):
        extra_route_fields["reusable_prefix_tokens"] = reusable_prefix_tokens
        extra_route_fields["affinity_committed"] = allow_affinity
    logger.info(
        "Routed request %s prefiller=%s decoder=%s source=%s affinity_key=%s%s",
        request_id,
        prefiller_client.base_url,
        decoder_client.base_url,
        route_source,
        affinity_key,
        (" " + " ".join(f"{k}={v}" for k, v in extra_route_fields.items())) if extra_route_fields else "",
    )
    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_kv_score,
        decoder_key=decoder["key"],
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
        prefiller_cached_tokens=prefiller_cached_tokens,
    )


async def reassign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    previous_instance: InstanceInfo,
    *,
    session_key: str | None = None,
    prefix_key: str | None = None,
) -> InstanceInfo:
    runtime = get_runtime()
    await runtime.schedule("release_prefill_kv", previous_instance.prefiller_key, previous_instance.prefiller_score)
    await runtime.schedule("release_decoder", previous_instance.decoder_key, previous_instance.decoder_score)
    return await assign_instances(
        api,
        req_data,
        request_length,
        is_initial_request=False,
        session_key=session_key,
        prefix_key=prefix_key,
    )


async def handle_completions_impl(api: str, request: Request):
    runtime = get_runtime()
    args = get_global_args()
    request_released = False
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        session_key, prefix_key = extract_affinity_keys(
            request.headers,
            req_data,
            enabled=args.enable_kv_cache_aware_routing,
            prefix_chars=args.prefix_hash_chars,
        )
        instance_info = await assign_instances(
            api,
            req_data,
            request_length,
            is_initial_request=True,
            session_key=session_key,
            prefix_key=prefix_key,
        )
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            nonlocal request_released
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens

            async def release_prefill_kv_once() -> None:
                nonlocal released_kv
                if not released_kv:
                    await runtime.schedule(
                        "release_prefill_kv", instance_info.prefiller_key, instance_info.prefiller_score
                    )
                    released_kv = True

            try:
                while retry:
                    retry = False
                    decoder_client = await runtime.get_client(ServerRole.DECODE, instance_info.decoder_key)
                    async for chunk in stream_service_response_with_retry(
                        decoder_client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=args.max_retries,
                        base_delay=args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            await release_prefill_kv_once()
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug("Skipping chunk: %s", chunk)
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        is_sse = chunk_str.startswith("data: ")
                        if is_sse:
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            logger.debug("Skipping chunk: %s", chunk_str)
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            if update_cached_tokens_in_chunk(chunk_json, reported_prefiller_cached_tokens):
                                chunk = encode_response_chunk(chunk_json, is_sse)
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens", 0))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await reassign_instances(
                                api,
                                req_data,
                                tmp_request_length,
                                instance_info,
                                session_key=session_key,
                                prefix_key=prefix_key,
                            )
                            released_kv = False
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = encode_response_chunk(chunk_json, is_sse)
                        yield chunk
            except asyncio.CancelledError:
                logger.warning(
                    "Streaming from decoder %s:%s was cancelled; releasing request %s resources",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    instance_info.request_id,
                )
                raise
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s while handling request %s; releasing prefiller KV",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    exc,
                    instance_info.request_id,
                )
            finally:
                await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv)
                released_kv = True
                request_released = True

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if not request_released and "instance_info" in locals():
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
        raise


async def adjust_instances_impl(adjust_mode: str, request: Request):
    req_data = await request.json()
    instance_type = req_data.get("type", "")
    instances = req_data.get("instances", [])
    if isinstance(instances, str):
        instances = [instances]
    parsed_instances = parse_server_addresses(instances)
    all_msg = f"{adjust_mode} {instance_type} instances: {[f'{host}:{port}' for host, port in parsed_instances]}."

    try:
        role = ServerRole(instance_type)
    except ValueError:
        return {
            "error": (
                f"Instance type {instance_type!r} is not supported. "
                f"Only '{ServerRole.PREFILL.value}' and '{ServerRole.DECODE.value}' are allowed."
            )
        }

    scheduler = get_runtime().scheduler

    if adjust_mode == "add":
        waiting_nodes = scheduler.add_instances(role, parsed_instances)
        if waiting_nodes:
            all_msg = f"Instances {waiting_nodes} are waiting to be added."
    elif adjust_mode == "remove":
        need_waiting = scheduler.remove_instances(role, parsed_instances)
        if need_waiting:
            all_msg = (
                f"Instances {[f'{host}:{port}' for host, port in parsed_instances]} "
                "are isolated and waiting to be removed."
            )

    snapshot = scheduler.get_snapshot()
    return {
        "message": all_msg,
        "current_prefill_instances": [f"{server['host']}:{server['port']}" for server in snapshot["prefill_instances"]],
        "current_decode_instances": [f"{server['host']}:{server['port']}" for server in snapshot["decode_instances"]],
    }


def parse_server_addresses(instances: list[str]) -> list[tuple[str, int]]:
    return [(host, int(port)) for host, port in (instance.split(":") for instance in instances)]


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await handle_completions_impl("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await handle_completions_impl("/chat/completions", request)


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    params = dict(request.query_params)
    runtime = get_runtime()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()
    # Prefillers own the prefix KV in this PD topology; Decoders run with
    # --no-enable-prefix-caching, so fan-out to decode is unnecessary.
    failures: list[str] = []
    backends: list[dict[str, Any]] = []
    for server in snapshot["prefill_instances"]:
        base_url = build_server_url(server["host"], server["port"])
        started = time.monotonic()
        result: dict[str, Any] = {"target": base_url}
        try:
            client = await runtime.get_client(ServerRole.PREFILL, server_key(server["host"], server["port"]))
            resp = await client.post(f"{base_url}/reset_prefix_cache", params=params)
            result["status_code"] = resp.status_code
            try:
                body: Any = resp.json()
            except ValueError:
                body = resp.text[:2048]
            result["body"] = body
            resp.raise_for_status()
            explicit_failure = body is False or (
                isinstance(body, dict)
                and (
                    body.get("success") is False
                    or body.get("reset") is False
                    or str(body.get("status", "")).lower() in {"failed", "failure", "error"}
                )
            )
            if explicit_failure:
                raise RuntimeError(f"backend reported reset failure: {body!r}")
        except Exception as e:
            logger.error("reset_prefix_cache failed for %s: %s", base_url, e)
            failures.append(base_url)
            result["error"] = repr(e)
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            backends.append(result)
    # A backend reset invalidates the ownership assumption behind both LRUs.
    # Clear conservatively even on partial reset failure: losing affinity only
    # falls back to load balancing, while retaining it can route to stale KV.
    runtime.scheduler.clear_affinity_caches()
    if failures:
        return JSONResponse(
            status_code=500,
            content={"success": False, "failed": failures, "backends": backends},
        )
    return JSONResponse(status_code=200, content={"success": True, "backends": backends})


@app.get("/healthcheck")
async def healthcheck():
    return get_runtime().scheduler.healthcheck()


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await adjust_instances_impl("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await adjust_instances_impl("remove", request)


if __name__ == "__main__":
    global_args = parse_args()
    setup_logging(global_args.log_level)
    bootstrap_parent_process(global_args)
    import uvicorn

    module_name = Path(__file__).stem
    try:
        uvicorn.run(
            f"{module_name}:create_app",
            host=global_args.host,
            port=global_args.port,
            workers=global_args.workers,
            factory=True,
            app_dir=str(Path(__file__).resolve().parent),
        )
    finally:
        cleanup_manager_config(global_args.port)
