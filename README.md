# vLLM Ascend KV-aware P/D proxy

A drop-in KV-cache-aware routing extension for
[vllm-ascend](https://github.com/vllm-project/vllm-ascend) disaggregated
Prefill/Decode deployments: Decoders stay load-balanced while Prefillers gain
session and prefix locality.

## Why this project

Multi-turn chat and shared-prefix traffic only benefit from prefix caching when
follow-up requests land on the Prefiller that already holds the KV. A pure
load-balancing proxy scatters that locality.

Stacks such as [llm-d](https://github.com/llm-d/llm-d) and Higress-based
KV-aware gateways use additional components. This project adds two affinity
policies directly to the Ascend P/D proxy, without a separate control plane or
sidecars. Existing **vllm-ascend** deployments can keep their Prefillers and
Decoders, replace the proxy script, and enable
`--enable-kv-cache-aware-routing`.

## Benchmark highlights

KV-aware routing turns cache locality into less Prefill work and lower latency.
In controlled A/B tests, enabling affinity on the same candidate proxy produced
the following results (mean of six paired repetitions; lower is better for
tokens and latency):

| Workload | Metric | Affinity off | Affinity on | Change (95% CI) |
| --- | --- | ---: | ---: | ---: |
| **Multi-turn sessions** | Cached-token ratio | 42.5% | **89.4%** | **+46.8 pp** (+44.9 to +49.0 pp) |
| | Computed prompt tokens / warm request | 1,605 | **298** | **-81.4%** (-82.1% to -80.8%) |
| | Prefill time, mean | 581 ms | **243 ms** | **-58.2%** (-59.6% to -56.7%) |
| | TTFT p95 | 2,532 ms | **1,266 ms** | **-49.9%** (-52.6% to -47.2%) |
| | End-to-end latency p95 | 4,804 ms | **3,548 ms** | **-26.1%** (-27.5% to -24.8%) |
| | Request throughput | 4.99 req/s | **5.88 req/s** | **+17.8%** (+16.2% to +19.2%) |
| **Shared prefixes** | Cached-token ratio | 85.1% | **97.3%** | **+12.1 pp** (+12.0 to +12.3 pp) |
| | Computed prompt tokens / warm request | 345 | **63** | **-81.7%** (-81.8% to -81.5%) |
| | Prefill time, mean | 171 ms | **91 ms** | **-46.9%** (-47.1% to -46.6%) |
| | TTFT p95 | 970 ms | **730 ms** | **-24.8%** (-25.7% to -24.1%) |
| | End-to-end latency p95 | 3,370 ms | **3,258 ms** | **-3.4%** (-4.9% to -1.7%) |

The session workload used 60 five-turn conversations with a 2,048-token system
prompt. The shared-prefix workload used 32 independently reusable 2,048-token
prefixes across a 2/5/10/20 QPS Poisson ramp. Both ran on Qwen3-32B with four
TP2 Prefillers and four TP2 Decoders across two 8× Ascend 910B2 nodes. Every
repetition reused an identical workload across groups, changed the workload
seed, reset and verified Prefiller caches, and rotated through all six A/B/C
execution orders. All requests succeeded.

These results apply to the workloads and topology described above. The
[benchmark methodology](docs/benchmark-methodology.md),
[reproduction guide](benchmarks/README.md), and
[controlled deployment](deploy/kubernetes/qwen3-32b-4p4d-tp2/README.md) describe
the experiment and how to rerun it.

## Two Prefiller affinity modes

With KV-aware routing enabled, Prefiller selection is:

```text
stable session key  →  text prefix hash  →  existing load heap
```

| Mode | How it binds | Best for |
| --- | --- | --- |
| **Session affinity** | Explicit session id (headers first, then body) | Multi-turn conversations |
| **Prefix-hash affinity** | Hash of a canonical text prefix | Cross-session shared system / prompt prefixes |

Session identifiers are matched first-match-wins. The proxy recognizes native
identifiers from OpenCode, Claude Code, Codex, Pi, Cline, Roo Code, and Gemini
CLI in addition to generic `X-Session-ID` and JSON fields. Codex `Thread-ID`
wins over its broader `Session-ID`; Claude Code subagents use a composite of
session and agent IDs. Equivalent generic aliases normalize to the same bounded
digest, so changing from a header to a body field does not lose an existing
binding or retain the original client ID in shared scheduler memory. See the
[compatibility matrix](docs/design.md#client-compatibility) for exact fields and
precedence.

Prefix hashing covers string `prompt` values and chat messages with plain-string
roles and contents. Tool calls, structured content, and multimodal requests skip
prefix affinity and fall back to load-based Prefiller routing. Decoder selection
is unchanged.

Bindings commit only after a successful Prefill. Failed Prefill rolls back
reservations; draining, removed, or prefix-cache-reset Prefillers invalidate
their bindings. Details: [docs/design.md](docs/design.md) and
[docs/limitations.md](docs/limitations.md).

## Quick start

Drop-in for vllm-ascend's
[`load_balance_proxy_server_example.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)
— same file name, upstream-compatible CLI with optional KV-aware flags, and the
same P→D routing by default. The upstream
[PD guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
applies verbatim; only KV-aware Prefiller selection is layered on top.

Requirements: Python 3.10+, at least one Prefiller and one Decoder, and the
proxy dependencies (FastAPI, HTTPX, Uvicorn — already present in the usual
vllm-ascend image). Prefillers must run with prefix caching enabled (for example,
`--enable-prefix-caching`) to realize the cache-hit and latency benefits; routing
affinity alone only preserves locality.

Run it directly from this clone (KV-aware routing on):

```bash
git clone https://github.com/mashuiping/vllm-ascend-kv-aware-proxy.git
cd vllm-ascend-kv-aware-proxy

python load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8000 --workers 1 \
  --prefiller-hosts <P_IP1> <P_IP2> --prefiller-ports <P_PORT1> <P_PORT2> \
  --decoder-hosts   <D_IP1> <D_IP2> --decoder-ports   <D_PORT1> <D_PORT2> \
  --enable-kv-cache-aware-routing \
  --session-lru-size 4096 --prefix-hash-chars 1024 --prefix-lru-size 1024
```

Or swap the upstream script in place so existing runbooks keep working (the
proxy behaves like the upstream one — drop the four `--enable-...` /
`--*-lru-size` / `--prefix-hash-chars` flags for the unbumped defaults):

```bash
cp /vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py{,.bak}
cp load_balance_proxy_server_example.py \
   /vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

Send a request through the proxy:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: demo-session-1' \
  -d '{
    "model": "qwen3-32b",
    "messages": [{"role": "user", "content": "Explain KV cache locality."}],
    "stream": true,
    "stream_options": {"include_usage": true},
    "max_tokens": 32
  }'
```

Omit `--enable-kv-cache-aware-routing` to keep the candidate proxy with affinity
disabled.

Prefix cache hits generally require the shared prompt to cover at least one KV
block. On Ascend the minimum `block_size` is typically 128 tokens, but the exact
value depends on the model architecture and vLLM configuration—short demo
prompts like the curl above may not produce a hit.

## Configuration

| Flag | Default | Role |
| --- | --- | --- |
| `--enable-kv-cache-aware-routing` | off | Turn on session + prefix Prefiller affinity |
| `--session-lru-size` | `4096` | Session→Prefiller LRU size (`0` disables session affinity) |
| `--prefix-hash-chars` | `1024` | Characters hashed for prefix affinity (`0` disables it) |
| `--prefix-lru-size` | `1024` | Prefix→Prefiller LRU size (`0` disables prefix affinity) |
| `--enable-reusable-prefix-affinity-gate` | off | Commit bindings only when Prefill reports reusable prefix tokens |
| `--affinity-miss-unbind-threshold` | `0` (off) | Unbind after N consecutive `cached_tokens == 0` outcomes on affinity hits |
| `--affinity-cache-discount-alpha` | `0` (off) | EMA smoothing for per-binding cache hit ratio; discounts affinity-hit compute reservations toward real prefill work (range `[0, 1]`) |
| `--workers` | `1` | Uvicorn workers; affinity state is shared via a parent-bootstrapped scheduler |

The affinity guards are independent opt-ins layered on KV-aware routing. The
miss-unbind threshold drops bindings whose KV has been evicted. The cache
discount makes compute reservations reflect the work left after cache hits, so
the heap does not overestimate load on a Prefiller serving cached prompts. An
overload escape valve (`--affinity-overload-factor`) existed through tag
`0.2.0` and was removed. In A/B/C experiments, instantaneous priority
comparison treated Poisson bursts as backlog. Every escape recomputed a large
prefix on a cold node, increasing TTFT p95 by up to 15%, while none of the runs
produced a sustained Prefill backlog. The
[design notes](docs/design.md#affinity-robustness-guards) cover these guards in
more detail.

`--enable-reusable-prefix-affinity-gate` commits session/prefix bindings only when
Prefill reports reusable prefix tokens
(`cached_tokens + created_cache_tokens > 0`). That skips sticky binding for short
prompts that leave no reusable KV, so a one-off short request does not pin its
session to a Prefiller. Prefillers must expose complete `prompt_tokens_details`
(enable **`--enable-prompt-tokens-details`** on vLLM, with prefix caching).
Without those fields the proxy logs a warning and falls back to optimistic bind.
See [docs/design.md](docs/design.md).

## License and provenance

Apache License 2.0. See [LICENSE](LICENSE).

The unmodified upstream baseline is pinned to vllm-ascend tag `v0.23.0rc1` under
[`baseline/`](baseline/README.md). This repository is an unofficial experimental
fork published so KV-aware Prefiller routing can be deployed and measured ahead
of upstream review.
