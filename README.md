# vLLM Ascend KV-aware P/D proxy

A drop-in extension of the [vllm-ascend](https://github.com/vllm-project/vllm-ascend)
disaggregated Prefill/Decode load-balancing proxy that adds **KV-cache-aware
Prefiller routing**. Decoder selection stays load-balanced; only Prefillers use
session and prefix affinity so reusable KV stays local.

Motivated by [vllm-project/vllm-ascend#12196](https://github.com/vllm-project/vllm-ascend/issues/12196).

## Why this project

Multi-turn chat and shared-prefix traffic only benefit from prefix caching when
follow-up requests land on the Prefiller that already holds the KV. A pure
load-balancing proxy scatters that locality.

Compared with heavier stacks such as [llm-d](https://github.com/llm-d/llm-d) or
Higress-based KV-aware gateways, this project keeps the ops surface small: no
separate control plane, no extra sidecars—just the familiar Ascend P/D proxy
plus two affinity policies. For **vllm-ascend** deployments it is intended to be
out of the box: run Prefillers and Decoders as usual, point this proxy at them,
and enable `--enable-kv-cache-aware-routing`.

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
— same file name, same CLI, same P→D routing by default. The upstream
[PD guide](https://docs.vllm.ai/projects/ascend/zh-cn/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
applies verbatim; only KV-aware Prefiller selection is layered on top.

Requirements: Python 3.10+, at least one Prefiller and one Decoder, and the
proxy dependencies (FastAPI, HTTPX, Uvicorn — already present in the usual
vllm-ascend image).

Run it directly from this clone (KV-aware routing on):

```bash
git clone -b 0.2.0 https://github.com/mashuiping/vllm-ascend-kv-aware-proxy.git
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

The affinity guards are independent opt-ins layered on KV-aware routing: the
miss-unbind threshold drops bindings whose KV has been evicted, and the cache
discount keeps load accounting honest for highly-cached prompts so heap
routing is not biased against exactly the nodes affinity is helping. An
overload escape valve (`--affinity-overload-factor`) existed through tag
`0.2.0` and was removed: A/B/C experiments showed instantaneous priority
comparison cannot tell Poisson bursts from real backlog, so it only produced
false-positive escapes that recomputed large prefixes on cold nodes (up to
+15% TTFT p95) while a genuine prefill backlog never materialised — affinity
itself keeps prefill cheap enough that the hot node does not saturate.
See [docs/design.md](docs/design.md#affinity-robustness-guards).

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
