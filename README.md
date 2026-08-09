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

Session identifiers are matched first-match-wins:

1. `X-Session-ID`
2. `X-Claude-Code-Session-ID`
3. top-level JSON `session_id`
4. JSON `session_params.session_id`

Prefix hashing covers string `prompt` values and chat messages with plain-string
roles and contents. Tool calls, structured content, and multimodal requests skip
prefix affinity and fall back to load-based Prefiller routing. Decoder selection
is unchanged.

Bindings commit only after a successful Prefill. Failed Prefill rolls back
reservations; draining, removed, or prefix-cache-reset Prefillers invalidate
their bindings. Details: [docs/design.md](docs/design.md) and
[docs/limitations.md](docs/limitations.md).

## Quick start

Requirements: Python 3.10+, at least one Prefiller and one Decoder, and the
proxy dependencies (FastAPI, HTTPX, Uvicorn—already present in the usual
vllm-ascend image).


```bash
python load_balance_proxy_server_example.py \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --prefiller-hosts 127.0.0.1 127.0.0.1 \
  --prefiller-ports 7100 7101 \
  --decoder-hosts 127.0.0.1 127.0.0.1 \
  --decoder-ports 7200 7201 \
  --enable-kv-cache-aware-routing \
  --session-lru-size 4096 \
  --prefix-hash-chars 1024 \
  --prefix-lru-size 1024
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

## Configuration

| Flag | Default | Role |
| --- | --- | --- |
| `--enable-kv-cache-aware-routing` | off | Turn on session + prefix Prefiller affinity |
| `--session-lru-size` | `4096` | Session→Prefiller LRU size (`0` disables session affinity) |
| `--prefix-hash-chars` | `1024` | Characters hashed for prefix affinity (`0` disables it) |
| `--prefix-lru-size` | `1024` | Prefix→Prefiller LRU size (`0` disables prefix affinity) |
| `--enable-reusable-prefix-affinity-gate` | off | Commit bindings only when Prefill reports reusable prefix tokens |
| `--workers` | `1` | Uvicorn workers; affinity state is shared via a parent-bootstrapped scheduler |

`--enable-reusable-prefix-affinity-gate` commits session/prefix bindings only when
Prefill reports reusable prefix tokens
(`cached_tokens + created_cache_tokens > 0`). That skips sticky binding for short
prompts that leave no reusable KV, so a one-off short request does not pin its
session to a Prefiller. Prefillers must expose complete `prompt_tokens_details`
(enable **`--enable-prompt-tokens-details`** on vLLM, with prefix caching).
Without those fields the proxy logs a warning and falls back to optimistic bind.
See [docs/design.md](docs/design.md).

## Results

Benchmark charts and published Ascend A/B/C numbers will be added here.

To reproduce measurements locally, see
[benchmarks/README.md](benchmarks/README.md) and
[docs/benchmark-methodology.md](docs/benchmark-methodology.md).
For a sample Ascend P/D Kubernetes topology, see
[deploy/kubernetes/qwen3-32b-4p4d-tp2/README.md](deploy/kubernetes/qwen3-32b-4p4d-tp2/README.md).

## License and provenance

Apache License 2.0. See [LICENSE](LICENSE).

The unmodified upstream baseline is pinned to vllm-ascend tag `v0.23.0rc1` under
[`baseline/`](baseline/README.md). This repository is an unofficial experimental
fork published so KV-aware Prefiller routing can be deployed and measured ahead
of upstream review.
