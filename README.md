# vLLM Ascend KV-aware P/D proxy

> [!IMPORTANT]
> This is an unofficial experimental project derived from vllm-ascend. It is
> published to make KV-cache-aware Prefiller routing deployable and measurable
> before the corresponding upstream change is reviewed.

This repository provides a minimal KV-cache-aware extension of vllm-ascend's
disaggregated Prefill/Decode load-balancing proxy, unit tests, a reproducible
Qwen3 deployment for Ascend 910B2, and an A/B/C benchmark harness.

The implementation is motivated by
[vllm-ascend issue #12196](https://github.com/vllm-project/vllm-ascend/issues/12196).
It keeps Decoder selection load-balanced and applies affinity only to
Prefillers, where reusable prefix KV resides.

## Status

- Candidate implementation: available and unit tested.
- Kubernetes deployment: 4 Prefillers + 4 Decoders, TP2, Qwen3-32B.
- Benchmark harness: available.
- Published Ascend measurements: pending.
- Upstream merge: not yet available.

Do not treat the current defaults as production tuning recommendations until
the A/B/C results have been published and reproduced.

## What changes

When `--enable-kv-cache-aware-routing` is present, the Prefiller route order is:

1. a stable session binding;
2. a text-only prefix-hash binding;
3. the existing load-tracking heap.

Supported session identifiers, in priority order:

1. `X-Session-ID` header;
2. `X-Claude-Code-Session-ID` header;
3. top-level JSON `session_id`;
4. JSON `session_params.session_id`.

Headers win over body fields. User, tenant, request and trace identifiers are
intentionally excluded because they are either too broad or change on every
request.

Prefix hashing supports string `prompt` values and chat messages whose roles
and contents are plain strings. Tool calls, tool definitions, structured
content and multimodal requests safely fall back to normal load-based routing
because the proxy cannot reproduce their model-specific token prefix exactly.

The candidate also restores the Prefiller `active_tokens` lifecycle that
existed before the shared-scheduler refactor: active Prefill compute and
longer-lived KV-transfer pressure are reserved together, then released at their
respective completion points.

See [docs/design.md](docs/design.md) and
[docs/limitations.md](docs/limitations.md) for details.

## Provenance

The unmodified upstream baseline is pinned to vllm-ascend commit:

```text
8f3fd59a203c3b35f29b6a77f459c9a78da7d5c0
```

The exact baseline is stored in `baseline/` so experiments do not silently
change when upstream `main` moves. See [UPSTREAM.md](UPSTREAM.md) for the source
path and checksums.

## Run unit tests

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check load_balance_proxy_server_example.py scripts tests
python -m ruff format --check load_balance_proxy_server_example.py scripts tests
```

The standalone proxy runtime needs FastAPI, HTTPX and Uvicorn. The supplied
vllm-ascend container image already includes the model-serving runtime used by
the Prefill and Decode backends.

## Run the proxy directly

Start at least one Prefiller and one Decoder, then run:

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

- `--enable-reusable-prefix-affinity-gate` (optional): skip affinity commit when
  the Prefill response shows no reusable prefix. See `docs/design.md`.

Example request:

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

Omit `--enable-kv-cache-aware-routing` to run the candidate with affinity
disabled.

## Kubernetes quick start

The first deployment is deliberately specific: two 8-NPU Ascend 910B2 nodes,
Qwen3-32B, four independent Prefill Pods, four independent Decode Pods, and
one TP2 process per Pod.

```bash
cd deploy/kubernetes/qwen3-32b-4p4d-tp2

export PREFILL_NODE='your-prefill-node'
export DECODE_NODE='your-decode-node'
export VLLM_IMAGE='your-tested-vllm-ascend-image'
export MODEL_HOST_PATH='/models'
export MODEL_PATH='/models/Qwen/Qwen3-32B'

# Candidate with KV-aware routing enabled.
export PROXY_VARIANT=candidate
export KV_AWARE_ROUTING=true
bash deploy.sh all
```

See the deployment-specific [README](deploy/kubernetes/qwen3-32b-4p4d-tp2/README.md)
before applying the non-privileged workload Pods to a cluster.

## Reproducible A/B/C comparison

Use the same P/D Pods and change only the proxy group:

| Group | `PROXY_VARIANT` | `KV_AWARE_ROUTING` | Purpose |
| --- | --- | --- | --- |
| A: `baseline` | `baseline` | `false` | Exact upstream behavior |
| B: `candidate-off` | `candidate` | `false` | Candidate and restored load accounting without affinity |
| C: `candidate-on` | `candidate` | `true` | KV-aware session/prefix affinity |

The recommended harness creates a temporary benchmark Pod in the model
namespace. The Pod talks directly to the Prefiller tokenizer and proxy Services,
so no port-forward is required. It generates one immutable, exact-token
workload for all three groups, performs system warm-up, resets backend prefix
caches, records cache-fill separately and then measures warm turns or QPS
stages. See [benchmarks/README.md](benchmarks/README.md) for workload details.

```bash
export PREFILL_NODE='your-prefill-node'
export VLLM_IMAGE='your-tested-vllm-ascend-image'
export MODEL='qwen3-32b'

bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

The completed workload, manifest, group results and comparison are copied from
the Pod to `results/runs/`. Set `KEEP_BENCHMARK_POD=true` to retain the Pod for
debugging. The old port-forward workflow remains available with
`BENCHMARK_EXECUTION_MODE=local` plus `BASE_URL` and `TOKENIZER_URL`.

For shared-prefix capacity and QPS-ladder testing, use
`benchmarks/profiles/shared-prefix-capacity.json`.

`run_experiment.sh` can still run one group. Set `WORKLOAD_FILE` to ensure it
uses the same frozen data as the other groups:

```bash
WORKLOAD_FILE=results/runs/<experiment>/workload.jsonl \
  bash scripts/run_experiment.sh candidate-on
```

Legacy on-the-fly generation remains available for debugging. Pass additional
benchmark arguments after the group name, for example:

```bash
bash scripts/run_experiment.sh candidate-on \
  --scenario shared-prefix \
  --sessions 128 \
  --turns 4 \
  --concurrency 32 \
  --prefill-metrics-url http://P0:7100/metrics \
  --prefill-metrics-url http://P1:7101/metrics
```

Raw runs go to `results/runs/` and are ignored by Git. Publish only reviewed,
redacted summaries under `results/published/`. The required metadata and
comparison rules are documented in
[docs/benchmark-methodology.md](docs/benchmark-methodology.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [UPSTREAM.md](UPSTREAM.md).
