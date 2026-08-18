# Qwen3-32B 4P4D TP2 on Ascend 910B2

This deployment starts four Prefill Pods with one independent TP2 vLLM process
per Pod on one 8-NPU node, four equivalent Decode Pods on another 8-NPU node,
and one P/D proxy. It is a controlled experiment topology, not a generic
production chart.

## Fixed topology

| Item | Value |
| --- | --- |
| Model | Qwen3-32B, unquantized |
| Prefill | 4 Pods, one independent process per Pod, TP2 |
| Decode | 4 Pods, one independent process per Pod, TP2 |
| NPU assignment | 2 devices per Pod, allocated by the Ascend device plugin |
| Prefill API port | `7100` in each Pod |
| Decode API port | `7200` in each Pod |
| Proxy port | `8000` |
| KV connector | MooncakeConnectorV1, Ascend Direct |
| KV block size | 128 |
| Prefix cache | enabled on Prefill, disabled on Decode |
| Proxy workers | 1 |

The vLLM containers are not privileged. The Ascend device plugin must inject
exactly two `/dev/davinciN` devices per Pod plus `/dev/davinci_manager`,
`/dev/devmm_svm` and `/dev/hisi_hdc`. The launcher validates this before it
starts vLLM and does not override the runtime's device visibility mapping.
Review every hostPath and security setting before using this deployment outside
an isolated test cluster.

## Prerequisites

- Two Kubernetes nodes with eight Ascend 910B2 NPUs each.
- Ascend device plugin and a known extended resource name.
- Ascend container runtime/hook configured so device-plugin allocations are
  materialized as device nodes in non-privileged workload containers.
- Cross-node HCCN connectivity for Ascend Direct.
- A compatible vllm-ascend image containing Mooncake, FastAPI, HTTPX and
  Uvicorn.
- The same model host path on both NPU nodes.
- `kubectl`, Bash and `sed` on the deployment machine.

Required driver paths and networking assumptions are visible directly in
`prefill.yaml`, `decode.yaml` and `launch_role.sh`. Validate them against the
target cluster rather than assuming another Ascend installation is identical.

The Prefill and Decode Services are headless. The proxy connects to stable
StatefulSet DNS names (`pd-prefill-0.pd-prefill` through `pd-prefill-3.pd-prefill`
and the corresponding Decode names), so each process remains an independently
tracked backend.

## Configuration

```bash
export NAMESPACE='qwen-pd'
export VLLM_IMAGE='your-tested-vllm-ascend-image'
export PREFILL_NODE='your-prefill-node'
export DECODE_NODE='your-decode-node'
export MODEL_HOST_PATH='/models'
export MODEL_PATH='/models/Qwen/Qwen3-32B'
export SERVED_MODEL_NAME='qwen3-32b'
export NPU_RESOURCE='huawei.com/Ascend910'
export NIC_NAME='eth0'
```

Optional settings:

| Variable | Default |
| --- | --- |
| `MAX_MODEL_LEN` | `32768` |
| `PREFILL_MAX_BATCHED_TOKENS` | `4096` |
| `DECODE_MAX_BATCHED_TOKENS` | `512` |
| `PREFILL_MAX_NUM_SEQS` | `32` |
| `DECODE_MAX_NUM_SEQS` | `64` |
| `PROXY_DECODER_COUNT` | `4` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `SESSION_LRU_SIZE` | `4096` |
| `PREFIX_HASH_CHARS` | `1024` |
| `PREFIX_LRU_SIZE` | `1024` |
| `PREFILL_ACTIVE_TOKEN_WEIGHT` | `1.0` |
| `AFFINITY_MISS_UNBIND_THRESHOLD` | `0` (guard off) |
| `AFFINITY_CACHE_DISCOUNT_ALPHA` | `0` (discount off) |

The default uses Qwen3-32B's native 32K context window and does not require
RoPE scaling. Prefill remains chunked at 4,096 tokens by default, so longer
prompts do not require a matching increase in `PREFILL_MAX_BATCHED_TOKENS`.

`PROXY_DECODER_COUNT` controls how many of the four already-running Decoder
Pods are included in the Proxy backend list. Values 1 through 4 select the
stable StatefulSet backends starting at `pd-decode-0`; changing it redeploys
only the Proxy and does not change or restart the Decoder engines. Use it for
pressure experiments that need fewer visible Decoders. `DECODE_MAX_NUM_SEQS`
is a vLLM engine startup argument, so changing it requires a Decoder rollout.

## Select an experiment group

Exact upstream baseline:

```bash
export PROXY_VARIANT=baseline
export KV_AWARE_ROUTING=false
```

Candidate with affinity disabled:

```bash
export PROXY_VARIANT=candidate
export KV_AWARE_ROUTING=false
```

Candidate with affinity enabled:

```bash
export PROXY_VARIANT=candidate
export KV_AWARE_ROUTING=true
```

The deploy script rejects `PROXY_VARIANT=baseline` together with
`KV_AWARE_ROUTING=true`, because the upstream script does not recognize the new
arguments.

## Deploy

```bash
bash deploy.sh all
```

Or deploy one role at a time:

```bash
bash deploy.sh prefill
bash deploy.sh decode
bash deploy.sh proxy
```

`deploy.sh proxy` packages the selected proxy file into a ConfigMap. Neither
proxy file needs to be present on the NPU nodes.

Inspect status and access the proxy locally:

```bash
bash deploy.sh status
kubectl -n "${NAMESPACE:-qwen-pd}" port-forward service/pd-proxy 8000:8000
```

The reproducible A/B/C harness does not require this port-forward. It creates a
temporary CPU-only Pod from `benchmark-pod.yaml`, uses in-cluster Service DNS
for tokenizer and benchmark traffic, and copies results back when complete.

## Cleanup

```bash
bash deploy.sh cleanup-proxy
bash deploy.sh cleanup-prefill
bash deploy.sh cleanup-decode
bash deploy.sh cleanup-benchmark
# Or remove all resources created by this deployment (including leftover
# pd-benchmark Pods/ConfigMaps):
bash deploy.sh cleanup-all
```

The Namespace itself is intentionally retained.
