# Qwen3-32B 4P4D TP2 on Ascend 910B2

This deployment starts four independent TP2 Prefill instances on one 8-NPU
node, four independent TP2 Decode instances on another 8-NPU node, and one
P/D proxy. It is a controlled experiment topology, not a generic production
chart.

## Fixed topology

| Item | Value |
| --- | --- |
| Model | Qwen3-32B, unquantized |
| Prefill | 4 independent instances, TP2 |
| Decode | 4 independent instances, TP2 |
| NPU assignment | `0,1`, `2,3`, `4,5`, `6,7` |
| Prefill API ports | `7100-7103` |
| Decode API ports | `7200-7203` |
| Proxy port | `8000` |
| KV connector | MooncakeConnectorV1, Ascend Direct |
| KV block size | 128 |
| Prefix cache | enabled on Prefill, disabled on Decode |
| Proxy workers | 1 |

The Pods are privileged because the supplied vllm-ascend runtime requires
Ascend device and driver access. Review every hostPath and security setting
before using this deployment outside an isolated test cluster.

## Prerequisites

- Two Kubernetes nodes with eight Ascend 910B2 NPUs each.
- Ascend device plugin and a known extended resource name.
- Cross-node HCCN connectivity for Ascend Direct.
- A compatible vllm-ascend image containing Mooncake, FastAPI, HTTPX and
  Uvicorn.
- The same model host path on both NPU nodes.
- `kubectl`, Bash and `sed` on the deployment machine.

Required driver paths and networking assumptions are visible directly in
`prefill.yaml`, `decode.yaml` and `launch_role.sh`. Validate them against the
target cluster rather than assuming another Ascend installation is identical.

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

Important optional settings:

| Variable | Default |
| --- | --- |
| `MAX_MODEL_LEN` | `8192` |
| `PREFILL_MAX_BATCHED_TOKENS` | `4096` |
| `DECODE_MAX_BATCHED_TOKENS` | `512` |
| `PREFILL_MAX_NUM_SEQS` | `32` |
| `DECODE_MAX_NUM_SEQS` | `64` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `SESSION_LRU_SIZE` | `4096` |
| `PREFIX_HASH_CHARS` | `1024` |
| `PREFIX_LRU_SIZE` | `1024` |

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

## Cleanup

```bash
bash deploy.sh cleanup-proxy
bash deploy.sh cleanup-prefill
bash deploy.sh cleanup-decode
# Or remove all resources created by this deployment:
bash deploy.sh cleanup-all
```

The Namespace itself is intentionally retained.

