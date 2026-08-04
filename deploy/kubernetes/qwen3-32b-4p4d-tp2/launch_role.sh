#!/usr/bin/env bash
set -Eeuo pipefail

: "${ROLE:?ROLE must be prefill or decode}"
: "${MODEL_PATH:?MODEL_PATH is required}"

case "${ROLE}" in
  prefill)
    KV_ROLE="kv_producer"
    PORT_START="${PORT_START:-7100}"
    MOONCAKE_KV_PORT="${MOONCAKE_KV_PORT:-30000}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    HCCL_BUFFSIZE_DEFAULT="256"
    ;;
  decode)
    KV_ROLE="kv_consumer"
    PORT_START="${PORT_START:-7200}"
    MOONCAKE_KV_PORT="${MOONCAKE_KV_PORT:-30200}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    HCCL_BUFFSIZE_DEFAULT="600"
    ;;
  *)
    echo "unsupported ROLE=${ROLE}; expected prefill or decode" >&2
    exit 2
    ;;
esac

DEVICE_COUNT="${DEVICE_COUNT:-2}"
INSTANCE_COUNT="${INSTANCE_COUNT:-1}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
NIC_NAME="${NIC_NAME:-eth0}"
POD_NAME="${POD_NAME:-${HOSTNAME:-${ROLE}-unknown}}"
ENGINE_ID="${ROLE}-${POD_NAME}"

for runtime_command in python vllm sed hostname awk; do
  if ! command -v "${runtime_command}" >/dev/null 2>&1; then
    echo "missing required runtime command: ${runtime_command}" >&2
    exit 2
  fi
done

LOCAL_IP="$(hostname -I | awk '{print $1}')"
if [[ ! "${LOCAL_IP}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "the first address returned by hostname -I is not an IPv4 address: ${LOCAL_IP:-<empty>}" >&2
  exit 2
fi

export HCCL_IF_IP="${LOCAL_IP}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_INTRA_ROCE_ENABLE="1"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-${HCCL_BUFFSIZE_DEFAULT}}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT="${VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:-480}"

# vLLM Ascend's Mooncake PD example installs the Ascend Direct transport here.
# Keep the image's existing library path after the required Mooncake/driver paths.
MOONCAKE_LIBRARY_DIR="/usr/local/lib64/python3.11/site-packages/mooncake"
export LD_LIBRARY_PATH="${MOONCAKE_LIBRARY_DIR}:/usr/local/lib:/usr/local/lib64:/usr/local/Ascend/driver/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if (( INSTANCE_COUNT * TP_SIZE != DEVICE_COUNT )); then
  echo "invalid topology: INSTANCE_COUNT(${INSTANCE_COUNT}) * TP_SIZE(${TP_SIZE}) must equal DEVICE_COUNT(${DEVICE_COUNT})" >&2
  exit 2
fi

# Kubernetes' Ascend device plugin must inject only the NPU device nodes
# allocated to this Pod. Do not override its mapping with host physical IDs.
shopt -s nullglob
device_nodes=(/dev/davinci[0-9]*)
shopt -u nullglob
if (( ${#device_nodes[@]} != DEVICE_COUNT )); then
  echo "expected ${DEVICE_COUNT} allocated NPU device nodes, found ${#device_nodes[@]}: ${device_nodes[*]:-<none>}" >&2
  exit 2
fi

for common_device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [[ ! -c "${common_device}" ]]; then
    echo "required Ascend device node was not injected: ${common_device}" >&2
    exit 2
  fi
done

echo "allocated NPU devices: ${device_nodes[*]}"
echo "device visibility: ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

MOONCAKE_CONFIG_PATH="/tmp/mooncake-${ROLE}.json"
export MOONCAKE_CONFIG_PATH
printf '{"local_hostname":"%s","device_name":"","protocol":"ascend"}\n' \
  "${LOCAL_IP}" >"${MOONCAKE_CONFIG_PATH}"

children=()
terminate_children() {
  if (( ${#children[@]} > 0 )); then
    kill "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
}
trap terminate_children EXIT INT TERM

if (( INSTANCE_COUNT != 1 )); then
  echo "INSTANCE_COUNT must be 1: launch one vLLM process per Pod" >&2
  exit 2
fi

http_port="${PORT_START}"
kv_port="${MOONCAKE_KV_PORT}"

kv_transfer_config="$(
    printf '%s' \
      "{\"kv_connector\":\"MooncakeConnectorV1\"," \
      "\"kv_role\":\"${KV_ROLE}\"," \
      "\"kv_port\":\"${kv_port}\"," \
      "\"engine_id\":\"${ENGINE_ID}\"," \
      "\"kv_connector_extra_config\":{" \
      "\"prefill\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}}," \
      "\"decode\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}}}}"
  )"

command=(
    vllm serve "${MODEL_PATH}"
    --host 0.0.0.0
    --port "${http_port}"
    --tensor-parallel-size "${TP_SIZE}"
    --seed 1024
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --block-size 128
    --trust-remote-code
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --enforce-eager
    --kv-transfer-config "${kv_transfer_config}"
  )

if [[ "${ROLE}" == "prefill" ]]; then
    command+=(
      --enable-chunked-prefill
      --enable-prefix-caching
      --enable-prompt-tokens-details
    )
  else
    command+=(--no-enable-prefix-caching)
  fi

if [[ -n "${REASONING_PARSER:-}" ]]; then
    command+=(--reasoning-parser "${REASONING_PARSER}")
  fi

echo "starting ${ROLE} pod=${POD_NAME} engine_id=${ENGINE_ID} http_port=${http_port} kv_port=${kv_port}"
(
  set -o pipefail
  "${command[@]}" 2>&1 | sed -u "s/^/[${POD_NAME}] /"
) &
children+=("$!")

python - "${PORT_START}" <<'PY'
import sys
import time
import urllib.request

port_start = int(sys.argv[1])
deadline = time.monotonic() + 3600
pending = {0}

while pending and time.monotonic() < deadline:
    for instance in list(pending):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port_start + instance}/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    pending.remove(instance)
                    print("instance 0 is ready", flush=True)
        except Exception:
            pass
    if pending:
        print("waiting for instance: 0", flush=True)
        time.sleep(5)

if pending:
    raise SystemExit("startup timed out; instance 0 is not ready")
PY

: >/tmp/pd-ready
echo "${ROLE} instance is ready"

wait -n "${children[@]}"
echo "a ${ROLE} instance exited; terminating the pod" >&2
exit 1
