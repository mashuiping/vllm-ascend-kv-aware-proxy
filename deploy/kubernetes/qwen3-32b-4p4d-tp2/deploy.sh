#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PROXY_VARIANT="${PROXY_VARIANT:-candidate}"
KV_AWARE_ROUTING="${KV_AWARE_ROUTING:-true}"
PROXY_DECODER_COUNT="${PROXY_DECODER_COUNT:-4}"
SESSION_LRU_SIZE="${SESSION_LRU_SIZE:-4096}"
PREFIX_HASH_CHARS="${PREFIX_HASH_CHARS:-1024}"
PREFIX_LRU_SIZE="${PREFIX_LRU_SIZE:-1024}"
PREFILL_ACTIVE_TOKEN_WEIGHT="${PREFILL_ACTIVE_TOKEN_WEIGHT:-1.0}"
AFFINITY_OVERLOAD_FACTOR="${AFFINITY_OVERLOAD_FACTOR:-0}"
AFFINITY_MISS_UNBIND_THRESHOLD="${AFFINITY_MISS_UNBIND_THRESHOLD:-0}"
AFFINITY_CACHE_DISCOUNT_ALPHA="${AFFINITY_CACHE_DISCOUNT_ALPHA:-0}"

case "${PROXY_VARIANT}" in
  candidate)
    DEFAULT_PROXY_SOURCE_PATH="${REPO_ROOT}/load_balance_proxy_server_example.py"
    ;;
  baseline)
    DEFAULT_PROXY_SOURCE_PATH="${REPO_ROOT}/baseline/load_balance_proxy_server_example.py"
    ;;
  *)
    echo "PROXY_VARIANT must be candidate or baseline: ${PROXY_VARIANT}" >&2
    exit 2
    ;;
esac
PROXY_SOURCE_PATH="${PROXY_SOURCE_PATH:-${DEFAULT_PROXY_SOURCE_PATH}}"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

NAMESPACE="${NAMESPACE:-qwen-pd}"
VLLM_IMAGE="${VLLM_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/models}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
NPU_RESOURCE="${NPU_RESOURCE:-huawei.com/Ascend910}"
NIC_NAME="${NIC_NAME:-eth0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PREFILL_MAX_BATCHED_TOKENS="${PREFILL_MAX_BATCHED_TOKENS:-4096}"
DECODE_MAX_BATCHED_TOKENS="${DECODE_MAX_BATCHED_TOKENS:-512}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-32}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
RESTART_TOKEN="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PROXY_CODE_SHA256="$(sha256_file "${PROXY_SOURCE_PATH}")"

usage() {
  cat <<'EOF'
Usage:
  deploy.sh prefill   # start 4 Prefill Pods, one TP2 process per Pod
  deploy.sh decode    # start 4 Decode Pods, one TP2 process per Pod
  deploy.sh proxy     # create/update the load-balancing proxy
  deploy.sh all       # Prefill -> Decode -> Proxy
  deploy.sh status
  deploy.sh cleanup-prefill
  deploy.sh cleanup-decode
  deploy.sh cleanup-proxy
  deploy.sh cleanup-benchmark
  deploy.sh cleanup-all

Required:
  PREFILL_NODE=<kubernetes node name>  (prefill/proxy/all)
  DECODE_NODE=<kubernetes node name>   (decode/all)

Supported settings and defaults:
  NAMESPACE=qwen-pd
  VLLM_IMAGE=quay.io/ascend/vllm-ascend:v0.18.0
  PROXY_SOURCE_PATH=<absolute path>/load_balance_proxy_server_example.py
  PROXY_VARIANT=candidate  # candidate or baseline
  KV_AWARE_ROUTING=true    # candidate only; true or false
  PROXY_DECODER_COUNT=4    # proxy-visible Decoder backends; 1 through 4
  SESSION_LRU_SIZE=4096
  PREFIX_HASH_CHARS=1024
  PREFIX_LRU_SIZE=1024
  MODEL_HOST_PATH=/models
  MODEL_PATH=/models/Qwen/Qwen3-32B
  SERVED_MODEL_NAME=qwen3-32b
  NIC_NAME=eth0
  NPU_RESOURCE=huawei.com/Ascend910
  MAX_MODEL_LEN=32768
  PREFILL_MAX_BATCHED_TOKENS=4096
  DECODE_MAX_BATCHED_TOKENS=512
  PREFILL_MAX_NUM_SEQS=32
  DECODE_MAX_NUM_SEQS=64
  GPU_MEMORY_UTILIZATION=0.90
  REASONING_PARSER=qwen3
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 2
  }
}

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required" >&2
    usage
    exit 2
  fi
}

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[&|\\]/\\&/g'
}

render() {
  local file="$1"
  sed \
    -e "s|__NAMESPACE__|$(escape_sed_replacement "${NAMESPACE}")|g" \
    -e "s|__PREFILL_NODE__|$(escape_sed_replacement "${PREFILL_NODE:-}")|g" \
    -e "s|__DECODE_NODE__|$(escape_sed_replacement "${DECODE_NODE:-}")|g" \
    -e "s|__VLLM_IMAGE__|$(escape_sed_replacement "${VLLM_IMAGE}")|g" \
    -e "s|__MODEL_HOST_PATH__|$(escape_sed_replacement "${MODEL_HOST_PATH}")|g" \
    -e "s|__MODEL_PATH__|$(escape_sed_replacement "${MODEL_PATH}")|g" \
    -e "s|__SERVED_MODEL_NAME__|$(escape_sed_replacement "${SERVED_MODEL_NAME}")|g" \
    -e "s|__NPU_RESOURCE__|$(escape_sed_replacement "${NPU_RESOURCE}")|g" \
    -e "s|__NIC_NAME__|$(escape_sed_replacement "${NIC_NAME}")|g" \
    -e "s|__MAX_MODEL_LEN__|$(escape_sed_replacement "${MAX_MODEL_LEN}")|g" \
    -e "s|__PREFILL_MAX_BATCHED_TOKENS__|$(escape_sed_replacement "${PREFILL_MAX_BATCHED_TOKENS}")|g" \
    -e "s|__DECODE_MAX_BATCHED_TOKENS__|$(escape_sed_replacement "${DECODE_MAX_BATCHED_TOKENS}")|g" \
    -e "s|__PREFILL_MAX_NUM_SEQS__|$(escape_sed_replacement "${PREFILL_MAX_NUM_SEQS}")|g" \
    -e "s|__DECODE_MAX_NUM_SEQS__|$(escape_sed_replacement "${DECODE_MAX_NUM_SEQS}")|g" \
    -e "s|__GPU_MEMORY_UTILIZATION__|$(escape_sed_replacement "${GPU_MEMORY_UTILIZATION}")|g" \
    -e "s|__REASONING_PARSER__|$(escape_sed_replacement "${REASONING_PARSER}")|g" \
    -e "s|__PROXY_VARIANT__|$(escape_sed_replacement "${PROXY_VARIANT}")|g" \
    -e "s|__PROXY_CODE_SHA256__|$(escape_sed_replacement "${PROXY_CODE_SHA256}")|g" \
    -e "s|__KV_AWARE_ROUTING__|$(escape_sed_replacement "${KV_AWARE_ROUTING}")|g" \
    -e "s|__PROXY_DECODER_COUNT__|$(escape_sed_replacement "${PROXY_DECODER_COUNT}")|g" \
    -e "s|__SESSION_LRU_SIZE__|$(escape_sed_replacement "${SESSION_LRU_SIZE}")|g" \
    -e "s|__PREFIX_HASH_CHARS__|$(escape_sed_replacement "${PREFIX_HASH_CHARS}")|g" \
    -e "s|__PREFIX_LRU_SIZE__|$(escape_sed_replacement "${PREFIX_LRU_SIZE}")|g" \
    -e "s|__PREFILL_ACTIVE_TOKEN_WEIGHT__|$(escape_sed_replacement "${PREFILL_ACTIVE_TOKEN_WEIGHT}")|g" \
    -e "s|__AFFINITY_OVERLOAD_FACTOR__|$(escape_sed_replacement "${AFFINITY_OVERLOAD_FACTOR}")|g" \
    -e "s|__AFFINITY_MISS_UNBIND_THRESHOLD__|$(escape_sed_replacement "${AFFINITY_MISS_UNBIND_THRESHOLD}")|g" \
    -e "s|__AFFINITY_CACHE_DISCOUNT_ALPHA__|$(escape_sed_replacement "${AFFINITY_CACHE_DISCOUNT_ALPHA}")|g" \
    -e "s|__RESTART_TOKEN__|$(escape_sed_replacement "${RESTART_TOKEN}")|g" \
    "${file}"
}

validate_proxy_source() {
  case "${KV_AWARE_ROUTING}" in
    true|false) ;;
    *)
      echo "KV_AWARE_ROUTING must be true or false: ${KV_AWARE_ROUTING}" >&2
      exit 2
      ;;
  esac
  if [[ "${PROXY_VARIANT}" == "baseline" && "${KV_AWARE_ROUTING}" != "false" ]]; then
    echo "KV_AWARE_ROUTING must be false when PROXY_VARIANT=baseline" >&2
    exit 2
  fi
  case "${PROXY_DECODER_COUNT}" in
    1|2|3|4) ;;
    *)
      echo "PROXY_DECODER_COUNT must be 1, 2, 3, or 4: ${PROXY_DECODER_COUNT}" >&2
      exit 2
      ;;
  esac
  if [[ "${PROXY_SOURCE_PATH}" != /* ]]; then
    echo "PROXY_SOURCE_PATH must be an absolute path: ${PROXY_SOURCE_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "${PROXY_SOURCE_PATH}" ]]; then
    echo "proxy source not found: ${PROXY_SOURCE_PATH}" >&2
    exit 2
  fi
}

apply_namespace() {
  kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
}

apply_launcher_config() {
  kubectl -n "${NAMESPACE}" create configmap pd-launcher \
    --from-file=launch_role.sh="${SCRIPT_DIR}/launch_role.sh" \
    --from-file=launch_proxy.sh="${SCRIPT_DIR}/launch_proxy.sh" \
    --dry-run=client -o yaml | kubectl apply -f -
}

apply_proxy_config() {
  kubectl -n "${NAMESPACE}" create configmap pd-proxy-code \
    --from-file=load_balance_proxy_server_example.py="${PROXY_SOURCE_PATH}" \
    --dry-run=client -o yaml | kubectl apply -f -
}

prepare_role_migration() {
  local role="$1"
  local service_name="pd-${role}"
  local cluster_ip
  local pod_management_policy

  # Remove objects from the former Deployment/non-headless-Service layout only
  # when they are actually present. A headless Service should stay available
  # during normal updates so the proxy does not lose backend DNS resolution.
  kubectl -n "${NAMESPACE}" delete "deployment/${service_name}" \
    --ignore-not-found=true --wait=true --timeout=5m

  cluster_ip="$(
    kubectl -n "${NAMESPACE}" get "service/${service_name}" \
      -o jsonpath='{.spec.clusterIP}' 2>/dev/null || true
  )"
  if [[ -n "${cluster_ip}" && "${cluster_ip}" != "None" ]]; then
    kubectl -n "${NAMESPACE}" delete "service/${service_name}" --wait=true --timeout=5m
  fi

  # podManagementPolicy is immutable. Recreate only StatefulSets created by an
  # earlier version that still use OrderedReady.
  pod_management_policy="$(
    kubectl -n "${NAMESPACE}" get "statefulset/${service_name}" \
      -o jsonpath='{.spec.podManagementPolicy}' 2>/dev/null || true
  )"
  if [[ -n "${pod_management_policy}" && "${pod_management_policy}" != "Parallel" ]]; then
    kubectl -n "${NAMESPACE}" delete "statefulset/${service_name}" --wait=true --timeout=5m
  fi
}

apply_prefill() {
  require_value PREFILL_NODE
  kubectl get node "${PREFILL_NODE}" >/dev/null
  prepare_role_migration prefill
  render "${SCRIPT_DIR}/prefill.yaml" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status statefulset/pd-prefill --timeout=60m
}

apply_decode() {
  require_value DECODE_NODE
  kubectl get node "${DECODE_NODE}" >/dev/null
  prepare_role_migration decode
  render "${SCRIPT_DIR}/decode.yaml" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status statefulset/pd-decode --timeout=60m
}

apply_proxy() {
  require_value PREFILL_NODE
  render "${SCRIPT_DIR}/proxy.yaml" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status deployment/pd-proxy --timeout=10m
}

show_status() {
  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "namespace ${NAMESPACE} does not exist"
    return 0
  fi
  kubectl -n "${NAMESPACE}" get pods,services -o wide
  if kubectl -n "${NAMESPACE}" get service pd-proxy >/dev/null 2>&1; then
    echo
    echo "Access locally:"
    echo "  kubectl -n ${NAMESPACE} port-forward service/pd-proxy 8000:8000"
  fi
}

cleanup_prefill() {
  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "namespace ${NAMESPACE} does not exist; nothing to clean"
    return 0
  fi
  kubectl -n "${NAMESPACE}" delete \
    statefulset/pd-prefill \
    deployment/pd-prefill \
    service/pd-prefill \
    --ignore-not-found=true \
    --wait=true \
    --timeout=5m
}

cleanup_decode() {
  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "namespace ${NAMESPACE} does not exist; nothing to clean"
    return 0
  fi
  kubectl -n "${NAMESPACE}" delete \
    statefulset/pd-decode \
    deployment/pd-decode \
    service/pd-decode \
    --ignore-not-found=true \
    --wait=true \
    --timeout=5m
}

cleanup_proxy() {
  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "namespace ${NAMESPACE} does not exist; nothing to clean"
    return 0
  fi
  kubectl -n "${NAMESPACE}" delete \
    deployment/pd-proxy \
    service/pd-proxy \
    configmap/pd-proxy-code \
    --ignore-not-found=true \
    --wait=true \
    --timeout=5m
}

cleanup_benchmark() {
  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "namespace ${NAMESPACE} does not exist; nothing to clean"
    return 0
  fi
  kubectl -n "${NAMESPACE}" delete pod,configmap \
    -l app.kubernetes.io/name=pd-benchmark \
    --ignore-not-found=true \
    --wait=true \
    --timeout=5m
  # Older runs created unlabeled configmaps named pd-benchmark-*.
  local leftover_cms
  leftover_cms="$(
    kubectl -n "${NAMESPACE}" get configmap -o name 2>/dev/null \
      | grep '^configmap/pd-benchmark-' || true
  )"
  if [[ -n "${leftover_cms}" ]]; then
    # shellcheck disable=SC2086
    kubectl -n "${NAMESPACE}" delete ${leftover_cms} \
      --ignore-not-found=true \
      --wait=true \
      --timeout=5m
  fi
}

cleanup_all() {
  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    echo "namespace ${NAMESPACE} does not exist; nothing to clean"
    return 0
  fi
  cleanup_benchmark
  kubectl -n "${NAMESPACE}" delete \
    statefulset/pd-prefill \
    statefulset/pd-decode \
    deployment/pd-prefill \
    deployment/pd-decode \
    deployment/pd-proxy \
    service/pd-prefill \
    service/pd-decode \
    service/pd-proxy \
    configmap/pd-launcher \
    configmap/pd-proxy-code \
    --ignore-not-found=true \
    --wait=true \
    --timeout=5m
}

require_command kubectl

case "${ACTION}" in
  prefill)
    require_value VLLM_IMAGE
    apply_namespace
    apply_launcher_config
    apply_prefill
    ;;
  decode)
    require_value VLLM_IMAGE
    apply_namespace
    apply_launcher_config
    apply_decode
    ;;
  proxy)
    require_value VLLM_IMAGE
    validate_proxy_source
    apply_namespace
    apply_launcher_config
    apply_proxy_config
    apply_proxy
    ;;
  all)
    require_value VLLM_IMAGE
    require_value PREFILL_NODE
    require_value DECODE_NODE
    if [[ "${PREFILL_NODE}" == "${DECODE_NODE}" ]]; then
      echo "PREFILL_NODE and DECODE_NODE must be different for this 4P4D TP2 layout" >&2
      exit 2
    fi
    validate_proxy_source
    apply_namespace
    apply_launcher_config
    apply_prefill
    apply_decode
    apply_proxy_config
    apply_proxy
    show_status
    ;;
  status)
    show_status
    ;;
  cleanup-prefill)
    cleanup_prefill
    ;;
  cleanup-decode)
    cleanup_decode
    ;;
  cleanup-proxy)
    cleanup_proxy
    ;;
  cleanup-benchmark)
    cleanup_benchmark
    ;;
  cleanup-all)
    cleanup_all
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    usage
    exit 2
    ;;
esac
