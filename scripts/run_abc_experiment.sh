#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 1 )); then
  echo "usage: MODEL=... PREFILL_NODE=... $0 benchmarks/profiles/<profile>.json [benchmark arguments]" >&2
  exit 2
fi

PROFILE="$1"
shift
: "${MODEL:?MODEL is required}"
: "${PREFILL_NODE:?PREFILL_NODE is required to roll out each proxy group}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deploy/kubernetes/qwen3-32b-4p4d-tp2"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXECUTION_MODE="${BENCHMARK_EXECUTION_MODE:-pod}"
NAMESPACE="${NAMESPACE:-qwen-pd}"
VLLM_IMAGE="${VLLM_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"

if [[ ! -f "${PROFILE}" ]]; then
  echo "workload profile not found: ${PROFILE}" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[&|\\]/\\&/g'
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
resource_timestamp="$(date -u +%Y%m%d-%H%M%S)"
experiment_dir="${ABC_RESULTS_DIR:-${REPO_ROOT}/results/runs/${timestamp}-abc}"
mkdir -p "${experiment_dir}"

profile_concurrency="$("${PYTHON_BIN}" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("load", {}).get("concurrency", 64))' \
  "${PROFILE}")"
concurrency="${CONCURRENCY:-${profile_concurrency}}"
read -r -a groups <<<"${GROUP_ORDER:-baseline candidate-off candidate-on}"

if (( ${#groups[@]} != 3 )); then
  echo "GROUP_ORDER must contain baseline, candidate-off, and candidate-on exactly once" >&2
  exit 2
fi
for expected_group in baseline candidate-off candidate-on; do
  matches=0
  for group in "${groups[@]}"; do
    [[ "${group}" == "${expected_group}" ]] && matches=$((matches + 1))
  done
  if (( matches != 1 )); then
    echo "GROUP_ORDER must contain baseline, candidate-off, and candidate-on exactly once" >&2
    exit 2
  fi
done

log "A/B/C experiment started: mode=${EXECUTION_MODE} profile=${PROFILE} output=${experiment_dir}"

run_local() {
  : "${BASE_URL:?BASE_URL is required in local mode}"
  : "${TOKENIZER_URL:?TOKENIZER_URL is required in local mode}"
  local workload_file="${experiment_dir}/workload.jsonl"

  log "generating one shared workload via ${TOKENIZER_URL}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_benchmark_workload.py" \
    --profile "${PROFILE}" \
    --output "${workload_file}" \
    --tokenizer-url "${TOKENIZER_URL}" \
    --model "${MODEL}" \
    --timeout "${TOKENIZER_TIMEOUT:-120}"
  log "shared workload generated: ${workload_file}"

  local group
  for group in "${groups[@]}"; do
    log "starting group=${group} (proxy rollout + local benchmark)"
    WORKLOAD_FILE="${workload_file}" \
    OUTPUT_DIR="${experiment_dir}/${group}" \
    CONCURRENCY="${concurrency}" \
      bash "${SCRIPT_DIR}/run_experiment.sh" "${group}" "$@"
    log "completed group=${group}"
  done

  log "comparing baseline, candidate-off, and candidate-on"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_abc_results.py" "${experiment_dir}"
}

deploy_proxy_group() {
  local group="$1"
  local proxy_variant
  local kv_aware
  case "${group}" in
    baseline)
      proxy_variant=baseline
      kv_aware=false
      ;;
    candidate-off)
      proxy_variant=candidate
      kv_aware=false
      ;;
    candidate-on)
      proxy_variant=candidate
      kv_aware=true
      ;;
    *)
      echo "unknown group: ${group}; expected baseline, candidate-off, or candidate-on" >&2
      return 2
      ;;
  esac
  log "${group}: deploying proxy variant=${proxy_variant} kv_aware=${kv_aware}"
  PROXY_VARIANT="${proxy_variant}" KV_AWARE_ROUTING="${kv_aware}" \
    bash "${DEPLOY_DIR}/deploy.sh" proxy
}

run_in_pod() {
  command -v kubectl >/dev/null 2>&1 || {
    echo "kubectl is required in pod mode" >&2
    return 2
  }

  local pod_name="pd-benchmark-${resource_timestamp}"
  local configmap_name="pd-benchmark-${resource_timestamp}"
  local workload_file=/results/workload.jsonl
  local tokenizer_url="${IN_CLUSTER_TOKENIZER_URL:-http://pd-prefill-0.pd-prefill:7100}"
  local base_url="${IN_CLUSTER_BASE_URL:-http://pd-proxy:8000}"
  local benchmark_manifest="${DEPLOY_DIR}/benchmark-pod.yaml"

  on_exit() {
    local status=$?
    if (( status != 0 )); then
      log "benchmark failed; keeping pod/${pod_name} and configmap/${configmap_name} for inspection"
      log "inspect with: kubectl -n ${NAMESPACE} exec -it ${pod_name} -- bash"
      log "recover results with: kubectl -n ${NAMESPACE} cp ${pod_name}:/results/. ${experiment_dir}"
    fi
  }
  trap on_exit EXIT

  log "creating benchmark assets configmap=${configmap_name}"
  kubectl -n "${NAMESPACE}" create configmap "${configmap_name}" \
    --from-file=benchmark_workload.py="${SCRIPT_DIR}/benchmark_workload.py" \
    --from-file=generate_benchmark_workload.py="${SCRIPT_DIR}/generate_benchmark_workload.py" \
    --from-file=benchmark_session_affinity.py="${SCRIPT_DIR}/benchmark_session_affinity.py" \
    --from-file=compare_abc_results.py="${SCRIPT_DIR}/compare_abc_results.py" \
    --from-file=profile.json="${PROFILE}" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n "${NAMESPACE}" label configmap "${configmap_name}" \
    app.kubernetes.io/name=pd-benchmark --overwrite

  log "creating benchmark pod=${pod_name} image=${VLLM_IMAGE}"
  sed \
    -e "s|__BENCHMARK_POD__|$(escape_sed_replacement "${pod_name}")|g" \
    -e "s|__BENCHMARK_CONFIGMAP__|$(escape_sed_replacement "${configmap_name}")|g" \
    -e "s|__NAMESPACE__|$(escape_sed_replacement "${NAMESPACE}")|g" \
    -e "s|__VLLM_IMAGE__|$(escape_sed_replacement "${VLLM_IMAGE}")|g" \
    "${benchmark_manifest}" | kubectl -n "${NAMESPACE}" apply -f -
  kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${pod_name}" --timeout=10m

  log "checking benchmark pod runtime dependencies"
  kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    python -c 'import requests'
  kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    /bin/bash -ec 'command -v tar >/dev/null'

  log "generating workload inside pod via ${tokenizer_url}"
  kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    python /opt/benchmark/generate_benchmark_workload.py \
      --profile /opt/benchmark/profile.json \
      --output "${workload_file}" \
      --tokenizer-url "${tokenizer_url}" \
      --model "${MODEL}" \
      --timeout "${TOKENIZER_TIMEOUT:-120}"

  local group
  for group in "${groups[@]}"; do
    deploy_proxy_group "${group}"
    log "${group}: proxy ready; benchmark pod is starting workload"
    kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
      python /opt/benchmark/benchmark_session_affinity.py \
        --base-url "${base_url}" \
        --model "${MODEL}" \
        --workload-file "${workload_file}" \
        --output-dir "/results/${group}" \
        --concurrency "${concurrency}" \
        --system-warmup-requests "${SYSTEM_WARMUP_REQUESTS:-8}" \
        --label "${group}" \
        --reset-before \
        "$@"
    log "completed group=${group}"
  done

  log "comparing baseline, candidate-off, and candidate-on inside benchmark pod"
  kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    python /opt/benchmark/compare_abc_results.py /results

  log "copying benchmark artifacts to ${experiment_dir}"
  kubectl -n "${NAMESPACE}" cp "${pod_name}:/results/." "${experiment_dir}"

  if [[ "${KEEP_BENCHMARK_POD:-false}" == "true" ]]; then
    log "keeping benchmark pod=${pod_name} by request"
  else
    log "removing temporary benchmark pod and assets configmap"
    kubectl -n "${NAMESPACE}" delete "pod/${pod_name}" "configmap/${configmap_name}" \
      --wait=true --timeout=5m
  fi
  trap - EXIT
}

case "${EXECUTION_MODE}" in
  pod)
    run_in_pod "$@"
    ;;
  local)
    run_local "$@"
    ;;
  *)
    echo "BENCHMARK_EXECUTION_MODE must be pod or local: ${EXECUTION_MODE}" >&2
    exit 2
    ;;
esac

echo "A/B/C experiment complete: ${experiment_dir}"
