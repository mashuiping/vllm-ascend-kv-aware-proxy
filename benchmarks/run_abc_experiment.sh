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
BASELINE_PROXY_SOURCE_PATH="${BASELINE_PROXY_SOURCE_PATH:-${REPO_ROOT}/baseline/load_balance_proxy_server_example.py}"
CANDIDATE_PROXY_SOURCE_PATH="${CANDIDATE_PROXY_SOURCE_PATH:-${REPO_ROOT}/load_balance_proxy_server_example.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXECUTION_MODE="${BENCHMARK_EXECUTION_MODE:-pod}"
NAMESPACE="${NAMESPACE:-qwen-pd}"
VLLM_IMAGE="${VLLM_IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
RESET_FAILURE_ACTION="${RESET_FAILURE_ACTION:-abort}"

if [[ -n "${PROXY_SOURCE_PATH:-}" ]]; then
  echo "PROXY_SOURCE_PATH is unsafe for A/B/C runs because it overrides every group; use BASELINE_PROXY_SOURCE_PATH and CANDIDATE_PROXY_SOURCE_PATH" >&2
  exit 2
fi

for proxy_source in "${BASELINE_PROXY_SOURCE_PATH}" "${CANDIDATE_PROXY_SOURCE_PATH}"; do
  if [[ "${proxy_source}" != /* || ! -f "${proxy_source}" ]]; then
    echo "proxy source must be an existing absolute path: ${proxy_source}" >&2
    exit 2
  fi
done

case "${RESET_FAILURE_ACTION}" in
  abort|restart) ;;
  *)
    echo "RESET_FAILURE_ACTION must be abort or restart" >&2
    exit 2
    ;;
esac

if [[ ! -f "${PROFILE}" ]]; then
  echo "workload profile not found: ${PROFILE}" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

# Transient kubectl/API blips are common after long runs. Retry once after a
# short delay; callers must be idempotent (overwrite the same objects/files).
retry_once() {
  local delay="${RETRY_DELAY_SECONDS:-10}"
  if "$@"; then
    return 0
  fi
  log "command failed; sleeping ${delay}s then retrying once"
  sleep "${delay}"
  "$@"
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
read -r profile_decoder_count profile_sample_interval <<<"$("${PYTHON_BIN}" -c \
  'import json, sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p.get("deployment", {}).get("proxy_decoder_count", 4), p.get("observability", {}).get("sample_interval_s", 1.0))' \
  "${PROFILE}")"
profile_scenario="$("${PYTHON_BIN}" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("scenario", ""))' \
  "${PROFILE}")"
profile_seed="$("${PYTHON_BIN}" -c \
  'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("seed", 20260724)))' \
  "${PROFILE}")"
workload_seed="${WORKLOAD_SEED:-${profile_seed}}"
if [[ ! "${workload_seed}" =~ ^-?[0-9]+$ ]]; then
  echo "WORKLOAD_SEED/profile seed must be an integer: ${workload_seed}" >&2
  exit 2
fi
comparison_mode="${ABC_COMPARISON_MODE:-auto}"
if [[ "${comparison_mode}" == "auto" ]]; then
  if [[ "${profile_scenario}" == "load-balance" ]]; then
    comparison_mode=active-token
  else
    comparison_mode=affinity
  fi
fi
case "${comparison_mode}" in
  affinity|active-token|affinity-guard) ;;
  *)
    echo "ABC_COMPARISON_MODE must be auto, affinity, active-token, or affinity-guard: ${comparison_mode}" >&2
    exit 2
    ;;
esac

# Guard settings apply to candidate-on only in affinity-guard mode. Defaults
# enable both mechanisms so the C group exercises escape and miss-unbind.
# Cache-discount alpha defaults off so existing guard runs stay comparable;
# opt in explicitly when the experiment targets reservation discounting.
guard_overload_factor="${AFFINITY_OVERLOAD_FACTOR:-1.5}"
guard_miss_unbind_threshold="${AFFINITY_MISS_UNBIND_THRESHOLD:-3}"
guard_cache_discount_alpha="${AFFINITY_CACHE_DISCOUNT_ALPHA:-0}"
decoder_count="${BENCHMARK_DECODER_COUNT:-${profile_decoder_count}}"
sample_interval="${SAMPLE_INTERVAL:-${profile_sample_interval}}"
case "${decoder_count}" in
  1|2|3|4) ;;
  *)
    echo "BENCHMARK_DECODER_COUNT/profile deployment.proxy_decoder_count must be 1, 2, 3, or 4" >&2
    exit 2
    ;;
esac
workload_override_args=(--seed "${workload_seed}")
if [[ -n "${SYSTEM_PROMPT_TOKENS:-}" ]]; then
  workload_override_args+=(--system-prompt-tokens "${SYSTEM_PROMPT_TOKENS}")
  log "overriding data.system_prompt_tokens=${SYSTEM_PROMPT_TOKENS} for generated workload"
fi
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

log "A/B/C experiment started: mode=${EXECUTION_MODE} comparison=${comparison_mode} profile=${PROFILE} seed=${workload_seed} decoder_count=${decoder_count} concurrency=${concurrency} sample_interval=${sample_interval}s output=${experiment_dir}"

run_local() {
  : "${BASE_URL:?BASE_URL is required in local mode}"
  : "${TOKENIZER_URL:?TOKENIZER_URL is required in local mode}"
  local workload_file="${experiment_dir}/workload.jsonl"
  local metadata_json
  metadata_json="$(build_metadata_json)"
  "${PYTHON_BIN}" -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2] + "\n", encoding="utf-8")' \
    "${experiment_dir}/metadata.json" "${metadata_json}"

  log "generating one shared workload via ${TOKENIZER_URL}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_benchmark_workload.py" \
    --profile "${PROFILE}" \
    --output "${workload_file}" \
    --tokenizer-url "${TOKENIZER_URL}" \
    --model "${MODEL}" \
    --timeout "${TOKENIZER_TIMEOUT:-120}" \
    --tokenizer-concurrency "${TOKENIZER_CONCURRENCY:-16}" \
    "${workload_override_args[@]}"
  log "shared workload generated: ${workload_file}"

  local group
  for group in "${groups[@]}"; do
    log "starting group=${group} (proxy rollout + local benchmark)"
    WORKLOAD_FILE="${workload_file}" \
    OUTPUT_DIR="${experiment_dir}/${group}" \
    CONCURRENCY="${concurrency}" \
    PROXY_DECODER_COUNT="${decoder_count}" \
    ABC_COMPARISON_MODE="${comparison_mode}" \
    BASELINE_PROXY_SOURCE_PATH="${BASELINE_PROXY_SOURCE_PATH}" \
    CANDIDATE_PROXY_SOURCE_PATH="${CANDIDATE_PROXY_SOURCE_PATH}" \
      bash "${SCRIPT_DIR}/run_experiment.sh" "${group}" --sample-interval "${sample_interval}" "$@"
    log "completed group=${group}"
  done

  log "comparing baseline, candidate-off, and candidate-on"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_abc_results.py" "${experiment_dir}"
  log "rendering HTML report"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/render_benchmark_report.py" "${experiment_dir}"
}

deploy_proxy_group() {
  local group="$1"
  local proxy_variant
  local kv_aware
  local proxy_source
  local active_token_weight="${PREFILL_ACTIVE_TOKEN_WEIGHT:-1.0}"
  local overload_factor=0
  local miss_unbind_threshold=0
  local cache_discount_alpha=0
  case "${group}" in
    baseline)
      if [[ "${comparison_mode}" == "affinity-guard" ]]; then
        # affinity-guard compares the candidate affinity policy against itself
        # with guards enabled; the upstream baseline source plays no role.
        proxy_variant=candidate
        kv_aware=true
        proxy_source="${CANDIDATE_PROXY_SOURCE_PATH}"
      else
        proxy_variant=baseline
        kv_aware=false
        proxy_source="${BASELINE_PROXY_SOURCE_PATH}"
      fi
      ;;
    candidate-off)
      proxy_variant=candidate
      proxy_source="${CANDIDATE_PROXY_SOURCE_PATH}"
      if [[ "${comparison_mode}" == "affinity-guard" ]]; then
        kv_aware=true
      else
        kv_aware=false
      fi
      if [[ "${comparison_mode}" == "active-token" ]]; then
        active_token_weight=0
      fi
      ;;
    candidate-on)
      proxy_variant=candidate
      if [[ "${comparison_mode}" == "active-token" ]]; then
        kv_aware=false
      else
        kv_aware=true
      fi
      proxy_source="${CANDIDATE_PROXY_SOURCE_PATH}"
      if [[ "${comparison_mode}" == "affinity-guard" ]]; then
        overload_factor="${guard_overload_factor}"
        miss_unbind_threshold="${guard_miss_unbind_threshold}"
        cache_discount_alpha="${guard_cache_discount_alpha}"
      fi
      ;;
    *)
      echo "unknown group: ${group}; expected baseline, candidate-off, or candidate-on" >&2
      return 2
      ;;
  esac
  log "${group}: deploying proxy variant=${proxy_variant} kv_aware=${kv_aware} decoder_count=${decoder_count} overload_factor=${overload_factor} miss_unbind_threshold=${miss_unbind_threshold} cache_discount_alpha=${cache_discount_alpha}"
  PROXY_VARIANT="${proxy_variant}" KV_AWARE_ROUTING="${kv_aware}" PROXY_DECODER_COUNT="${decoder_count}" \
    PREFILL_ACTIVE_TOKEN_WEIGHT="${active_token_weight}" \
    AFFINITY_OVERLOAD_FACTOR="${overload_factor}" \
    AFFINITY_MISS_UNBIND_THRESHOLD="${miss_unbind_threshold}" \
    AFFINITY_CACHE_DISCOUNT_ALPHA="${cache_discount_alpha}" \
    PROXY_SOURCE_PATH="${proxy_source}" \
    bash "${DEPLOY_DIR}/deploy.sh" proxy
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

build_metadata_json() {
  local baseline_sha candidate_sha repository_commit repository_dirty benchmark_sha workload_generator_sha comparison_sha profile_sha
  baseline_sha="$(sha256_file "${BASELINE_PROXY_SOURCE_PATH}")"
  candidate_sha="$(sha256_file "${CANDIDATE_PROXY_SOURCE_PATH}")"
  repository_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
    repository_dirty=true
  else
    repository_dirty=false
  fi
  benchmark_sha="$(sha256_file "${SCRIPT_DIR}/benchmark_session_affinity.py")"
  workload_generator_sha="$(sha256_file "${SCRIPT_DIR}/generate_benchmark_workload.py")"
  comparison_sha="$(sha256_file "${SCRIPT_DIR}/compare_abc_results.py")"
  profile_sha="$(sha256_file "${PROFILE}")"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.dumps({"comparison_mode":sys.argv[1],"group_order":sys.argv[2].split(),"profile":sys.argv[3],"profile_scenario":sys.argv[4],"repository_commit":sys.argv[5],"repository_dirty":sys.argv[6]=="true","baseline_source_sha256":sys.argv[7],"candidate_source_sha256":sys.argv[8],"benchmark_sha256":sys.argv[9],"workload_generator_sha256":sys.argv[10],"comparison_tool_sha256":sys.argv[11],"profile_sha256":sys.argv[12],"model":sys.argv[13],"image":sys.argv[14],"proxy_decoder_count":int(sys.argv[15]),"sample_interval_seconds":float(sys.argv[16]),"workload_seed":int(sys.argv[17])}, separators=(",", ":")))' \
    "${comparison_mode}" "${groups[*]}" "${PROFILE}" "${profile_scenario}" "${repository_commit}" "${repository_dirty}" \
    "${baseline_sha}" "${candidate_sha}" "${benchmark_sha}" "${workload_generator_sha}" "${comparison_sha}" "${profile_sha}" \
    "${MODEL}" "${VLLM_IMAGE}" "${decoder_count}" "${sample_interval}" "${workload_seed}"
}

verify_proxy_group() {
  local group="$1"
  local expected_variant expected_kv_aware expected_source expected_active_token_weight
  local expected_overload_factor=0 expected_miss_unbind_threshold=0 expected_cache_discount_alpha=0
  expected_active_token_weight="${PREFILL_ACTIVE_TOKEN_WEIGHT:-1.0}"
  case "${group}" in
    baseline)
      if [[ "${comparison_mode}" == "affinity-guard" ]]; then
        expected_variant=candidate
        expected_kv_aware=true
        expected_source="${CANDIDATE_PROXY_SOURCE_PATH}"
      else
        expected_variant=baseline
        expected_kv_aware=false
        expected_source="${BASELINE_PROXY_SOURCE_PATH}"
      fi
      ;;
    candidate-off)
      expected_variant=candidate
      expected_source="${CANDIDATE_PROXY_SOURCE_PATH}"
      if [[ "${comparison_mode}" == "affinity-guard" ]]; then
        expected_kv_aware=true
      else
        expected_kv_aware=false
      fi
      if [[ "${comparison_mode}" == "active-token" ]]; then
        expected_active_token_weight=0
      fi
      ;;
    candidate-on)
      expected_variant=candidate
      if [[ "${comparison_mode}" == "active-token" ]]; then
        expected_kv_aware=false
      else
        expected_kv_aware=true
      fi
      expected_source="${CANDIDATE_PROXY_SOURCE_PATH}"
      if [[ "${comparison_mode}" == "affinity-guard" ]]; then
        expected_overload_factor="${guard_overload_factor}"
        expected_miss_unbind_threshold="${guard_miss_unbind_threshold}"
        expected_cache_discount_alpha="${guard_cache_discount_alpha}"
      fi
      ;;
  esac

  local pod_name expected_sha actual_sha actual_variant actual_kv_aware actual_active_token_weight pod_uid image_id
  local actual_overload_factor actual_miss_unbind_threshold actual_cache_discount_alpha
  pod_name="$(kubectl -n "${NAMESPACE}" get pods -l app.kubernetes.io/name=pd-proxy \
    --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')"
  [[ -n "${pod_name}" ]] || { echo "${group}: no proxy pod found" >&2; return 1; }
  expected_sha="$(sha256_file "${expected_source}")"
  actual_sha="$(kubectl -n "${NAMESPACE}" exec "${pod_name}" -- python -c \
    'import hashlib, pathlib; print(hashlib.sha256(pathlib.Path("/opt/pd-proxy/load_balance_proxy_server_example.py").read_bytes()).hexdigest())')"
  read -r actual_variant actual_kv_aware actual_active_token_weight actual_overload_factor actual_miss_unbind_threshold actual_cache_discount_alpha < <(kubectl -n "${NAMESPACE}" exec "${pod_name}" -- python -c \
    'import os; print(os.environ.get("PROXY_VARIANT", ""), os.environ.get("KV_AWARE_ROUTING", ""), os.environ.get("PREFILL_ACTIVE_TOKEN_WEIGHT", ""), os.environ.get("AFFINITY_OVERLOAD_FACTOR", "0"), os.environ.get("AFFINITY_MISS_UNBIND_THRESHOLD", "0"), os.environ.get("AFFINITY_CACHE_DISCOUNT_ALPHA", "0"))')
  pod_uid="$(kubectl -n "${NAMESPACE}" get pod "${pod_name}" -o jsonpath='{.metadata.uid}')"
  image_id="$(kubectl -n "${NAMESPACE}" get pod "${pod_name}" -o jsonpath='{.status.containerStatuses[?(@.name=="proxy")].imageID}')"

  if [[ "${actual_sha}" != "${expected_sha}" || "${actual_variant}" != "${expected_variant}" || "${actual_kv_aware}" != "${expected_kv_aware}" || "${actual_active_token_weight}" != "${expected_active_token_weight}" || "${actual_overload_factor}" != "${expected_overload_factor}" || "${actual_miss_unbind_threshold}" != "${expected_miss_unbind_threshold}" || "${actual_cache_discount_alpha}" != "${expected_cache_discount_alpha}" ]]; then
    echo "${group}: proxy identity mismatch expected=${expected_variant}/${expected_kv_aware}/weight-${expected_active_token_weight}/guard-${expected_overload_factor}-${expected_miss_unbind_threshold}-${expected_cache_discount_alpha}/${expected_sha} actual=${actual_variant}/${actual_kv_aware}/weight-${actual_active_token_weight}/guard-${actual_overload_factor}-${actual_miss_unbind_threshold}-${actual_cache_discount_alpha}/${actual_sha}" >&2
    return 1
  fi

  "${PYTHON_BIN}" -c \
    'import json,sys; print(json.dumps(dict(zip(("group","comparison_mode","expected_variant","actual_variant","expected_kv_aware","actual_kv_aware","expected_active_token_weight","actual_active_token_weight","expected_affinity_overload_factor","actual_affinity_overload_factor","expected_affinity_miss_unbind_threshold","actual_affinity_miss_unbind_threshold","expected_affinity_cache_discount_alpha","actual_affinity_cache_discount_alpha","expected_source_sha256","actual_source_sha256","pod_name","pod_uid","image_id"), sys.argv[1:])), separators=(",", ":")))' \
    "${group}" "${comparison_mode}" "${expected_variant}" "${actual_variant}" "${expected_kv_aware}" "${actual_kv_aware}" \
    "${expected_active_token_weight}" "${actual_active_token_weight}" \
    "${expected_overload_factor}" "${actual_overload_factor}" \
    "${expected_miss_unbind_threshold}" "${actual_miss_unbind_threshold}" \
    "${expected_cache_discount_alpha}" "${actual_cache_discount_alpha}" \
    "${expected_sha}" "${actual_sha}" "${pod_name}" "${pod_uid}" "${image_id}"
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
  local archive_name="benchmark-results.tar.gz"
  local remote_archive="/tmp/${archive_name}"
  local local_archive="${experiment_dir}/${archive_name}"
  local results_copied=false
  local -a backend_args=()
  local index
  for index in 0 1 2 3; do
    backend_args+=(
      --prefill-base-url "http://pd-prefill-${index}.pd-prefill:7100"
      --prefill-metrics-url "http://pd-prefill-${index}.pd-prefill:7100/metrics"
    )
  done
  for ((index = 0; index < decoder_count; index++)); do
    backend_args+=(--decode-metrics-url "http://pd-decode-${index}.pd-decode:7200/metrics")
  done

  # Overwrite the same archive/files on every attempt so a partial kubectl cp
  # can be retried without leaving a corrupt local tarball.
  copy_pod_results() {
    log "archiving benchmark artifacts inside pod"
    kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
      tar -C /results -czf "${remote_archive}" .
    rm -f "${local_archive}"
    log "copying compressed archive to ${local_archive}"
    kubectl -n "${NAMESPACE}" cp "${pod_name}:${remote_archive}" "${local_archive}"
    tar -tzf "${local_archive}" >/dev/null
    log "extracting archive into ${experiment_dir}"
    tar -xzf "${local_archive}" -C "${experiment_dir}"
    rm -f "${local_archive}"
    results_copied=true
  }

  on_exit() {
    local status=$?
    if (( status == 0 )); then
      return
    fi
    if [[ "${results_copied}" != "true" ]]; then
      log "benchmark failed; retrying result copy after ${RETRY_DELAY_SECONDS:-10}s"
      sleep "${RETRY_DELAY_SECONDS:-10}"
      copy_pod_results || true
    fi
    if [[ "${results_copied}" == "true" ]]; then
      log "benchmark failed; copied results into ${experiment_dir}; keeping pod/${pod_name} and configmap/${configmap_name} for inspection"
      log "inspect with: kubectl -n ${NAMESPACE} exec -it ${pod_name} -- bash"
      "${PYTHON_BIN}" "${SCRIPT_DIR}/render_benchmark_report.py" "${experiment_dir}" || true
      return
    fi
    log "benchmark failed; keeping pod/${pod_name} and configmap/${configmap_name} for inspection"
    log "inspect with: kubectl -n ${NAMESPACE} exec -it ${pod_name} -- bash"
    log "recover results with:"
    log "  kubectl -n ${NAMESPACE} exec ${pod_name} -- tar -C /results -czf /tmp/benchmark-results.tar.gz ."
    log "  kubectl -n ${NAMESPACE} cp ${pod_name}:/tmp/benchmark-results.tar.gz ${experiment_dir}/benchmark-results.tar.gz"
    log "  tar -xzf ${experiment_dir}/benchmark-results.tar.gz -C ${experiment_dir}"
    log "  rm -f ${experiment_dir}/benchmark-results.tar.gz"
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
  retry_once kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${pod_name}" --timeout=10m

  local metadata_json
  metadata_json="$(build_metadata_json)"
  retry_once kubectl -n "${NAMESPACE}" exec "${pod_name}" -- python -c \
    'import pathlib,sys; pathlib.Path("/results/metadata.json").write_text(sys.argv[1] + "\n", encoding="utf-8")' \
    "${metadata_json}"

  log "checking benchmark pod runtime dependencies"
  retry_once kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    python -c 'import requests'
  retry_once kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    /bin/bash -ec 'command -v tar >/dev/null'

  log "generating workload inside pod via ${tokenizer_url}"
  retry_once kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    python /opt/benchmark/generate_benchmark_workload.py \
      --profile /opt/benchmark/profile.json \
      --output "${workload_file}" \
      --tokenizer-url "${tokenizer_url}" \
      --model "${MODEL}" \
      --timeout "${TOKENIZER_TIMEOUT:-120}" \
      --tokenizer-concurrency "${TOKENIZER_CONCURRENCY:-16}" \
      "${workload_override_args[@]}"

  local group
  for group in "${groups[@]}"; do
    deploy_proxy_group "${group}"
    local proxy_identity_json
    proxy_identity_json="$(verify_proxy_group "${group}")"
    log "${group}: proxy ready; benchmark pod is starting workload"
    local -a benchmark_command=(
      python /opt/benchmark/benchmark_session_affinity.py
      --base-url "${base_url}"
      --model "${MODEL}"
      --workload-file "${workload_file}"
      --output-dir "/results/${group}"
      --concurrency "${concurrency}"
      --sample-interval "${sample_interval}"
      --expected-decode-count "${decoder_count}"
      --system-warmup-requests "${SYSTEM_WARMUP_REQUESTS:-8}"
      --label "${group}"
      --reset-before
      --verify-reset
      --expected-prefill-count 4
      --reset-drain-timeout "${RESET_DRAIN_TIMEOUT:-120}"
      "${backend_args[@]}"
      "$@"
    )
    if ! kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
      "${benchmark_command[@]}" --proxy-identity-json "${proxy_identity_json}"; then
      if [[ "${RESET_FAILURE_ACTION}" != "restart" ]] || ! kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
        python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); sys.exit(0 if p.is_file() and not json.loads(p.read_text()).get("verified") else 1)' \
        "/results/${group}/reset-validation.json"; then
        return 1
      fi
      log "${group}: reset verification failed; restarting Prefill StatefulSet by request"
      kubectl -n "${NAMESPACE}" rollout restart statefulset/pd-prefill
      kubectl -n "${NAMESPACE}" rollout status statefulset/pd-prefill --timeout=60m
      deploy_proxy_group "${group}"
      proxy_identity_json="$(verify_proxy_group "${group}")"
      log "${group}: Prefillers restarted; retrying benchmark once"
      kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
        "${benchmark_command[@]}" --proxy-identity-json "${proxy_identity_json}"
    fi
    log "completed group=${group}"
  done

  log "comparing baseline, candidate-off, and candidate-on inside benchmark pod"
  retry_once kubectl -n "${NAMESPACE}" exec "${pod_name}" -- \
    python /opt/benchmark/compare_abc_results.py /results

  retry_once copy_pod_results
  kubectl -n "${NAMESPACE}" exec "${pod_name}" -- rm -f "${remote_archive}" || true

  log "rendering HTML report"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/render_benchmark_report.py" "${experiment_dir}"

  if [[ "${KEEP_BENCHMARK_POD:-false}" == "true" ]]; then
    log "keeping benchmark pod=${pod_name} by request"
  else
    log "removing temporary benchmark pod and assets configmap"
    retry_once kubectl -n "${NAMESPACE}" delete "pod/${pod_name}" "configmap/${configmap_name}" \
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
