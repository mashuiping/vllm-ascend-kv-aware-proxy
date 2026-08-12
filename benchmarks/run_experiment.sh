#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 1 )); then
  echo "usage: BASE_URL=... MODEL=... $0 baseline|candidate-off|candidate-on [benchmark arguments]" >&2
  exit 2
fi

GROUP="$1"
shift
: "${BASE_URL:?BASE_URL is required, for example http://127.0.0.1:8000}"
: "${MODEL:?MODEL is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deploy/kubernetes/qwen3-32b-4p4d-tp2"
PYTHON_BIN="${PYTHON_BIN:-python}"
comparison_mode="${ABC_COMPARISON_MODE:-affinity}"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

guard_overload_factor="${AFFINITY_OVERLOAD_FACTOR:-1.5}"
guard_miss_unbind_threshold="${AFFINITY_MISS_UNBIND_THRESHOLD:-3}"
guard_cache_discount_alpha="${AFFINITY_CACHE_DISCOUNT_ALPHA:-0}"
export AFFINITY_OVERLOAD_FACTOR=0
export AFFINITY_MISS_UNBIND_THRESHOLD=0
export AFFINITY_CACHE_DISCOUNT_ALPHA=0

case "${GROUP}" in
  baseline)
    if [[ "${comparison_mode}" == "affinity-guard" ]]; then
      export PROXY_VARIANT=candidate
      export KV_AWARE_ROUTING=true
      export PROXY_SOURCE_PATH="${CANDIDATE_PROXY_SOURCE_PATH:-${REPO_ROOT}/load_balance_proxy_server_example.py}"
    else
      export PROXY_VARIANT=baseline
      export KV_AWARE_ROUTING=false
      export PROXY_SOURCE_PATH="${BASELINE_PROXY_SOURCE_PATH:-${REPO_ROOT}/baseline/load_balance_proxy_server_example.py}"
    fi
    ;;
  candidate-off)
    export PROXY_VARIANT=candidate
    if [[ "${comparison_mode}" == "affinity-guard" ]]; then
      export KV_AWARE_ROUTING=true
    else
      export KV_AWARE_ROUTING=false
    fi
    export PROXY_SOURCE_PATH="${CANDIDATE_PROXY_SOURCE_PATH:-${REPO_ROOT}/load_balance_proxy_server_example.py}"
    if [[ "${comparison_mode}" == "active-token" ]]; then
      export PREFILL_ACTIVE_TOKEN_WEIGHT=0
    fi
    ;;
  candidate-on)
    export PROXY_VARIANT=candidate
    if [[ "${comparison_mode}" == "active-token" ]]; then
      export KV_AWARE_ROUTING=false
    else
      export KV_AWARE_ROUTING=true
    fi
    export PROXY_SOURCE_PATH="${CANDIDATE_PROXY_SOURCE_PATH:-${REPO_ROOT}/load_balance_proxy_server_example.py}"
    if [[ "${comparison_mode}" == "affinity-guard" ]]; then
      export AFFINITY_OVERLOAD_FACTOR="${guard_overload_factor}"
      export AFFINITY_MISS_UNBIND_THRESHOLD="${guard_miss_unbind_threshold}"
      export AFFINITY_CACHE_DISCOUNT_ALPHA="${guard_cache_discount_alpha}"
    fi
    ;;
  *)
    echo "unknown group: ${GROUP}; expected baseline, candidate-off, or candidate-on" >&2
    exit 2
    ;;
esac

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${OUTPUT_DIR:-${RESULTS_DIR:-${REPO_ROOT}/results/runs}/${timestamp}-${GROUP}}"
benchmark_args=(
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --scenario "${SCENARIO:-session-long}"
  --sessions "${SESSIONS:-256}"
  --turns "${TURNS:-6}"
  --concurrency "${CONCURRENCY:-64}"
  --expected-decode-count "${PROXY_DECODER_COUNT:-4}"
  --prefix-words "${PREFIX_WORDS:-1024}"
  --max-tokens "${MAX_TOKENS:-16}"
  --seed "${SEED:-20260724}"
  --label "${GROUP}"
  --output-dir "${output_dir}"
  --reset-before
  --system-warmup-requests "${SYSTEM_WARMUP_REQUESTS:-8}"
)

if [[ -n "${WORKLOAD_FILE:-}" ]]; then
  benchmark_args+=(--workload-file "${WORKLOAD_FILE}")
fi

if [[ -n "${COMPARE_WITH:-}" ]]; then
  benchmark_args+=(--compare-with "${COMPARE_WITH}")
fi

# Only the proxy is restarted. Keeping the P/D topology fixed reduces the
# number of variables between groups; --reset-before clears prefix caches.
log "${GROUP}: deploying proxy variant=${PROXY_VARIANT} kv_aware=${KV_AWARE_ROUTING} active_token_weight=${PREFILL_ACTIVE_TOKEN_WEIGHT:-1.0}"
bash "${DEPLOY_DIR}/deploy.sh" proxy
log "${GROUP}: proxy ready; starting benchmark output=${output_dir}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_session_affinity.py" "${benchmark_args[@]}" "$@"

log "${GROUP}: benchmark complete"
echo "experiment complete: ${output_dir}"
