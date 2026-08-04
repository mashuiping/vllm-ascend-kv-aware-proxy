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

case "${GROUP}" in
  baseline)
    export PROXY_VARIANT=baseline
    export KV_AWARE_ROUTING=false
    ;;
  candidate-off)
    export PROXY_VARIANT=candidate
    export KV_AWARE_ROUTING=false
    ;;
  candidate-on)
    export PROXY_VARIANT=candidate
    export KV_AWARE_ROUTING=true
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
bash "${DEPLOY_DIR}/deploy.sh" proxy
"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_session_affinity.py" "${benchmark_args[@]}" "$@"

echo "experiment complete: ${output_dir}"
