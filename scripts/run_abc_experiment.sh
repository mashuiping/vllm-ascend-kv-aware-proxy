#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 1 )); then
  echo "usage: BASE_URL=... MODEL=... TOKENIZER_URL=... $0 benchmarks/profiles/<profile>.json [benchmark arguments]" >&2
  exit 2
fi

PROFILE="$1"
shift
: "${BASE_URL:?BASE_URL is required, for example http://127.0.0.1:8000}"
: "${MODEL:?MODEL is required}"
: "${TOKENIZER_URL:?TOKENIZER_URL must point directly to a vLLM server exposing /tokenize}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${PROFILE}" ]]; then
  echo "workload profile not found: ${PROFILE}" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
experiment_dir="${ABC_RESULTS_DIR:-${REPO_ROOT}/results/runs/${timestamp}-abc}"
workload_file="${experiment_dir}/workload.jsonl"
mkdir -p "${experiment_dir}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_benchmark_workload.py" \
  --profile "${PROFILE}" \
  --output "${workload_file}" \
  --tokenizer-url "${TOKENIZER_URL}" \
  --model "${MODEL}"

profile_concurrency="$("${PYTHON_BIN}" -c \
  'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("load", {}).get("concurrency", 64))' \
  "${PROFILE}")"

read -r -a groups <<<"${GROUP_ORDER:-baseline candidate-off candidate-on}"
for group in "${groups[@]}"; do
  WORKLOAD_FILE="${workload_file}" \
  OUTPUT_DIR="${experiment_dir}/${group}" \
  CONCURRENCY="${CONCURRENCY:-${profile_concurrency}}" \
    bash "${SCRIPT_DIR}/run_experiment.sh" "${group}" "$@"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_abc_results.py" "${experiment_dir}"

echo "A/B/C experiment complete: ${experiment_dir}"
