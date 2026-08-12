#!/usr/bin/env bash
# Re-run the legacy affinity experiments on the hardened A/B/C harness.
#
# The 2026-08-06/07 affinity runs predate proxy-identity verification, the
# candidate-off noise control and the affinity stats instrumentation, so they
# cannot be judged by the same standard as the active-token analysis. This
# script produces six verifiable repetitions per profile with rotated group
# orders (full cyclic Latin square) and distinct workload seeds, then applies
# the noise-gate verdict.
#
# Judgment standard (same as docs/analysis-active-token-weight.md):
#   1. Every repetition must be valid (proxy identity + reset checks pass).
#   2. B (candidate-off) versus A is the same-policy control; its per-run
#      |delta| defines the noise band for each metric.
#   3. Claim an effect only when |C - A| clears the same-run noise band in a
#      consistent direction across repetitions, and the paired bootstrap CI
#      from compare_repetitions.py excludes zero.
#
# Usage:
#   MODEL=qwen3-32b PREFILL_NODE=<node> \
#     bash benchmarks/run_affinity_repetitions.sh \
#       [benchmarks/profiles/session-affinity.json ...]
#
# Optional:
#   SEED_BASE=20260812      # seeds are SEED_BASE*10+1 .. +N
#   REPETITIONS=6           # 3 = first Latin-square half, 6 = full square
#   ABC_COMPARISON_MODE=affinity|affinity-guard (default: profile default)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED_BASE="${SEED_BASE:-20260812}"
REPETITIONS="${REPETITIONS:-6}"

if (( REPETITIONS < 1 || REPETITIONS > 6 )); then
  echo "REPETITIONS must be between 1 and 6" >&2
  exit 2
fi

PROFILES=("$@")
if (( ${#PROFILES[@]} == 0 )); then
  PROFILES=(
    "${SCRIPT_DIR}/profiles/session-affinity.json"
    "${SCRIPT_DIR}/profiles/shared-prefix-capacity.json"
  )
fi

# Full cyclic Latin square plus reversals: every group occupies every
# execution position, forwards and backwards.
GROUP_ORDERS=(
  "baseline candidate-off candidate-on"
  "candidate-off candidate-on baseline"
  "candidate-on baseline candidate-off"
  "baseline candidate-on candidate-off"
  "candidate-on candidate-off baseline"
  "candidate-off baseline candidate-on"
)

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

for profile in "${PROFILES[@]}"; do
  if [[ ! -f "${profile}" ]]; then
    echo "profile not found: ${profile}" >&2
    exit 2
  fi
done

declare -a all_run_dirs=()
for profile in "${PROFILES[@]}"; do
  profile_name="$(basename "${profile}" .json)"
  run_dirs=()
  for ((index = 0; index < REPETITIONS; index++)); do
    seed=$((SEED_BASE * 10 + index + 1))
    order="${GROUP_ORDERS[index]}"
    run_dir="${REPO_ROOT}/results/runs/$(date -u +%Y%m%dT%H%M%SZ)-abc"
    log "profile=${profile_name} repetition=$((index + 1))/${REPETITIONS} seed=${seed} order='${order}'"
    ABC_RESULTS_DIR="${run_dir}" \
    WORKLOAD_SEED="${seed}" \
    GROUP_ORDER="${order}" \
      bash "${SCRIPT_DIR}/run_abc_experiment.sh" "${profile}"
    run_dirs+=("${run_dir}")
    all_run_dirs+=("${run_dir}")
  done

  # bash 3.2 (macOS default) rejects negative array subscripts.
  last_run_dir="${run_dirs[$((${#run_dirs[@]} - 1))]}"
  log "aggregating ${#run_dirs[@]} repetitions for ${profile_name}"
  for comparison in A_vs_B A_vs_C; do
    aggregate_out="${last_run_dir}/repetitions-${profile_name}-${comparison}.json"
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/compare_repetitions.py" \
      --comparison "${comparison}" \
      "${run_dirs[@]}" \
      > "${aggregate_out}"; then
      rm -f "${aggregate_out}"
      log "WARNING: ${comparison} aggregation failed for ${profile_name}; inspect the runs above"
    fi
  done
  "${PYTHON_BIN}" "${SCRIPT_DIR}/judge_affinity_repetitions.py" "${run_dirs[@]}" \
    | tee "${last_run_dir}/noise-gate-verdict-${profile_name}.txt"
done

log "all repetitions complete"
for run_dir in "${all_run_dirs[@]}"; do
  log "run: ${run_dir}"
done
