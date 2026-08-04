#!/usr/bin/env bash
set -Eeuo pipefail

PROXY_SCRIPT="${PROXY_SCRIPT:-/opt/pd-proxy/load_balance_proxy_server_example.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROXY_VARIANT="${PROXY_VARIANT:-candidate}"
KV_AWARE_ROUTING="${KV_AWARE_ROUTING:-true}"

if [[ ! -f "${PROXY_SCRIPT}" ]]; then
  echo "proxy script not found: ${PROXY_SCRIPT}" >&2
  exit 2
fi

case "${PROXY_VARIANT}" in
  baseline|candidate) ;;
  *)
    echo "PROXY_VARIANT must be baseline or candidate: ${PROXY_VARIANT}" >&2
    exit 2
    ;;
esac

case "${KV_AWARE_ROUTING}" in
  true|false) ;;
  *)
    echo "KV_AWARE_ROUTING must be true or false: ${KV_AWARE_ROUTING}" >&2
    exit 2
    ;;
esac

# The upstream baseline does not recognize candidate-only flags. Assemble
# those flags here so the same Kubernetes Deployment can run all A/B/C groups
# without maintaining nearly identical proxy manifests.
candidate_args=()
if [[ "${KV_AWARE_ROUTING}" == "true" ]]; then
  if [[ "${PROXY_VARIANT}" != "candidate" ]]; then
    echo "KV-aware routing can only be enabled for the candidate proxy" >&2
    exit 2
  fi
  candidate_args+=(
    --enable-kv-cache-aware-routing
    --session-lru-size "${SESSION_LRU_SIZE:-4096}"
    --prefix-hash-chars "${PREFIX_HASH_CHARS:-1024}"
    --prefix-lru-size "${PREFIX_LRU_SIZE:-1024}"
  )
fi

echo "starting proxy variant=${PROXY_VARIANT} kv_aware=${KV_AWARE_ROUTING} script=${PROXY_SCRIPT}"
if (( ${#candidate_args[@]} > 0 )); then
  exec "${PYTHON_BIN}" "${PROXY_SCRIPT}" "$@" "${candidate_args[@]}"
fi
exec "${PYTHON_BIN}" "${PROXY_SCRIPT}" "$@"
