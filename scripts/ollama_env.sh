#!/usr/bin/env bash
# Print settings; quantization is OFF unless opted into via config or --kv-quantization.
# eval "$(scripts/ollama_env.sh --kv-quantization)"; then restart Ollama.
# --apply writes the existing systemd drop-in and requires root.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_APPLY=0
TASK_ARGS=()
for TASK_ARG in "$@"; do
    if [[ "${TASK_ARG}" == "--apply" ]]; then
        TASK_APPLY=1
    else
        TASK_ARGS+=("${TASK_ARG}")
    fi
done
if [[ "${TASK_APPLY}" == 1 ]]; then
    TASK_DROP_IN=/etc/systemd/system/ollama.service.d/cfl.conf
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "--apply requires root to write ${TASK_DROP_IN}" >&2
        exit 1
    fi
    TASK_CONFIG="$("${TASK_ROOT}/.venv/bin/cfl" runtime-env "${TASK_ARGS[@]}" --format systemd)"
    mkdir -p "$(dirname -- "${TASK_DROP_IN}")"
    printf '%s\n' "${TASK_CONFIG}" > "${TASK_DROP_IN}"
    echo "Written ${TASK_DROP_IN}. Run systemctl daemon-reload and restart ollama to apply."
else
    "${TASK_ROOT}/.venv/bin/cfl" runtime-env "${TASK_ARGS[@]}"
fi
