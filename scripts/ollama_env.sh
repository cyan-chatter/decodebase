#!/usr/bin/env bash
# CodeFlowLens – recommended Ollama environment variables
# Usage: source scripts/ollama_env.sh          # just print
#        source scripts/ollama_env.sh --apply  # write systemd drop-in (requires root)

set -euo pipefail

OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_KEEP_ALIVE=-1
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0

echo "Recommended Ollama environment variables:"
echo "  OLLAMA_NUM_PARALLEL=1"
echo "  OLLAMA_MAX_LOADED_MODELS=2"
echo "  OLLAMA_KEEP_ALIVE=-1"
echo "  OLLAMA_FLASH_ATTENTION=1"
echo "  OLLAMA_KV_CACHE_TYPE=q8_0"

if [[ "${1:-}" == "--apply" ]]; then
    DROP_IN=/etc/systemd/system/ollama.service.d/cfl.conf
    if [[ "$(id -u)" -ne 0 ]]; then
        echo "WARNING: --apply requires root. Run with sudo to write ${DROP_IN}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${DROP_IN}")"
    cat > "${DROP_IN}" <<EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
    echo "Written ${DROP_IN}. Run: systemctl daemon-reload && systemctl restart ollama"
fi
