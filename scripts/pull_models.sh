#!/usr/bin/env bash
# Pull the models required by CodeFlowLens
set -euo pipefail

echo "Pulling qwen3.5:9b ..."
ollama pull qwen3.5:9b

echo "Pulling nomic-embed-text ..."
ollama pull nomic-embed-text

echo "Done. Both models are ready."
