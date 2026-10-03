#!/usr/bin/env bash
# Pull the models required by CodeFlowLens
set -euo pipefail

echo "Pulling qwen2.5-coder:7b ..."
ollama pull qwen2.5-coder:7b

echo "Pulling nomic-embed-text ..."
ollama pull nomic-embed-text

echo "Done. Both models are ready."
