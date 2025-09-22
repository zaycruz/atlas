#!/usr/bin/env bash
set -euo pipefail

: "${ATLAS_CHAT_MODEL:=qwen2.5:latest}"
: "${ATLAS_EMBED_MODEL:=mxbai-embed-large}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama command not found; skipping model pulls." >&2
  exit 0
fi

pull_model() {
  local model="$1"
  if ollama show "$model" >/dev/null 2>&1; then
    echo "Model '$model' already present."
    return
  fi
  echo "Pulling Ollama model '$model'..."
  if ! ollama pull "$model"; then
    echo "Warning: failed to pull model '$model'." >&2
  fi
}

pull_model "$ATLAS_CHAT_MODEL"
pull_model "$ATLAS_EMBED_MODEL"
