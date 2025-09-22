#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ "${1:-}" == "--install-command" ]]; then
  TARGET_DIR="$HOME/.local/bin"
  mkdir -p "$TARGET_DIR"
  ln -sf "$REPO_DIR/atlas.sh" "$TARGET_DIR/atlas"
  echo "Created symlink at $TARGET_DIR/atlas"
  echo "Ensure $TARGET_DIR is on your PATH (export PATH=\"$TARGET_DIR:\$PATH\" in your shell profile)."
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  if [ -x "scripts/install_ollama.sh" ]; then
    ./scripts/install_ollama.sh || true
  else
    cat <<'MSG'
WARNING: Ollama does not appear to be installed. Download it from https://ollama.com/download
         and make sure the daemon is running (default: http://localhost:11434).
MSG
  fi
fi

if command -v ollama >/dev/null 2>&1 && [ -x "scripts/pull_models.sh" ]; then
  ./scripts/pull_models.sh || true
fi

if [ ! -d ".venv" ]; then
  echo "Setting up the Atlas environment (first run) ..."
  ./scripts/setup.sh
fi

./scripts/run.sh "$@"
