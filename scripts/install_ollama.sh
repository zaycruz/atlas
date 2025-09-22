#!/usr/bin/env bash
set -euo pipefail

if command -v ollama >/dev/null 2>&1; then
  exit 0
fi

echo "Ollama is not installed."
read -r -p "Would you like to install it automatically now? [y/N] " choice
case "$choice" in
  y|Y|yes|YES) ;;
  *)
    cat <<'MSG'
Please install Ollama manually from https://ollama.com/download and make sure it is running.
Once installed, rerun this script.
MSG
    exit 1
    ;;
esac

echo "Attempting automatic installation..."
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [[ "$OS_NAME" == "darwin" ]];
then
  if command -v brew >/dev/null 2>&1; then
    brew update && brew install ollama
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
elif [[ "$OS_NAME" == "linux" ]]; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  cat <<'MSG'
Automatic Ollama installation is not supported on this OS via shell script.
Please install it manually from https://ollama.com/download and ensure it is running.
MSG
  exit 1
fi

echo "Ollama installation complete."
