#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required but was not found. Install Python 3.9 or newer first." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating local virtual environment (.venv)..."
  python3 -m venv .venv
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "\nSetup complete!"
echo "Run the assistant with: ./scripts/run.sh"
