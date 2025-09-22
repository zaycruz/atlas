#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m atlas_main.cli "$@"
