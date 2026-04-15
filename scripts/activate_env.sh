#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$ROOT/venv/bin/activate" ]; then
  echo "Virtual environment not found. Please run scripts/setup_env.sh first."
  read -r -p "Press Enter to exit"
  exit 1
fi

echo "Opening a new shell session with the virtual environment activated..."
cd "$ROOT" || exit 1
exec "$SHELL" -c "source '$ROOT/venv/bin/activate'; exec $SHELL -i"
