#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Setting up the virtual environment..."
python3 -m venv venv

./venv/bin/pip install -r requirements.txt

echo "Setup complete."
