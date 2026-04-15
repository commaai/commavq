#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONPATH=".:${PYTHONPATH:-}"

if [ -x "venv/bin/python3" ]; then
  PY="venv/bin/python3"
else
  PY="python3"
fi

"$PY" -m pytest tests/test_compressor_config.py \
  --cov=utils.vqvae \
  --cov-report=term-missing \
  --cov-report=html:tests/coverage_report_html \
  --cov-report=xml:tests/coverage.xml
