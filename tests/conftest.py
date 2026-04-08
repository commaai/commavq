"""Shared pytest setup for the V&V test suite."""

import sys
from pathlib import Path

# Make the repo root importable so tests can do `from utils.vqvae import ...`
# without needing the package to be pip-installed.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
  """Register custom markers so pytest doesn't warn about them."""
  config.addinivalue_line(
    "markers",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
  )
