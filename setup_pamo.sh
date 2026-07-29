#!/usr/bin/env bash
# Build PaMO stages 1–3 into the repo-root .venv from root pyproject.toml.
# Requires: uv, CUDA toolkit (default: $CUDA_HOME or /usr/local/cuda), sm_120 GPU.
#
# Usage (from repo root):
#   bash setup_pamo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"

echo "=== PaMO setup ==="
echo "  ROOT=$ROOT"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"

# Pure-Python pins + path/git stage packages all live in pyproject.toml.
# no-build-isolation-package is set there so extensions that import torch in
# setup.py build against the venv's torch (not an isolated empty env).
#
# Reinstall native packages every run: uv's wheel cache keys only on source, so a
# torch upgrade can leave a stale ABI-incompatible wheel that fails at import
# with missing c10:: symbols.
uv sync \
    --reinstall-package torchcumesh2sdf \
    --reinstall-package pdmc \
    --reinstall-package pamo \
    --reinstall-package pamo-safe-project

echo "=== Smoke imports ==="
"$VIRTUAL_ENV/bin/python" - <<'PY'
import numpy
import torch
import warp
import pamo
import pamo_safe_project
import pdmc  # noqa: F401
import torchcumesh2sdf  # noqa: F401

print(f"numpy {numpy.__version__}")
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu   {torch.cuda.get_device_name(0)}")
print(f"warp  {warp.__version__}")
print("pamo OK")
print("pamo_safe_project OK")
print("pdmc / torchcumesh2sdf OK")
assert numpy.__version__.startswith("1.26"), f"numpy pin broken: {numpy.__version__}"
assert torch.__version__.startswith("2.8"), f"torch pin broken: {torch.__version__}"
PY

echo "PaMO setup complete (arch=${TORCH_CUDA_ARCH_LIST}, cuda=${CUDA_HOME})."
echo "Activate with: source .venv/bin/activate"
