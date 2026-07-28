#!/usr/bin/env bash
# Legacy install entrypoint. Prefer setup_pamo.sh (uv + torch 2.8 cu128 + sm_120).
#
# This script no longer builds simp_cuda/safe_project/warp_ (Rabbit-Hu fork).
# Stage 3 uses stock warp-lang; SPD is kernels/spd_project_native.py.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/setup_pamo.sh" ]]; then
  echo "Delegating to setup_pamo.sh (recommended)..."
  exec bash "$ROOT/setup_pamo.sh" "$@"
fi

echo "setup_pamo.sh not found; falling back to pip path (no warp_ build)."

# Stage 1: Remeshing
echo "Installing stage 1: remeshing..."
pip install git+https://github.com/eliphatfs/cumesh2sdf.git
pip install pdmc

# Stage 2: Simplification
echo "Installing stage 2: simplification..."
cd simp_cuda
pip install .
cd ..

# Stage 3: Safe Projection (stock warp-lang from PyPI)
echo "Installing stage 3: safe projection..."
pip install "warp-lang>=1.8.1,<1.16"
cd simp_cuda/safe_project
pip install .
cd "$ROOT"

echo "Installation complete"
