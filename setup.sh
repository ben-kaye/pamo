#!/usr/bin/env bash
# Install PaMO stages 1–3.
#
# Stage 3 uses stock warp-lang 1.14 from PyPI (no warp_ submodule / build_lib.py).
# SPD projection lives in simp_cuda/safe_project/.../spd_project_native.py.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Stage 1: Remeshing
echo "Installing stage 1: remeshing..."
pip install git+https://github.com/eliphatfs/cumesh2sdf.git
pip install pdmc

# Stage 2: Simplification
echo "Installing stage 2: simplification..."
cd simp_cuda
pip install .
cd ..

# Stage 3: Safe Projection
echo "Installing stage 3: safe projection..."
pip install "warp-lang>=1.14,<1.15"
cd simp_cuda/safe_project
pip install .
cd "$ROOT"

echo "Installation complete"
