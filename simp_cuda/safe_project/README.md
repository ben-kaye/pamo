# Stage 3 (safe project)

GPU safe-projection pass used as PaMO stage 3 (`pamo_safe_project`).

## Dependencies

- **stock `warp-lang`** (PyPI) — do **not** build the old `warp_` Rabbit-Hu submodule
- numpy, scipy, trimesh, libigl
- CUDA-capable GPU (this tree targets **sm_120** / CUDA 12.8)

Hinge Hessian SPD projection used to require a forked Warp (`wp.spd_project_blocks`).
That is now a local `@wp.func_native` extension:

`src/pamo_safe_project/kernels/spd_project_native.py`

wired from `kernels/utils_kernels.py` → `block_spd_project_kernel`.

## Installation

Prefer the repo one-shot (torch 2.8 cu128 + stage 1/2/3):

```bash
# from PaMO repo root
bash setup_pamo.sh
source .venv/bin/activate
```

Standalone editable install of this package only (after `warp-lang` is available):

```bash
cd simp_cuda/safe_project
pip install -e .
```

`setup.py` already depends on `warp-lang>=1.8.1`. No `build_lib.py` / packman.

The empty `warp_/` directory and `.gitmodules` entry are **legacy**; leave them
alone or remove later (see `improvements.md` E3). Do not `git submodule update`
for Warp.

## Running

```python
import pamo_safe_project as stage3

config = stage3.config.Stage3Config()  # default config
system = stage3.system.Stage3System(config)

stage3_V, stage3_F = stage3.process(
    gt_mesh.vertices,
    gt_mesh.faces,
    stage2_mesh.vertices,
    stage2_mesh.faces,
    5,
    system=system,  # if provided, reuse the same system to avoid memory allocation
    config=config,
)
```

## SPD smoke test

```bash
# GPU required
python simp_cuda/safe_project/tests/test_spd_project.py
```

## Scripts

Try processing a single mesh:

```bash
python scripts/try_process.py --id 1087134
python scripts/try_process.py --stage2_dir data/examples/stage2 --gt_dir data/examples/gt --id cubehole
```

Evaluate stage2 outputs (mainly for sanity check):

```bash
python scripts/test_stage2.py
```

Run stage 3 on all meshes:

```bash
python scripts/run_stage3.py --save_mesh
```

Verify that all meshes are intersection-free using CGAL:

```bash
./scripts/cgal_test_all.sh output/stage3_0506_matrix_free/meshes output/log.txt
```

## Notes

- Elasticity
    - [x] Stretching (edge spring, or linear elasticity)
        - [x] Implemented StVK elasticity (needs testing)
    - [x] Bending (vertex curvature by Laplacian)
        - Doesn't work at all
    - [x] Bending (hinge angle)
        - Seems to work
- Distance to target
    - [x] output vertex to GT triangle
    - [x] GT sample to output triangle
- Collision barrier
    - [x] IPC collision barrier
    - [x] CCD
    - [ ] Buffer-free collision barrier energy
        - Needs to re-detect collision pairs at every CCD, compute_energy, compute_diff, compute_hess_dx
        - No need to clamp updates
