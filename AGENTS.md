# AGENTS.md

Goal: 
make PAMO stable across multiple runs.
Build sm_120 with CU12.8

Fix research students code with hardened cuda code.

## Env setup

Root `pyproject.toml` pins torch 2.8 cu128 + warp-lang. One-shot:

```bash
bash setup_pamo.sh
source .venv/bin/activate
```

Requires `uv`, `CUDA_HOME` (default `/usr/local/cuda`), and `TORCH_CUDA_ARCH_LIST=12.0` (default in script).

## Warp / stage 3

Use stock `warp-lang` from PyPI (see root `pyproject.toml`). Do **not** build the
`simp_cuda/safe_project/warp_` Rabbit-Hu submodule.

The fork only existed for `wp.spd_project_blocks`. That is now a local
`@wp.func_native` extension in
`simp_cuda/safe_project/src/pamo_safe_project/kernels/spd_project_native.py`.
The project is potentially retarded code from a smart person (pre-agent era tbh)

## Known issues (stage 2 is the problem)

Full write-up: `HANDOFF2.md` §3. Repro meshes: `mesh/offending/` +
`scripts/multi_run_harness.py`. All serious defects are in **stage 2** (parallel
edge collapse: `simp_cuda/src/cusimp_free.cu`, wrapper `simp_cuda/pamo/__init__.py`).
Stage 3 is face-count neutral. §3.1 and §3.2 are likely the **same race**.

| ID | Symptom | Severity | Repro mesh(es) |
|----|---------|----------|----------------|
| **§3.1** | Intermittent `cudaErrorIllegalAddress` in `pamo.forward` / thrust sync | crash | `motorcycle_1` lod2 (best), `helmet_1` lod2, `stand_mixer_1` lod1; `archer_1` unconfirmed |
| **§3.2** | Silently wrong face count; same inputs ≠ same output | correctness | **all** meshes jitter ±1–3%; `bust_1` lod2 once returned **33934** faces vs target ~270 (good runs ~290) |
| **§3.3** | Collapse floor: stops short of coarse targets; LODs can be non-monotonic | design limit | `toilet_1` worst (28× lod2); also `stand_mixer_1`, `boat_1`, `tank_1`, `plant_1`, `motorcycle_1`, `helmet_1` |
| **§3.4** | Was `min_verts` (misnamed face floor, default 1e10) — renamed to `min_faces` (default 0) | fixed | use `min_faces` |
| **§4** | CoACD zero-volume sliver (not PaMO) | downstream | `rescue_truck_1` lod0 only |

### Operational notes

- CoACD in the **same process** massively aggravates §3.1. Isolate libraries.
- Process isolation mitigates but does **not** eliminate §3.1 (`motorcycle_1`
  still crashed once in a PaMO-only process).
- Trailing `structs.cuh` / `allocator.cuh` errors at shutdown are **collateral**
  from torchcumesh2sdf destructors on a poisoned context — do not debug there.
- Stage 2 RNG is deterministically seeded; nondeterminism ⇒ data race, not RNG.
- Mitigation for §3.2 (not a fix): bound + re-roll. Bound deliberately loose so
  legitimate §3.3 collapse-floor overshoot still passes, e.g.
  `F <= max(k * target, target + floor_slack)` with `k≈4`.
- Face-target floor is `min_faces` (default 0); do not confuse with vertex count (§3.4).
- `threshold` is not a usable knob for §3.3 (non-monotonic, costs fidelity).
- `compute-sanitizer --tool memcheck` is installed (`/usr/local/cuda-12.8/bin/`)
  but has **not** been run yet on these repros.

### Harness

```bash
python scripts/multi_run_harness.py -i mesh/offending/motorcycle_1.obj --target-faces 1200 -n 10
python scripts/multi_run_harness.py -i mesh/offending/bust_1.obj --lod lod2 -n 20
python scripts/multi_run_harness.py --all-offending --lod lod2 -n 5
```

Experiment inputs under `mesh/offending/` are `*.obj` re-exports of the source
`*.glb` (geometry only; face counts match). Prefer `.obj` for harness runs.
