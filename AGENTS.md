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
| **§3.1** | Intermittent `cudaErrorIllegalAddress` in `pamo.forward` / thrust sync | crash (improved) | `motorcycle_1` lod2 best; HANDOFF5: **0 illegal-address** in moto n≈19 clean + 8-mesh×2 breadth after Phase A; still isolate vs CoACD |
| **§3.2** | Silently wrong face count; same inputs ≠ same output | correctness | **all** meshes jitter ±1–3%; `bust_1` lod2 once returned **33934** faces vs target ~270 (good runs ~290) |
| **§3.3** | Collapse floor: stops short of coarse targets; LODs can be non-monotonic | design limit | `toilet_1` worst (28×+ lod2); also `stand_mixer_1`, `boat_1`, `tank_1`, `plant_1`, `motorcycle_1`, `helmet_1` |
| **§3.4** | Was `min_verts` (misnamed face floor, default 1e10) — renamed to `min_faces` (default 0) | fixed | use `min_faces` |
| **§4** | CoACD zero-volume sliver (not PaMO) | downstream | `rescue_truck_1` lod0 only |

Phase A safety stopgaps landed (`HANDOFF4`, commit `0a54fc9`). Measure notes: `HANDOFF5`.  
**Phase B item 1** (mesh-sized growing stage-3 buffers): landed — see `HANDOFF6.md`. Stage 3 no longer eagerly reserves ~5 GiB.  
**Phase B item 2** (O(F log F) hinge adjacency): landed — see `HANDOFF7.md`. Replaces O(F²) face-pair hinge preprocess; launch dims use `n_hinges` (boundary-safe).  
**Phase B items 3–5** (two-pass SI packing, SI pool, lazy stages/calcs): landed — see `HANDOFF8.md`. Rebuild stage-2 with `uv pip install -e simp_cuda --no-build-isolation`.

### Operational notes

- CoACD in the **same process** massively aggravates §3.1. Isolate libraries.
- Process isolation still recommended in production; in-process crash rate improved
  post–Phase A but was never proven zero forever.
- Trailing `structs.cuh` / `allocator.cuh` errors at shutdown are **collateral**
  from torchcumesh2sdf destructors on a poisoned context — do not debug there.
- Stage 2 RNG is deterministically seeded; nondeterminism ⇒ data race, not RNG.
- Mitigation for §3.2 (not a fix): bound + re-roll. Bound deliberately loose so
  legitimate §3.3 collapse-floor overshoot still passes, e.g.
  `F <= max(k * target, target + floor_slack)` with `k≈4`.
- Face-target floor is `min_faces` (default 0); do not confuse with vertex count (§3.4).
- `threshold` is not a usable knob for §3.3 (non-monotonic, costs fidelity).
- Prefer harness **breadth** (many meshes × small n) over moto×20 soaks.
  Prefer `--disable_stage3` when measuring stage-2 crashes (isolates stage 2).
  Full pipeline with stage 3 is now practical (auto capacity, not ~5 GiB).
- `compute-sanitizer --tool memcheck` post–Phase A on motorcycle stage-2 n=1:
  **0 errors** (HANDOFF5). Racecheck optional for §3.2.
- Stage 3 config: `auto_capacity=True` (default) sizes buffers from the registered
  mesh and grows on contact overflow; set `False` for legacy full `max_*` prealloc.
  Hard ceilings remain `max_particles` / `max_blocks` / `max_gt_*`.
- Stage 3 hinges: `build_hinge_indices` (CPU sorted half-edges) + `hinge_fill_rest_kernel`.
  Legacy `hinge_preprocess_slow_kernel` kept for parity tests only.

### Harness

```bash
# fast green check (prefer this)
for m in archer_1 boat_1 bust_1 helmet_1 plant_1 stand_mixer_1 toilet_1; do
  python scripts/multi_run_harness.py -i mesh/offending/${m}.obj --lod lod2 -n 2 --disable_stage3
done
# §3.1 deep repro only when needed
python scripts/multi_run_harness.py -i mesh/offending/motorcycle_1.obj --target-faces 1200 -n 10 --disable_stage3
```

Experiment inputs under `mesh/offending/` are `*.obj` re-exports of the source
`*.glb` (geometry only; face counts match). Prefer `.obj` for harness runs.
