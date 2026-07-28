# PaMO improvements plan

Shared plan for making PaMO **buildable and usable** on this machine (RTX 5090 / **sm_120**, **CUDA 12.8**) and **stable across multiple runs**.

**Canonical status:** this file. Sync agents / repro READMEs from here (see D6), not the other way around.

Sources: `AGENTS.md`, `HANDOFF2.md` (defect taxonomy), `HANDOFF3.md` (partial §3.1), `AUDIT.md` (static P0/P1 review), `warp-fork-notes.md`, root `pyproject.toml` + `setup_pamo.sh`.

---

## Goals

| Priority | Goal | Success bar |
|----------|------|-------------|
| P0 | **Build on sm_120 + CU12.8** | Full pipeline imports and runs `example.py` on the 5090 |
| P0 | **Drop Warp fork** | Stage 3 uses stock `warp-lang`; no `build_lib.py` / packman |
| P0 | **Multi-run stability** | Same mesh + args: no crash; face count within a defined bound; no silent “orders of magnitude wrong” LOD |
| P1 | **Reproducible env** | One script (`setup_pamo.sh`) gets a clean machine to green |
| P1 | **Stage-2 / stage-3 hardening** | AUDIT P0s that survive C7 (overflow contracts, CUDA checks, solver safety) |
| P2 | **Code hygiene** | Dead paths, hardcoded devices, stale docs, Blender/docker paths cleaned up |

Non-goals (for now):

- Matching paper timings bit-for-bit on 4090.
- Upstreaming SPD into NVIDIA Warp.
- Making the CUDA simplification kernel fully deterministic (see stability strategy).
- Using `threshold` as a knob for §3.3 collapse floor (non-monotonic, costs fidelity).
- Fixing CoACD zero-volume slivers (§4) inside PaMO.

---

## Constraints (do not break)

1. **Torch `2.8.*` + cu128** — torch 2.11+ breaks `torch::from_blob` for torchcumesh2sdf; 2.8 still builds stage 1 and covers sm_120.
2. **`TORCH_CUDA_ARCH_LIST=12.0`** (or equiv) when compiling native extensions.
3. **No build isolation** for packages that `import torch` in `setup.py` (cumesh2sdf, pdmc, `simp_cuda`).
4. **Stage-1 wheel cache trap** — uv keys cache on source only; after a torch change, force reinstall (see `setup_pamo.sh`) or you get silent ABI mismatch (`c10::` missing at import).
5. **Stock `warp-lang` only** — do not build `simp_cuda/safe_project/warp_` (Rabbit-Hu submodule).
6. **Local tree install** — build against this repo (`$ROOT/simp_cuda`), not a second vendor clone.
7. **CoACD isolation** — do not run CoACD in the same process as PaMO stage 2; same-process use massively aggravates CUDA-context poison (HANDOFF2 §3.1).
8. **`pamo` is non-editable** — after CUDA/C++ edits under `simp_cuda/src/` or `simp_cuda/pamo/`, rebuild:

```bash
export TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda
uv pip install --no-build-isolation --force-reinstall --no-deps ./simp_cuda
```

`pamo_safe_project` **is** editable (source under `simp_cuda/safe_project/src/`).

Hardware reference (this box):

- GPU: RTX 5090 (sm_120)
- Toolkit: CUDA 12.8 (`CUDA_HOME=/usr/local/cuda-12.8`)
- Driver: supports ≥12.x (currently reports 13.2)
- Python: 3.12 + `uv`

---

## Architecture snapshot

```
input mesh
   │
   ▼
[Stage 1] remesh          torchcumesh2sdf + pdmc          (CUDA ext ↔ torch)
   │
   ▼
[Stage 2] simplify        pamo._C (simp_cuda)             (CUDA ext ↔ torch)
   │                       cusimp_free.cu + bvh/self_intersect.cuh
   ▼
[Stage 3] safe project    pamo_safe_project + warp-lang   (Warp kernels)
                              └── SPD: local @wp.func_native
                                  (kernels/spd_project_native.py)
```

Fork reality check (from `warp-fork-notes.md`): the Rabbit-Hu Warp fork only contributed **`spd_project_blocks`** for the hinge Hessian. Everything else PaMO uses is stock Warp API.

All serious multi-run defects are **stage 2**. Stage 3 is face-count neutral (but has its own solver/safety issues in AUDIT).

---

## Workstreams

### A. Environment & build (P0)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| A1 | Root `pyproject.toml`: torch 2.8 cu128, numpy/libigl/trimesh pins, `warp-lang` | **Done** | `warp-lang>=1.15,<1.16`; was briefly under `venv_setup/` — now root only |
| A2 | `setup_pamo.sh`: uv sync + reinstall native packages + smoke imports | **Done** | No warp_ build; reinstalls torchcumesh2sdf/pdmc/pamo/pamo-safe-project |
| A3 | First full build on this machine | **Done** | sm_120 / CU12.8; torch 2.8.0+cu128 |
| A4 | Import smoke: `torch`, `pamo`, `pamo_safe_project`, `warp` | **Done** | cap (12,0); warp 1.15.0 |
| A5 | `example.py` single-mesh run | **Done** | Dumbbell 249k→24.5k faces; stage1+2+3 |
| A6 | Document rebuild recipe when torch changes / after CUDA edits | **Todo** | Root README + setup comments; non-editable `pamo` rebuild (constraint 8) |

### B. Warp: stock + SPD extension (P0)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| B1 | Local SPD via `@wp.func_native` | **Done** | `kernels/spd_project_native.py` |
| B2 | Wire `block_spd_project_kernel` → local func | **Done** | `utils_kernels.py` |
| B3 | `pamo_safe_project` depends on `warp-lang` | **Done** | `setup.py` |
| B4 | Compile/run hinge path on stock Warp | **Done** | warp 1.15 + sm_120; `mat33.data[k][l]` OK |
| B5 | Optional: pure `@wp.func` port if native snippet fights Warp version | **Skipped** | Native path green; keep as fallback only if a future Warp breaks layout |
| B6 | Stop documenting / requiring `warp_` submodule | **Done** | safe_project README + setup no longer build fork; dir may still exist as empty/legacy |
| B7 | Numeric spot-check vs old fork (if available) | **Done** (proxy) | Fork tree empty; NumPy PSD ref rel err ~1e-7; hinge path PSD. See `tests/test_spd_project.py` |

**Decision (current):** prefer `@wp.func_native` over pure Warp for 1:1 fork parity; fall back to pure `@wp.func` if snippet layout breaks across Warp versions. **B4 verified 2026-07-28** — no B5 needed on warp 1.15.

### C. Multi-run stability (P0)

Known defects (full detail: `HANDOFF2.md` §3; repro set `mesh/offending/`; prefer `*.obj`):

| Defect | Symptom | Status | Repro |
|--------|---------|--------|-------|
| **§3.1** | Intermittent CUDA illegal-address / invalid-arg in stage 2 | **Mitigated** (moto crash bar) | was `motorcycle_1` lod2; helmet/mixer/archer not remeasured post-fix |
| **§3.2** | Silent wrong / nondeterministic face counts; `bust_1` lod2 once **33934** vs ~270 | **Open** | all meshes still jitter; catastrophic tail not re-hit post-C7 |
| **§3.3** | Collapse floor overshoots coarse targets; non-monotonic LODs | **Open** (design limit) | `toilet_1` worst (28×); boat/tank/mixer/plant/… |
| **§3.4** | Default face floor was 1e10 under misnamed `min_verts` | **Fixed** | `min_faces` default 0 (`C8`) |
| **§4** | CoACD zero-volume sliver (not PaMO) | **Out of scope** | `rescue_truck_1` lod0 |

**Mental model (updated post-C7):** §3.1 crash path was concrete OOB + multi-iter VRAM leak + undersized candidate buffer — **not** a pure race. §3.2 face jitter may be residual collapse nondeterminism (parallel QEM, stuck path, uninit/stale state) and is **not assumed fixed** by C7. Stage 3 is face-count neutral. CoACD same-process still unwise.

| ID | Task | Status | Notes |
|----|------|--------|-------|
| C1 | Define acceptance bar | **Done** (in harness) | k=4, floor_slack=64; see below |
| C2 | Multi-run harness (N≥10, fixed mesh + ratio) | **Done** | `scripts/multi_run_harness.py`; `--all-offending --lod`; reports crash/overshoot/reroll |
| C3 | Bound + re-roll wrapper around `PaMO.run` | **Done** | `run_with_bound_reroll`; `--reroll R` (overshoot only, not crash recovery) |
| C4 | Seed / config audit (stage3 `seed`, warp, torch) | **Todo** | Won’t fix all nondet; still good hygiene |
| C5 | Identify which stage blows up (1 vs 2 vs 3) | **Done** | Stage 2 only for crash/face; stage 3 neutral |
| C6 | Cap collapse floor / early-exit when stuck | **Todo** | Stage 2 already has `tolerance`; may need harder fail + documented overshoot rate |
| C7 | Root-cause §3.1 crash (compute-sanitizer, fix kernels) | **Done** (moto crash bar) | See status log; helmet/mixer still need C13 |
| C8 | §3.4 default face floor 0 + rename `min_verts`→`min_faces` | **Done** | `pamo/__init__.py` |
| C9 | §3.2 root-cause track (collapse nondeterminism) | **Todo** | Primary suspects: AUDIT P0-1 edge costs (re-verify — tree now writes `COST_INVALID` early), parallel collapse races, stuck-path cost reuse, silent SI under-detection (P0-3). Goal: shrink jitter / eliminate catastrophic tail (`bust_1` 33934) |
| C10 | Intersection pair buffer: stored vs total, hard fail on overflow | **Partial** | AUDIT P0-2; tree already tracks `d_stored`/`d_total` and throws on overflow — re-verify under dense SI; clamp all undo consumers to stored count |
| C11 | Silent per-face `BUFFER_SIZE` candidate cap → hard fail or two-pass | **Todo** | AUDIT P0-3; cap may still miss pairs without diagnostic |
| C12 | CUDA checks that throw / abort (no log-and-continue) | **Todo** | AUDIT P0-4; `CHECK_CUDA` boolean + unchecked BVH launches |
| C13 | Multi-mesh remeasure post-C7 | **Todo** | At least: `motorcycle_1` (baseline), `helmet_1` lod2, `stand_mixer_1` lod1, `bust_1` lod2, `toilet_1` lod2; in-process + `--isolate` |
| C14 | Isolation policy decision | **Todo** | Product call: (a) in-process multi-run must be crash-free, or (b) production callers **must** use process isolation (`--isolate` / HANDOFF2 subprocess pattern). Document choice; do not leave implicit |
| C15 | Sanitizer suite beyond moto memcheck n=1 | **Todo** | memcheck on bust + multi-iter path; `racecheck` / `initcheck` where feasible; record residual rates |

**Acceptance bar (C1):**

For a fixed input mesh and `ratio` (and default stage flags):

1. **No crash / exception** over N consecutive runs (N=20 preferred, N=10 minimum).
2. **Face count** `F` satisfies  
   `target ≤ F ≤ max(k * target, target + floor_slack)`  
   with e.g. `k = 4` and small `floor_slack` for already-coarse meshes (bound deliberately loose so legitimate collapse-floor overshoot still passes).
3. On violation of (2): **re-roll** up to `R` times (e.g. R=3); fail hard only if all attempts overshoot.
4. Harness summary must report: crash rate, F min/max/mean, overshoot rate, reroll rate, `pass_c1`.
5. Optional (P2): Chamfer / Hausdorff to input within a loose ε after stage 3.

Re-roll is a **safety net**, not a substitute for C9. Track overshoot rate; do not declare §3.2 done because re-roll passes.

**Harness (quick ref):**

```bash
python scripts/multi_run_harness.py -i mesh/offending/motorcycle_1.obj --lod lod2 -n 20
python scripts/multi_run_harness.py -i mesh/offending/bust_1.obj --lod lod2 -n 20 --isolate
python scripts/multi_run_harness.py --all-offending --lod lod2 -n 5 --disable_stage3
```

Prefer `mesh/offending/*.obj` (geometry-only re-exports; face counts match GLB).

### D. Env polish & docs (P1)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| D1 | Root README: how to build / rebuild / run demo | **Todo** | Point at `setup_pamo.sh` + constraint 8 rebuild |
| D2 | Align root `README.md` install path with this plan | **Todo** | Conda path can stay as “upstream”; document local path first |
| D3 | Safe-project README: remove warp_ build instructions | **Done** | Point at stock warp + SPD module |
| D4 | `env.yaml` cleanup or mark deprecated | **Todo** | Lists both `warp` and `warp-lang`; misleading |
| D5 | Pin exact warp-lang version once green | **Todo** | A5 already green; pin e.g. `==1.15.0` when ready |
| D6 | Sync stale status docs with this plan | **Todo** | `AGENTS.md`, `mesh/offending/README.md`, optionally HANDOFF3 header — still claim §3.1 unfixed / sanitizer not run / “same race” |

### E. Code hygiene (P2)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| E1 | `pamo/__init__.py`: duplicate imports, hardcoded `cuda:0` | **Todo** | AUDIT P1-9 device/stream |
| E2 | Dead / commented PaSP demo paths | **Todo** | Low value |
| E3 | Remove `warp_` submodule / empty dir from tree | **Todo** | B4 proven; no longer blocked |
| E4 | License note for Burkardt Jacobi (LGPL) in SPD module | **Todo** | Already attributed in file header |
| E5 | Native wrapper destructor leaks (`pts_occ` / `pts_map`) | **Todo** | AUDIT P1-1 |
| E6 | Honest non-autograd API / drop misleading `Function` history | **Todo** | AUDIT P1-11; low urgency |
| E7 | `pamo_blender/` process-global simplifier audit | **Todo** | Separate C ABI; out of multi-run harness scope unless used |
| E8 | `docker/` path: document or deprecate | **Todo** | Exists under `docker/`; not the primary env path |

### F. Stage-3 solver safety (P1 — not face-count)

Stage 3 does not change face counts, but AUDIT flags real quality/crash risks after stage 2.

| ID | Task | Status | Notes |
|----|------|--------|-------|
| F1 | Line search: reject last candidate if energy does not decrease | **Todo** | AUDIT P0-5; restore `q_prev_newton` |
| F2 | CG: residual termination + zero/breakdown guards | **Todo** | AUDIT P0-6; guard `zr_new/zr`, `pAp` |
| F3 | Stage-3 default ~5 GiB eager alloc → mesh-sized grow | **Todo** | AUDIT P1-2; P1 usability |
| F4 | Finite checks / contact overflow recovery | **Todo** | AUDIT P1-6, P1-7 |

---

## AUDIT.md map (do not lose this)

Static review (`AUDIT.md`, 2026-07-28). Several items partially fixed since write; always re-verify against tree before re-implementing.

| AUDIT | Symptom | Plan ID | Notes |
|-------|---------|---------|-------|
| P0-1 | Uninit / stale `edge_cost` | C9 | Tree now assigns `COST_INVALID` at top of `compute_edge_cost_kernel` — re-verify all early-return paths + host reuse |
| P0-2 | SI pair overflow → OOB in undo | C10 | Partial: `d_stored`/`d_total` + throw; confirm consumers |
| P0-3 | Silent `BUFFER_SIZE` candidate cap | C11 | Still a correctness gap for dense SI |
| P0-4 | `CHECK_CUDA` log-and-continue | C12 | |
| P0-5 | Line search accepts uphill step | F1 | |
| P0-6 | CG unguarded ÷0 / no breakdown | F2 | |
| P1-1 | Wrapper device leaks on destroy | E5 | |
| P1-2 | ~5 GiB stage-3 default | F3 | |
| P1-9 / P1-10 | Device/stream / tensor contracts | E1 | |
| P1-11 | Misleading autograd wrapper | E6 | |

AUDIT remediation Phase A (safety stopgaps) ≈ C9–C12 + F1–F2 + E5. Phases B–C (memory asymptotics, architecture rewrite) are out of scope until multi-run P0 is green.

---

## Execution order

```text
Phase 0 ── DONE
  A1 A2 B1 B2 B3

Phase 1 ── DONE (first green path)
  A3 → A4 → B4 → A5   (+ Warp 1.15 port fixes; B6 B7)

Phase 2a ── DONE (crash bar on primary repro)
  C1 → C2 → C5 → C3 → C8 → C7
  Evidence: moto lod2 in-process n=20, 0 crash, pass_c1; memcheck 0

Phase 2b ── CURRENT (correctness + multi-mesh)
  C13 remeasure → C14 isolation policy
  C9 §3.2 root cause → C10/C11/C12 as needed
  C6 collapse-floor policy → C4 seed audit → C15 sanitizer suite
  (re-roll remains safety net; do not stop at re-roll)

Phase 2c ── stage-3 hardening (can parallelize after 2b starts)
  F1 F2; later F3 F4

Phase 3 ── lock in (after C14 decided and C13 green or exceptions documented)
  A6 D1 D5 D2 D6 D4
  optional: E* ; do not polish README as “stable” until 2b status is honest
```

Do **not** polish docs as “fully stable” until Phase 2b has multi-mesh evidence and an isolation policy (C14).

---

## File map

| Path | Role |
|------|------|
| `AGENTS.md` | Short agent goals / hard rules (keep in sync via D6) |
| `improvements.md` | **This plan** (canonical living status) |
| `HANDOFF2.md` | Defect taxonomy §3, CoACD pipeline context |
| `HANDOFF3.md` | Partial §3.1 fix session (superseded by status log for crash rate) |
| `AUDIT.md` | Static P0/P1 review + remediation sequence |
| `warp-fork-notes.md` | Warp-fork analysis (may fold into this later) |
| `pyproject.toml` | UV deps: torch cu128, warp-lang, stage packages |
| `setup_pamo.sh` | One-shot build + smoke imports |
| `uv.lock` | Locked UV resolution |
| `scripts/multi_run_harness.py` | C1/C2/C3 harness (`--isolate`, `--reroll`, `--all-offending`) |
| `mesh/offending/` | Repro meshes + `manifest.json` + README (prefer `*.obj`) |
| `mesh/` | Demo meshes (Dumbbell etc.) |
| `simp_cuda/` | Stage 2 + `pamo` package (non-editable install) |
| `simp_cuda/src/cusimp_free.cu` | Stage-2 collapse core |
| `simp_cuda/src/bvh/self_intersect.cuh` | SI / BVH (C7 fixes) |
| `simp_cuda/safe_project/` | Stage 3 package (editable) |
| `.../kernels/spd_project_native.py` | SPD extension (no Warp fork) |
| `example.py` / `demo.sh` | Smoke targets |
| `env.yaml` | Legacy conda-ish env (D4) |
| `docker/` | Alternate packaging (E8) |
| `pamo_blender/` | Blender C ABI wrapper (E7) |

---

## Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `@wp.func_native` mat33 layout differs across Warp versions | Stage 3 hinge fails to compile | Pure `@wp.func` port (B5); pin Warp version (D5) |
| Stage 1 / 2 don’t emit sm_120 code | Illegal instruction / no kernel image | `TORCH_CUDA_ARCH_LIST=12.0`; verify with small CUDA launch |
| uv wheel cache ABI trap | Import-time `c10` errors after torch bump | reinstall flags in `setup_pamo.sh`; document (A6) |
| Silent bad LOD (§3.2) | Wrong mesh ships downstream | Bound + re-roll (C3) + C9 root cause + harness metrics (C2) |
| Collapse floor (§3.3) | Coarse LODs miss target by large factor | Loose C1 bound; C6 hard fail / document; do not abuse `threshold` |
| Residual crash on non-moto meshes | False confidence from moto-only C7 | C13 remeasure |
| CoACD same process | Inflates §3.1-like poison | Constraint 7; subprocess isolation |
| Stale docs (AGENTS / offending README) | Agents re-debug fixed crashes | D6 |
| Torch >2.8 needed later | Stage 1 broken | Stay on 2.8 until cumesh2sdf `from_blob` fixed upstream |
| AGPL upstream PaMO | Distribution constraints | Keep awareness; local research use OK |
| Ignoring AUDIT after C7 | Overflow / log-and-continue bugs remain | C10–C12, F* |

---

## Definition of done (overall)

1. `bash setup_pamo.sh` succeeds on this machine with CUDA 12.8.
2. `python example.py` produces a mesh under stage 1+2+3 defaults.
3. Multi-run harness (C2) meets the C1 bar for the **offending set** (at least `motorcycle_1`, `bust_1`, `toilet_1`) or documents per-mesh exceptions (especially §3.3 floor).
4. C14 isolation policy is written (in-process required vs subprocess required).
5. No install path depends on building `warp_`.
6. §3.1 crash: moto in-process N≥20 at 0 crashes **and** C13 either green or exceptions documented.
7. §3.2: either root-caused (C9) or catastrophic-tail rate bounded with evidence on `bust_1`; re-roll is not the only control.
8. Remaining open items are only P2 hygiene / optional F3–F4 / documented design limits (§3.3).

---

## Status log

| Date | Note |
|------|------|
| 2026-07-28 | Plan created. SPD `@wp.func_native` + stock warp deps drafted; full build not yet run. |
| 2026-07-28 | Machine verified: 5090, nvcc 12.8, uv 0.10, Python 3.12. |
| 2026-07-28 | **B complete:** stock Warp 1.15 + local SPD on sm_120. `block_spd_project_kernel` compiles; random 9×9 PSD vs NumPy rel err ~1e-7; hinge preprocess/energy/diff/SPD/hess_dx OK. Regression: `simp_cuda/safe_project/tests/test_spd_project.py`. Docs no longer require `warp_`. |
| 2026-07-28 | **Phase 1 green (A3–A5).** Stock Warp 1.15 breakages fixed for stage 3: `wp_slice` uses native slicing (dropped `owner=`); `wp.select`→`wp.where` (arg order flipped) in collision_energy + grad_funcs(+_struct). `example.py` Dumbbell: 249644→24506 faces in ~19s (cold Warp compile). Env lives at **repo root** (`pyproject.toml`, `setup_pamo.sh`) — not `venv_setup/`. |
| 2026-07-28 | **C3/C8 + partial C7.** Harness `--reroll` / `--isolate`; face-floor default 0; later renamed `min_verts`→`min_faces`. Sanitizer: OOB in `self_intersect.cuh` on `V[-1]` for deleted faces — fixed + vector resize. memcheck 0 errors on moto n=1. Residual in-process crashes ~4/15 moto. Full write-up: **`HANDOFF3.md`**. |
| 2026-07-28 | **C7 crash bar met (moto).** After HANDOFF3 residual work: (1) `forward()` re-`cudaMalloc` without free of `collapsed_edge_idx` / `n_collapsed` / `edges_undo` / `n_edges_undo` / `n_intersect` / `intersected_triangle_idx` — multi-iter VRAM leak → intermittent `invalid argument`; free-before-alloc + skip SI when `h_n_collapsed==0`. (2) `intersect_candidates` sized `allocated_tris * BUFFER_SIZE` but each face packs **2×** `num_found` slots — resized to `* BUFFER_SIZE * 2`, per-face cap `BUFFER_SIZE`, grow-after-scan. (3) Query skips deleted partners (`F.i==-1`, stride 3); stack bound in BVH walk. **Evidence:** moto lod2 in-process n=20 → **20/20 OK, 0 crash, 0 overshoot**, faces 2490–3238 (target 1200, bound 4800), `pass_c1=True` (`/tmp/moto_inproc20.json`). Prior post-`-1`-only fix was 11/15. memcheck recheck moto n=1: **0 errors** (`/tmp/sanitizer_moto3.log`). §3.2 face jitter + collapse floor (§3.3) unchanged. |
| 2026-07-28 | **Plan refresh.** Fixed `venv_setup/` path drift → root; added C9–C15, F*, AUDIT map, isolation policy, multi-mesh DoD, rebuild constraint; marked this file canonical over AGENTS/offending README for status. |

---

## Next action

**Phase 2b — in order:**

1. **C13** multi-mesh remeasure (helmet, mixer, bust, toilet; moto as control) in-process and `--isolate`.
2. **C14** write isolation policy from those numbers.
3. **C9** §3.2 root cause (start with re-verifying edge-cost init vs AUDIT P0-1; then parallel collapse / stuck path).
4. **C10–C12 / C15** as evidence demands; **C6** for collapse floor; **C4** seed hygiene.
5. **D6** only after status is stable enough to not thrash AGENTS daily — or do a one-shot “§3.1 mitigated on moto” note now so agents stop re-debugging fixed OOB.

Do **not** declare multi-run done because moto is green. Do **not** treat re-roll as a §3.2 fix.
