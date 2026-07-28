
Date: 2026-07-28  
Scope: repository-wide static review of the Python, CUDA/C++, Warp, packaging, and test code  
Constraint: no CUDA environment was available, so GPU findings were established by manual control/data-flow tracing rather than runtime sanitizer evidence

## Executive assessment

PaMO's pipeline and core algorithmic idea are strong, but this repository is not yet production-safe. The main risks are concentrated at the boundaries between research kernels and long-lived application code:

- Stage 2 contains paths that consume uninitialized or stale edge costs.
- The self-intersection checker can silently omit candidates and, when its result buffer overflows, expose an out-of-bounds read to the undo kernel.
- CUDA failures are generally logged and then ignored, so allocation or launch failures can become later memory corruption.
- Stage 3's line search may accept an energy-increasing step, and its CG implementation has unguarded zero divisions and no convergence/breakdown handling.
- Stage 3 eagerly allocates approximately **5.05 GiB** with the default configuration before BVH, mesh, CUDA graph, allocator, and module overhead.
- Both native Python wrappers leak two device buffers when destroyed.

The safest route is not a rewrite of the algorithms. It is to establish explicit ownership, capacity, and solver contracts around them, then replace a few pathological preprocessing and allocation strategies.

## Architecture and execution flow

The public `PaMO` module combines three stages:

1. Stage 1 converts the input mesh to an SDF and reconstructs it with dual marching cubes.
2. Stage 2 repeatedly calls a custom Torch/CUDA extension to perform parallel QEM-style edge collapses, checks self-intersections with an LBVH, and undoes unsafe collapses.
3. Stage 3 copies the result into a preallocated Warp system and runs distance, elasticity, hinge, collision, CCD, CG, and line-search kernels.

There is also a separate Blender-facing C ABI wrapper that includes the stage-2 `.cu` implementation directly and uses a process-global simplifier instance.

The Python API currently hides important state:

- `PaMO` is tied to the constructor's `input_mesh` for stage 3, but `run()` accepts independent points and triangles.
- Resolution, stage-2 allocation high-water marks, stage-3 buffers, CUDA graphs, contact radius, and BVHs all persist across calls.
- The code is presented as an `nn.Module`/`autograd.Function`, although it is not differentiable.

## Critical findings

### P0-1: Stage 2 uses uninitialized or unrelated edge costs

In `compute_edge_cost_kernel`, multiple branches return without writing `edge_cost[edge_index]`:

- the `is_stuck` branch returns immediately;
- the `dup_num != 2` branch returns;
- only some invalid branches explicitly write `UINT32_MAX`.

The buffer is allocated with `cudaMalloc`, not initialized, and is subsequently read by both `propagate_edge_cost_kernel` and `collapse_edge_kernel`. On later calls it may contain a cost for a different edge because the edge list is rebuilt. `original_edge_cost` is copied but never read.

Evidence:

- `simp_cuda/src/cusimp_free.cu:383-445`
- `simp_cuda/src/cusimp_free.cu:588-610`
- `simp_cuda/src/cusimp_free.cu:640-652`
- `simp_cuda/src/cusimp_free.cu:985-993`

Impact: nondeterministic collapses, boundary/non-manifold behavior depending on allocator history, and possible violation of topology and intersection guarantees.

Required fix: initialize every cost to `UINT32_MAX` before the kernel or assign it at the top of every thread, then overwrite only after a candidate has passed every check. If “stuck” mode is intended to reuse prior costs, persist costs by a stable edge key rather than array position.

### P0-2: Intersection result overflow can cause an out-of-bounds GPU read

The self-intersection path allocates `d_intersections` for `2 * num_faces` integers. Every detected intersection increments `d_pos` by two even after the storage cap is reached. Writes are truncated, but the untruncated `d_pos` is copied into `sp->n_intersect`. `get_undo_candidate_kernel` then loops to `*sp.n_intersect` while reading `sp.intersected_triangle_idx`, whose capacity is only approximately `2 * num_faces`.

Evidence:

- `simp_cuda/src/bvh/self_intersect.cuh:239-249`
- `simp_cuda/src/bvh/self_intersect.cuh:380-407`
- `simp_cuda/src/cusimp_free.cu:777-825`

Impact: out-of-bounds device reads on meshes with more recorded intersections than the fixed face-proportional capacity, followed by incorrect or missed undo decisions.

Required fix: maintain separate `total_count` and `stored_count`, clamp all consumers to `stored_count`, expose overflow as a hard failure/retry, and dynamically grow or compact the pair buffer.

### P0-3: The claimed intersection guarantee is limited by silent candidate caps

Each face's BVH overlap query is capped at 512 candidates. The final intersection-pair buffer is also face-proportional and silently truncates writes. A dense, coincident, badly self-intersecting, or adversarial mesh can therefore contain relevant intersections that are never tested or never passed to the undo logic.

Evidence:

- `simp_cuda/src/bvh/self_intersect.cuh:13-15`
- `simp_cuda/src/bvh/self_intersect.cuh:79-83`
- `simp_cuda/src/bvh/self_intersect.cuh:108-112`
- `simp_cuda/src/bvh/self_intersect.cuh:239-245`

Impact: an output may be reported as safe even though not all candidate pairs were examined.

Required fix: use a two-pass count/scan/fill design without a per-face correctness cap. A configurable cap may remain as an explicit resource limit, but overflow must terminate the operation with a diagnostic rather than weaken the guarantee.

### P0-4: CUDA errors are reported but execution continues

Both stage-2 implementations define `CHECK_CUDA` as a boolean-returning logger. Callers discard the result and continue after failed allocations, copies, or frees. Many kernel launches and CUDA calls in the BVH path are not checked at all.

Evidence:

- `simp_cuda/src/cusimp_free.cu:17-27`
- `simp_cuda/src/cusimp.cu:14-24`
- `simp_cuda/src/bvh/self_intersect.cuh:203-249`
- `simp_cuda/src/bvh/self_intersect.cuh:405-414`
- `pamo_blender/cusimpinterface.cu:19-31`

Impact: an ordinary OOM or invalid launch can become a null/stale pointer dereference or corrupt output, while Python receives no exception.

Required fix: use an exception-throwing CUDA check in host code, check every kernel with `cudaGetLastError`, and synchronize only at intentional error/ownership boundaries. Propagate failures as `TORCH_CHECK`/Python exceptions or a structured C ABI status.

### P0-5: Stage 3 can accept a line-search step that increases energy

The line search evaluates a fixed sequence of candidates. A successful earlier candidate is retained, but if all candidates fail, the final candidate remains in `q`; there is no rejection/restoration after the last energy evaluation. There is also no Armijo or directional-derivative condition.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/kernels/solver_kernels.py:61-74`
- `simp_cuda/safe_project/src/pamo_safe_project/system.py:393-410`

Impact: Newton iterations can increase the objective, enter the collision barrier, or amplify a bad CG direction.

Required fix: track acceptance explicitly; after the last attempt restore `q_prev_newton` if no candidate decreases a finite objective. Prefer an Armijo condition and reject non-finite energies.

### P0-6: CG has unguarded breakdown and zero-residual divisions

`update_v_kernel` always computes `zr_new / zr`. This is undefined for an initially solved system, convergence to zero residual, an all-zero/invalid preconditioner result, or numerical underflow. The solver runs exactly 40 iterations and has no residual tolerance, finite check, negative-curvature policy, or reliable breakdown exit. When `v_A_v <= 1e-16`, the state update is skipped but the subsequent ratio is still evaluated.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/kernels/solver_kernels.py:18-36`
- `simp_cuda/safe_project/src/pamo_safe_project/cg_solver.py:46-93`
- `simp_cuda/safe_project/src/pamo_safe_project/cg_solver.py:136-162`

Impact: NaNs on benign stationary cases and unstable behavior when the Hessian is singular or indefinite.

Required fix: implement residual-based termination and explicit `pAp`, `rz`, and finite-value guards. Decide on an indefinite-system strategy: projected/Gauss-Newton Hessians plus PCG, or a solver designed for indefinite systems.

## High-severity findings

### P1-1: Native wrapper destruction leaks device memory

`CUSimp` and `CUSimp_Free` allocate `pts_occ` and `pts_map`, but neither pybind destructor frees them.

Evidence:

- allocation: `simp_cuda/src/cusimp_free.cu:47-84`
- allocation: `simp_cuda/src/cusimp.cu:44-61`
- destructors: `simp_cuda/src/pybind.cpp:21-49`
- destructors: `simp_cuda/src/pybind.cpp:124-138`

Impact: approximately eight bytes per allocated vertex remain on the device each time a wrapper is destroyed, excluding allocator growth padding.

Required fix: give `CUSimp`/`CUSimp_Free` real RAII destructors and delete manual ownership from the pybind wrapper. Use typed device-buffer owners or at least one centralized `release()` routine. Set freed pointers to null.

### P1-2: Default stage-3 construction eagerly reserves about 5.05 GiB

Using the configured maxima and Warp scalar/matrix element sizes, the major arrays account for approximately:

| Area | Approximate allocation |
|---|---:|
| Base system arrays | 0.813 GiB |
| CG arrays | 0.047 GiB |
| Mesh-to-GT arrays | 0.016 GiB |
| Elastic arrays | 0.039 GiB |
| Hinge arrays | 1.759 GiB |
| Collision arrays | 2.375 GiB |
| **Total accounted** | **5.048 GiB** |

This excludes BVHs, Warp meshes, CUDA graphs, compiled modules, allocator fragmentation, stage 1, stage 2, and input/output tensors.

The dominant allocations are:

- collision capacity of `2^25` contacts, including a `(MB, 4)` `vec3` derivative array;
- a `(ME, 4, 4)` matrix-block array for all possible hinge edges;
- GT capacity for `2^24` vertices and roughly twice as many faces.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/config.py:11-15`
- `simp_cuda/safe_project/src/pamo_safe_project/system.py:54-105`
- `simp_cuda/safe_project/src/pamo_safe_project/energy.py:591-605`
- `simp_cuda/safe_project/src/pamo_safe_project/energy.py:709-724`

Impact: avoidable OOM on modest GPUs, large startup latency, and retained high-water memory even for tiny output meshes.

Required fix: size buffers from the registered mesh, grow geometrically on explicit overflow, and allocate optional energy buffers only when the calculator is enabled. Cache by capacity class if reuse is important.

### P1-3: Hinge preprocessing is quadratic in face count

The hinge preprocessor launches over `(n_triangles, n_triangles)` and compares every oriented edge pair.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/energy.py:607-627`
- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/hinge_energy.py:21-58`

Impact: stage 3 degenerates to \(O(F^2)\), inconsistent with the large configured capacities and the pipeline's parallel-performance goals.

Required fix: derive interior-edge adjacency from a sorted/hash edge table in \(O(F \log F)\) or expected \(O(F)\). The `trimesh` preprocessing already computes unique edges on CPU and can provide adjacency for a first robust implementation.

### P1-4: Capacity validation is late, incomplete, or assertion-based

`register_mesh` assigns into fixed slices without first issuing a clear, domain-specific capacity error. `wp_slice` uses `assert`, which is not appropriate for runtime input validation and disappears under optimized Python. There is no up-front check for particles, GT particles, faces, edges, samples, hinge count, or contact count.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/utils.py:15-24`
- `simp_cuda/safe_project/src/pamo_safe_project/system.py:167-205`
- `simp_cuda/safe_project/src/pamo_safe_project/energy.py:629-632`

Impact: failures appear as low-level slice/assignment errors or kernel memory faults rather than actionable messages.

Required fix: validate all mesh-derived counts before any device copy; replace assertions with typed exceptions; expose actual and allowed counts.

### P1-5: Degenerate geometry is warned about after unsafe math is already scheduled

Stage 2 normalizes triangle normals without checking zero area, uses Heron's formula without guarding a small negative radicand, and divides by mesh scale, neighbor edge count, and surviving triangle count. Stage 3 normalizes edges and triangle bases and inverts a rest matrix for triangles it only warns may be tiny.

Evidence:

- `simp_cuda/src/cusimp_free.cu:338-375`
- `simp_cuda/src/cusimp_free.cu:486-585`
- `simp_cuda/safe_project/src/pamo_safe_project/system.py:206-225`
- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/elastic_energy.py:24-38`
- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/hinge_energy.py:125-175`

Impact: NaN/Inf edge costs, elastic data, gradients, and Hessians. Float-to-unsigned conversion of NaN is not a meaningful rejection policy.

Required fix: establish a preprocessing contract that rejects or repairs non-finite vertices, invalid indices, repeated-index faces, zero-area faces, zero-extent meshes, non-manifold edges where unsupported, and inconsistent orientation.

### P1-6: Contact overflow recovery can recurse forever

When contact count exceeds capacity, detection halves the radius and recursively retries. Contacts at or below `d_hat` remain candidates even as the radius tends to zero, so a mesh with more than `max_blocks` near/exact contacts has no terminating success condition.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/energy.py:938-946`
- counter-before-cap behavior: `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/contact_detection.py:151-164`
- counter-before-cap behavior: `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/contact_detection.py:203-216`

Impact: recursion failure, repeated expensive BVH traversal, and no reliable diagnostic.

Required fix: use an iterative bounded retry policy, distinguish total from stored contacts, dynamically grow when feasible, and raise a capacity exception when the irreducible set is too large.

### P1-7: Exact contacts produce singular barrier evaluations

The barrier evaluates `log(d / d_hat)`, `1/d`, and `1/d^2` for `d < d_hat`. At `d == 0`, energy and derivatives become infinite or undefined.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/collision_energy.py:8-29`

Impact: stage 3 cannot robustly accept an already intersecting/coincident input; it can inject Inf/NaN into energy, gradient, Hessian, and CG.

Required fix: explicitly validate the intersection-free precondition, or implement a separate untangling/restoration phase. Do not silently clamp the barrier and claim the same guarantee.

### P1-8: Stage-1 resolution state and normalization are inconsistent across calls

`band` and `margin` are computed once for `R=256`. `run()` preprocesses the mesh with those values, then changes `R` to 128 or 64 based on target size without recomputing the normalization parameters. `R` is only decreased, so a later run requesting a larger target retains the earlier lower resolution.

Evidence:

- `simp_cuda/pamo/__init__.py:48-53`
- `simp_cuda/pamo/__init__.py:60-72`
- `simp_cuda/pamo/__init__.py:102-115`

Impact: repeated runs are history-dependent, and small-target runs mix grid parameters from different resolutions.

Required fix: compute an immutable per-run stage-1 parameter object (`R`, `band`, `margin`) before preprocessing and pass it explicitly through normalization, SDF, reconstruction, and inverse transformation.

### P1-9: Device and stream handling is not safe

Python hard-codes `cuda:0` in several places. The extension verifies only “is CUDA,” allocates and launches through raw CUDA without guarding the input tensor's device, and creates output tensors with generic `torch::kCUDA` options. Destructors synchronize/free on whichever device is current. Raw kernels also do not use PyTorch's current CUDA stream.

Evidence:

- `simp_cuda/pamo/__init__.py:48`
- `simp_cuda/pamo/__init__.py:79-91`
- `simp_cuda/pamo/__init__.py:104`
- `simp_cuda/src/pybind.cpp:54-100`
- `simp_cuda/src/pybind.cpp:124-179`

Impact: wrong-device allocations, cross-device pointer use, unnecessary global synchronization, and races when callers use non-default streams.

Required fix: add `CUDAGuard` for the tensor device, allocate outputs on `points.options()`, launch on `at::cuda::getCurrentCUDAStream()`, and make the Python device explicit/configurable.

### P1-10: Native tensor shape/index contracts are not checked

The extension checks device, contiguity, and scalar type, but not rank or trailing dimension. It reinterprets buffers as packed three-element structs. Triangle values are not checked for range or partial negative indices.

Evidence:

- `simp_cuda/src/pybind.cpp:52-72`
- `simp_cuda/src/pybind.cpp:140-157`
- unchecked indexing begins at `simp_cuda/src/cusimp_free.cu:277-304`

Impact: malformed but contiguous tensors can produce immediate out-of-bounds reads/writes.

Required fix: require exact `[N,3]` shapes, matching devices, non-empty finite points, and indices in `[0,N)` before entering CUDA. Treat `-1` as internal-only, never accepted public input.

### P1-11: The autograd abstraction is misleading and can retain iteration history

The wrappers subclass `torch.autograd.Function`, store input tensors on `ctx`, and provide no `backward`. If called with gradient-tracked inputs, the loop can retain prior tensors through the autograd graph and ultimately cannot backpropagate.

Evidence:

- `simp_cuda/pamo/__init__.py:38-46`
- `simp_cuda/pamo/__init__.py:133`
- `simp_cuda/pamo/__init__.py:188-196`

Impact: avoidable memory growth and an API that advertises semantics it does not implement.

Required fix: call the native object directly under `torch.inference_mode()` and make the public result explicitly non-differentiable, or implement a real backward pass.

### P1-12: Stage 3 rejects meshes with boundary edges through an incorrect hinge count contract

The hinge kernel creates entries only for oppositely oriented shared edges, but preprocessing asserts that hinge count equals all unique edges. Boundary edges are unique edges but are not hinges.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/energy.py:611-632`
- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/hinge_energy.py:32-49`

Impact: stage 3 fails when stage 1 is disabled on an open mesh, and produces an assertion rather than a documented topology error.

Required fix: track `n_hinges` separately from `n_edges`, or explicitly require a closed oriented 2-manifold during input validation.

### P1-13: The elastic linearization and preconditioner are incomplete

The elastic differential kernel has an explicit TODO for the Hessian diagonal, even though Jacobi is the default preconditioner. The HVP includes the StVK term, which may be indefinite away from the rest state, while the solver assumes positive curvature. The deformed local basis is recomputed from `x`, but its variation is not visibly included in the derivative/HVP formulas; this requires finite-difference verification.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/elastic_energy.py:113-145`
- `simp_cuda/safe_project/src/pamo_safe_project/kernels/energy_kernels/elastic_energy.py:173-220`
- `simp_cuda/safe_project/src/pamo_safe_project/cg_solver.py:63-67`

Impact: weak/incorrect preconditioning, possible negative curvature, and potentially inconsistent energy/gradient/HVP triples.

Required fix: add finite-difference and symmetry tests first; then use a consistent projected or Gauss-Newton elastic Hessian and compute its block diagonal.

## Medium-severity and maintainability findings

### P2-1: Stage-2 implementations are substantially duplicated

`cusimp.cu/.h` and `cusimp_free.cu/.h` duplicate structs, allocators, adjacency construction, QEM costs, and collapse logic. The Blender wrapper includes a `.cu` implementation directly. Fixes can easily land in only one path.

Recommendation: keep one templated/core simplifier library with optional safety/undo policies and thin Torch/C ABI adapters. Compile implementation files normally rather than including `.cu` files.

### P2-2: The Blender wrapper is process-global and not thread-safe

`pamo_blender/cusimpinterface.cu:53` declares a global mutable `CUSimp_Free`. Concurrent calls can race over all buffers and counters. Its persistent high-water allocations are never explicitly released before process exit.

Recommendation: expose create/run/destroy handle functions and make ownership per caller.

### P2-3: Optional stages allocate eagerly

`PaMO` always constructs and moves `DMC` to CUDA even when stage 1 is disabled. Stage 3 constructs all maximum-size Warp arrays at object initialization.

Recommendation: initialize each stage lazily on first use and release it explicitly through `close()`/context-manager semantics.

### P2-4: Configuration is mutable but caches do not consistently follow it

`Stage3System.clear()` adds newly configured energy calculators but never removes old ones. `_compute_energy()` iterates all instantiated calculators, while derivatives iterate the current configured classes. A mutated config can therefore produce an objective inconsistent with its gradient.

Evidence:

- `simp_cuda/safe_project/src/pamo_safe_project/system.py:106-113`
- `simp_cuda/safe_project/src/pamo_safe_project/system.py:325-352`

Recommendation: use a validated frozen dataclass and rebuild calculator/capture state when structural configuration changes.

### P2-5: Public API and naming obscure contracts

- ~~`min_verts` actually floors a face target.~~ Renamed to `min_faces`.
- constructor mesh and `run()` tensors can describe different meshes.
- ratio, iteration counts, tolerance, threshold, and empty inputs are not validated.
- unused constructor `scale` divides by zero for a zero-extent mesh.
- imports are duplicated and include unused modules.

Evidence:

- `simp_cuda/pamo/__init__.py:1-15`
- `simp_cuda/pamo/__init__.py:29-36`
- `simp_cuda/pamo/__init__.py:95-103`

Recommendation: expose `simplify(mesh, options) -> Result`, with stage-specific validated option objects and metrics.

### P2-6: Packaging has conflicting compatibility contracts

- root project requires Python 3.12, while `env.yaml` specifies Python 3.10;
- root pins Warp to `>=1.15,<1.16`, while the stage-3 package declares only `>=1.8.1`;
- stage-3 package version is `0.0.0`;
- `simp_cuda/setup.py` falls back to a C++ extension when CUDA is unavailable even though the source still includes CUDA headers/types and does not provide a CPU implementation.

Recommendation: keep one supported environment definition, one compatibility matrix, and fail configuration early with a clear “CUDA build required” message.

### P2-7: Tests do not cover the safety claims

There are only two test files:

- `test_dummy_energy.py` imports an obsolete `stage3` module name, executes at import time instead of defining tests, and requires CUDA.
- `test_spd_project.py` covers the local SPD helper and a small hinge path.

There are no tests for repeated-call memory behavior, boundary meshes, degenerate inputs, candidate/contact overflow, line-search rejection, zero-residual CG, multi-device/stream behavior, stage-1 resolution changes, or the intersection-free guarantee.

There is no CI configuration in the repository. The current local environment also lacks `pytest`, so the suite could not be collected here.

## Recommended target architecture

### 1. A validated mesh boundary

Introduce a `MeshData`/`ValidatedMesh` value object:

- host/device, dtype, shape, contiguity, and index checks;
- finite coordinates and nonzero extent;
- repeated-index and near-zero-area face policy;
- manifold, boundary, orientation, and initial-intersection status;
- normalization transform stored once and inverted once.

Every stage should consume and return this contract, not loose arrays plus implicit state.

### 2. Stage-local option and result objects

Use frozen dataclasses such as `RemeshOptions`, `SimplifyOptions`, and `ProjectionOptions`. Return a `PaMOResult` with mesh, achieved ratio, termination reason, timings, rejected collapse/contact counts, capacity growth, and safety validation result.

### 3. RAII and capacity-aware GPU workspaces

- Native C++ owns all CUDA buffers through move-only RAII wrappers.
- Torch and Blender adapters own native handles, not raw buffers.
- Warp workspaces allocate from actual mesh counts and grow geometrically.
- Every bounded append uses `(required, stored, capacity, overflow)` semantics.
- Destruction and reuse are explicit; no global mutable instance.

### 4. One stage-2 core

Unify safe and non-safe simplifiers behind policies:

- common adjacency/QEM/collapse machinery;
- safety policy supplies intersection detection and undo;
- one stable edge representation;
- deterministic mode for tests;
- a two-pass uncapped candidate pipeline for guaranteed mode.

### 5. A solver with explicit invariants

- objective, gradient, HVP, and diagonal are tested as a consistent bundle;
- finite checks at every Newton boundary;
- PCG convergence/breakdown handling;
- Hessian policy appropriate for PCG;
- Armijo line search with explicit reject;
- collision-free precondition checked before barrier optimization;
- iteration result includes convergence reason.

### 6. Separate library behavior from diagnostics

Replace `print`/device `printf` and global logger mutation with structured diagnostics. Library code should raise typed errors; CLI/demo code decides how to display them.

## Remediation sequence

### Phase A: Safety stopgaps

1. Initialize every edge cost to `UINT32_MAX`.
2. Fix intersection stored-count handling and make all overflow fatal.
3. Make CUDA checks throw/return failure immediately.
4. Add exact tensor shape/index/device validation.
5. Fix line-search rejection and CG zero/breakdown guards.
6. Free `pts_occ` and `pts_map`; centralize native release.
7. Reject degenerate/non-finite input before GPU work.

These should land before performance refactors.

### Phase B: Memory and asymptotic behavior

1. Replace default maximum preallocation with mesh-sized growing buffers. **Done (HANDOFF6):** `auto_capacity=True` default; `Stage3System.ensure_capacity` + per-calculator `ensure_capacity`; contact buffer grows on overflow before radius shrink.
2. Replace quadratic hinge discovery with edge adjacency. **Done (HANDOFF7):** `build_hinge_indices` O(F log F) half-edge table; `HingeEnergyCalculator` uses `n_hinges` for launch dims (P1-12 boundary contract).
3. Replace capped BVH candidate packing with two-pass dynamic packing. **Done (HANDOFF8):** uncapped count/scan/fill; grow `intersect_candidates` to exclusive-scan total; remove `BUFFER_SIZE=512` hard-fail (soft `SELF_X_MAX_TOTAL_SLOTS` resource limit remains).
4. Remove per-call temporary `cudaMalloc/cudaFree` from self-intersection checks. **Done (HANDOFF8):** pooled `si_*` scratch + grow-only collapse/undo buffers on `CUSimp_Free`.
5. Make optional stages and calculators lazy. **Done (HANDOFF8):** `lazy_calculators=True`; PaMO defers DMC + Stage3System to first use.

### Phase C: Architecture cleanup

1. Merge the two stage-2 implementations.
2. Replace autograd wrappers with an honest non-differentiable API.
3. Add device/stream-aware Torch integration.
4. Replace mutable config and implicit reuse with explicit workspace objects.
5. Align packaging, versioning, and supported environment definitions.

## Verification plan for a CUDA machine

The first GPU validation run should be designed to disprove safety, not just reproduce the demo:

1. Build debug kernels with line information and run Compute Sanitizer `memcheck`, `racecheck`, and `initcheck`.
2. Repeatedly create/run/destroy stage-2 wrappers across increasing and decreasing mesh sizes; compare `cudaMemGetInfo` after synchronization and allocator cleanup.
3. Exercise:
   - empty and single-triangle inputs;
   - zero-extent and repeated-vertex meshes;
   - open, non-manifold, inconsistently wound, and heavily self-intersecting meshes;
   - meshes with more than 512 AABB overlaps per face;
   - exact coincident contacts;
   - negative/zero/greater-than-one ratios;
   - repeated small-target then large-target runs;
   - non-default CUDA streams and a nonzero GPU device.
4. For every stage-3 energy, finite-difference-check energy vs. gradient and gradient vs. HVP on small random nondegenerate meshes.
5. Test line search with a deliberately uphill direction and CG with zero residual, singular diagonal, and negative curvature.
6. Validate every produced mesh independently with a CPU exact/self-intersection implementation and topology checks.
7. Add long-running soak tests that assert bounded high-water device memory.

## Review limitations

- Python syntax compilation succeeded for the package and scripts, with one invalid-escape `SyntaxWarning` in `energy.py`.
- Tests were not runnable in this environment because `pytest`, project dependencies, and CUDA are absent.
- GPU race behavior, Warp destruction behavior, and exact kernel numerical output must be confirmed on CUDA.
- Generated symbolic distance derivative files were inspected for call-site guards and singular operations, but not algebraically re-derived term by term. They should be validated by finite differences rather than manual symbolic review.

