"""SPD block projection on stock warp-lang (no Rabbit-Hu fork).

Verifies ``spd_project_blocks`` / ``block_spd_project_kernel`` compile and match
a NumPy PSD reference, then exercises the hinge Hessian path end-to-end.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False})

from pamo_safe_project.kernels.utils_kernels import block_spd_project_kernel
from pamo_safe_project.kernels.energy_kernels.hinge_energy import (
    hinge_preprocess_slow_kernel,
    hinge_energy_kernel,
    hinge_diff_kernel,
    hinge_hess_dx_kernel,
)


DEVICE = "cuda:0"


def _pack_9x9_to_blocks(M9: np.ndarray) -> np.ndarray:
    blocks = np.zeros((1, 4, 4, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            blocks[0, i, j] = M9[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
    return blocks


def _unpack_blocks_to_9x9(blocks: np.ndarray) -> np.ndarray:
    M = np.zeros((9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = blocks[0, i, j]
    return M


def test_spd_project_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((9, 9)).astype(np.float64)
    A = 0.5 * (A + A.T)

    blocks = wp.array(_pack_9x9_to_blocks(A.astype(np.float32)), dtype=wp.mat33, device=DEVICE)
    wp.launch(block_spd_project_kernel, dim=1, inputs=[blocks, 100], device=DEVICE)
    wp.synchronize()

    M_out = _unpack_blocks_to_9x9(blocks.numpy())
    evals_out = np.linalg.eigvalsh(M_out)
    assert evals_out.min() >= -1e-4, f"not PSD: min eval {evals_out.min()}"

    evals, evecs = np.linalg.eigh(A)
    M_ref = evecs @ np.diag(np.maximum(evals, 0.0)) @ evecs.T
    rel = np.linalg.norm(M_out - M_ref) / (np.linalg.norm(M_ref) + 1e-12)
    assert rel < 1e-3, f"rel frobenius error {rel}"

    # Kernel fills row/col 3 as linear dependence on the top-left 3x3 blocks.
    b = blocks.numpy()[0]
    s = -(b[0, 0] + b[1, 0] + b[2, 0])
    assert np.allclose(b[3, 0], s, atol=1e-6)


def test_hinge_path_spd():
    verts = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0.2, 0.2, 1],
        ],
        dtype=np.float32,
    )
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    x_rest = wp.array(verts, dtype=wp.vec3, device=DEVICE)
    triangles = wp.array(tris, dtype=wp.int32, device=DEVICE)
    hinge_counter = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    max_h = 8
    hinge_indices = wp.zeros((max_h, 4), dtype=wp.int32, device=DEVICE)
    rest_angles = wp.zeros(max_h, dtype=wp.float32, device=DEVICE)
    rest_elens = wp.zeros(max_h, dtype=wp.float32, device=DEVICE)

    wp.launch(
        hinge_preprocess_slow_kernel,
        dim=(2, 2),
        inputs=[x_rest, triangles, hinge_counter, hinge_indices, rest_angles, rest_elens],
        device=DEVICE,
    )
    wp.synchronize()
    n_h = int(hinge_counter.numpy()[0])
    assert n_h == 1

    verts2 = verts.copy()
    verts2[3, 2] += 0.3
    x = wp.array(verts2, dtype=wp.vec3, device=DEVICE)
    energy = wp.zeros(1, dtype=wp.float32, device=DEVICE)
    wp.launch(
        hinge_energy_kernel,
        dim=n_h,
        inputs=[x, hinge_indices, rest_angles, rest_elens, np.float32(1e-2), energy],
        device=DEVICE,
    )

    blocks = wp.zeros((n_h, 4, 4), dtype=wp.mat33, device=DEVICE)
    grad = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
    hess_diag = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
    wp.launch(
        hinge_diff_kernel,
        dim=n_h,
        inputs=[
            x,
            hinge_indices,
            rest_angles,
            rest_elens,
            np.float32(1e-2),
            np.float32(1.0),
        ],
        outputs=[blocks, grad, hess_diag],
        device=DEVICE,
    )
    wp.launch(
        block_spd_project_kernel,
        dim=n_h,
        inputs=[blocks, 3],
        device=DEVICE,
    )
    wp.synchronize()

    bnp = blocks.numpy()
    H = np.zeros((12, 12), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            H[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = bnp[0, i, j]
    evals_h = np.linalg.eigvalsh(0.5 * (H + H.T))
    assert evals_h.min() >= -1e-3, f"hinge Hessian not PSD: {evals_h.min()}"

    dx = wp.array(
        np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32) * 0.01,
        dtype=wp.vec3,
        device=DEVICE,
    )
    hess_dx = wp.zeros(4, dtype=wp.vec3, device=DEVICE)
    wp.launch(
        hinge_hess_dx_kernel,
        dim=n_h,
        inputs=[hinge_indices, blocks, dx],
        outputs=[hess_dx],
        device=DEVICE,
    )
    wp.synchronize()
    assert np.isfinite(hess_dx.numpy()).all()
    assert np.isfinite(energy.numpy()).all()


if __name__ == "__main__":
    test_spd_project_matches_numpy()
    print("test_spd_project_matches_numpy OK")
    test_hinge_path_spd()
    print("test_hinge_path_spd OK")
