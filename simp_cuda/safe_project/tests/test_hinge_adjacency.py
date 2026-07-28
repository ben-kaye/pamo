"""Phase B #2: O(F log F) hinge adjacency vs legacy O(F^2) kernel."""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False})

from pamo_safe_project.kernels.energy_kernels.hinge_energy import (
    build_hinge_indices,
    hinge_preprocess_slow_kernel,
    hinge_fill_rest_kernel,
)


DEVICE = "cuda:0"


def _sort_hinges(h: np.ndarray) -> np.ndarray:
    if h.size == 0:
        return h.reshape(0, 4)
    # Canonical row order: shared edge (v_lo, v_hi), then opposites.
    keys = np.lexsort((h[:, 3], h[:, 0], h[:, 2], h[:, 1]))
    return h[keys]


def _slow_hinges(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_f = tris.shape[0]
    max_h = max(n_f * 3, 8)
    x_rest = wp.array(verts.astype(np.float32), dtype=wp.vec3, device=DEVICE)
    triangles = wp.array(tris.astype(np.int32), dtype=wp.int32, device=DEVICE)
    hinge_counter = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    hinge_indices = wp.zeros((max_h, 4), dtype=wp.int32, device=DEVICE)
    rest_angles = wp.zeros(max_h, dtype=wp.float32, device=DEVICE)
    rest_elens = wp.zeros(max_h, dtype=wp.float32, device=DEVICE)
    wp.launch(
        hinge_preprocess_slow_kernel,
        dim=(n_f, n_f),
        inputs=[x_rest, triangles, hinge_counter, hinge_indices, rest_angles, rest_elens],
        device=DEVICE,
    )
    wp.synchronize()
    n_h = int(hinge_counter.numpy()[0])
    return (
        hinge_indices.numpy()[:n_h].copy(),
        rest_angles.numpy()[:n_h].copy(),
        rest_elens.numpy()[:n_h].copy(),
    )


def test_two_face_hinge():
    tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    h = build_hinge_indices(tris)
    assert h.shape == (1, 4)
    # shared edge 1--2 with 1 < 2; opposites 0 and 3
    assert list(h[0]) == [0, 1, 2, 3]


def test_boundary_no_extra_hinge():
    # Single triangle: three boundary edges, zero hinges.
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    h = build_hinge_indices(tris)
    assert h.shape == (0, 4)


def test_tetrahedron_parity_with_slow():
    # Closed tet: 4 faces, 6 edges, all interior hinges.
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
        ],
        dtype=np.float32,
    )
    tris = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )

    fast = _sort_hinges(build_hinge_indices(tris))
    slow, rest_a_slow, rest_e_slow = _slow_hinges(verts, tris)
    slow = _sort_hinges(slow)

    assert fast.shape[0] == 6, f"expected 6 tet hinges, got {fast.shape[0]}"
    assert slow.shape[0] == 6
    np.testing.assert_array_equal(fast, slow)

    # Rest fill from fast indices matches slow rest values (same order after sort).
    max_h = 8
    x_rest = wp.array(verts, dtype=wp.vec3, device=DEVICE)
    hinge_indices = wp.zeros((max_h, 4), dtype=wp.int32, device=DEVICE)
    rest_angles = wp.zeros(max_h, dtype=wp.float32, device=DEVICE)
    rest_elens = wp.zeros(max_h, dtype=wp.float32, device=DEVICE)
    hinge_indices.assign(np.vstack([fast, np.zeros((max_h - 6, 4), dtype=np.int32)]))
    wp.launch(
        hinge_fill_rest_kernel,
        dim=6,
        inputs=[x_rest, hinge_indices, rest_angles, rest_elens],
        device=DEVICE,
    )
    wp.synchronize()

    # Reorder slow rest by same hinge sort order.
    # Recompute slow in sorted hinge order via dict.
    slow_raw, ra, re = _slow_hinges(verts, tris)
    key = lambda row: (int(row[1]), int(row[2]), int(row[0]), int(row[3]))
    slow_map = {key(slow_raw[i]): (ra[i], re[i]) for i in range(len(slow_raw))}
    ra_fast = rest_angles.numpy()[:6]
    re_fast = rest_elens.numpy()[:6]
    for i in range(6):
        ra_s, re_s = slow_map[key(fast[i])]
        assert abs(ra_fast[i] - ra_s) < 1e-5, (i, ra_fast[i], ra_s)
        assert abs(re_fast[i] - re_s) < 1e-5, (i, re_fast[i], re_s)


def test_larger_random_mesh_parity():
    rng = np.random.default_rng(0)
    # Build a small grid-like strip (open) and a closed-ish fan.
    # Two-sided strip of quads split into tris: has interior hinges + boundary.
    verts = []
    for y in range(4):
        for x in range(5):
            verts.append([x, y, 0.05 * rng.random()])
    verts = np.asarray(verts, dtype=np.float32)
    tris = []
    def vid(x, y):
        return y * 5 + x
    for y in range(3):
        for x in range(4):
            v00, v10 = vid(x, y), vid(x + 1, y)
            v01, v11 = vid(x, y + 1), vid(x + 1, y + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    tris = np.asarray(tris, dtype=np.int32)

    fast = _sort_hinges(build_hinge_indices(tris))
    slow, _, _ = _slow_hinges(verts, tris)
    slow = _sort_hinges(slow)
    assert fast.shape == slow.shape, (fast.shape, slow.shape)
    np.testing.assert_array_equal(fast, slow)


if __name__ == "__main__":
    test_two_face_hinge()
    print("test_two_face_hinge OK")
    test_boundary_no_extra_hinge()
    print("test_boundary_no_extra_hinge OK")
    test_tetrahedron_parity_with_slow()
    print("test_tetrahedron_parity_with_slow OK")
    test_larger_random_mesh_parity()
    print("test_larger_random_mesh_parity OK")
