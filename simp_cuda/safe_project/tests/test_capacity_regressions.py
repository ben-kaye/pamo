"""Regressions for stage-3 auto_capacity edge / contact sizing (P1/P2)."""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False})

from pamo_safe_project.config import Stage3Config
from pamo_safe_project.energy import HingeEnergyCalculator
from pamo_safe_project.system import Stage3System


DEVICE = "cuda:0"


def _two_face_mesh():
    """Two triangles sharing one edge (manifold: 1 hinge, 5 unique edges? wait 5?).

    Faces: (0,1,2) and (1,0,3) share edge 0-1 → 1 hinge, 5 unique edges? 
    Edges: 01,12,20,10,03,31 → unique: 01,12,20,03,31 = 5 edges, 1 hinge.
    """
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
    return V, F


def _dense_nonmanifold(n_pairs: int = 20):
    """Many faces share one undirected edge with opposite winding.

    ``n_pairs`` faces in each orientation → ``n_pairs**2`` hinges on edge 0-1,
    while unique edges are only ``1 + 4*n_pairs``. With n_pairs=20 that is
    400 hinges vs ~81 edges, exceeding auto_capacity edge headroom so the
    hinge path must grow *hinge* buffers without touching mesh edges.
    """
    V = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    F = []
    for i in range(n_pairs):
        V.append([0.5, float(i), 1.0])
        F.append([0, 1, 2 + i])
    for i in range(n_pairs):
        V.append([0.5, float(i), -1.0])
        F.append([1, 0, 2 + n_pairs + i])
    return np.asarray(V, dtype=np.float64), np.asarray(F, dtype=np.int32)


def test_contact_hint_clamped_to_max_blocks():
    """P2: initial contact heuristic must not exceed max_blocks."""
    c = Stage3Config()
    c.auto_capacity = True
    c.lazy_calculators = True
    c.max_blocks = 1024
    c.min_blocks = 1 << 16  # larger than max_blocks
    c.blocks_per_edge = 16
    c.energy_calcs = [HingeEnergyCalculator]  # avoid collision/BVH path
    c.max_gt_samples = 64
    c.use_cuda_graph = False

    V, F = _two_face_mesh()
    s = Stage3System(config=c, device=DEVICE)
    s.register_mesh(V, F, V, F)
    assert s.cap_blocks <= c.max_blocks, (
        f"cap_blocks={s.cap_blocks} exceeded max_blocks={c.max_blocks}"
    )
    assert s.cap_blocks == c.max_blocks
    print("test_contact_hint_clamped_to_max_blocks OK")


def test_hinge_growth_does_not_zero_mesh_edges():
    """P1: hinge preprocess must not reallocate/zero mesh edges after upload."""
    c = Stage3Config()
    c.auto_capacity = True
    c.lazy_calculators = True
    c.energy_calcs = [HingeEnergyCalculator]
    c.max_gt_samples = 64
    c.use_cuda_graph = False
    # Tiny max_blocks so registration is light.
    c.max_blocks = 4096
    c.min_blocks = 64

    V, F = _dense_nonmanifold(20)
    s = Stage3System(config=c, device=DEVICE)
    # Capacity sized for unique edges only; upload + edge BVH happen in
    # register_mesh *before* hinge preprocess.
    s.register_mesh(V, F, V, F)

    edges_before = s.edges.numpy()[: s.n_edges].copy()
    lowers_before = s.edge_lowers.numpy()[: s.n_edges].copy()
    cap_edges_before = s.cap_edges
    n_edges_before = s.n_edges

    ec = s.energy_calcs[HingeEnergyCalculator]
    assert ec.n_hinges == 400, f"expected 400 non-manifold hinges, got {ec.n_hinges}"
    assert ec.n_hinges > cap_edges_before, (
        "test mesh must force hinge capacity past mesh edge capacity"
    )
    # Hinge buffers grew; mesh edge capacity and contents must not change.
    assert ec._cap_edges >= 400
    assert s.cap_edges == cap_edges_before
    assert s.n_edges == n_edges_before
    np.testing.assert_array_equal(
        s.edges.numpy()[: s.n_edges],
        edges_before,
        err_msg="mesh edges were clobbered by hinge preprocess growth",
    )
    np.testing.assert_allclose(
        s.edge_lowers.numpy()[: s.n_edges],
        lowers_before,
        rtol=0,
        atol=0,
        err_msg="edge AABBs were clobbered by hinge preprocess growth",
    )
    print("test_hinge_growth_does_not_zero_mesh_edges OK")


def test_edge_realloc_preserves_contents():
    """Defense-in-depth: growing edge capacity after upload copies + rebuilds BVH."""
    c = Stage3Config()
    c.auto_capacity = True
    c.lazy_calculators = True
    c.energy_calcs = [HingeEnergyCalculator]
    c.max_gt_samples = 64
    c.use_cuda_graph = False
    c.max_blocks = 4096
    c.min_blocks = 64

    V, F = _two_face_mesh()
    s = Stage3System(config=c, device=DEVICE)
    s.register_mesh(V, F, V, F)

    edges_before = s.edges.numpy()[: s.n_edges].copy()
    lowers_before = s.edge_lowers.numpy()[: s.n_edges].copy()
    old_cap = s.cap_edges
    need = old_cap + 32
    s.ensure_capacity(n_edges=need)
    assert s.cap_edges >= need
    np.testing.assert_array_equal(s.edges.numpy()[: s.n_edges], edges_before)
    np.testing.assert_allclose(
        s.edge_lowers.numpy()[: s.n_edges], lowers_before, rtol=0, atol=0
    )
    assert s.edge_bvh is not None
    print("test_edge_realloc_preserves_contents OK")


if __name__ == "__main__":
    test_contact_hint_clamped_to_max_blocks()
    test_hinge_growth_does_not_zero_mesh_edges()
    test_edge_realloc_preserves_contents()
    print("all capacity regression tests passed")
