from __future__ import annotations
import numpy as np
import warp as wp
import igl
import scipy.sparse as sp
from typing import Union, List

from .kernels.energy_kernels.collision_energy import collision_hess_dx_kernel

from .kernels.ccd_kernels import *
from .kernels.energy_kernels.distance_energy import *
from .kernels.energy_kernels.elastic_energy import *
from .kernels.energy_kernels.lb_curvature_energy import *
from .kernels.energy_kernels.hinge_energy import *
from .kernels.energy_kernels.collision_energy import *
from .kernels.energy_kernels.contact_detection import *
from .kernels.geometry_kernels import *
from .kernels.utils_kernels import *
from .utils import wp_slice
from .utils import stage3_logger as logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import Stage3System


mat23 = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


class EnergyCalculator:
    name = None
    
    def __init__(self, system: Stage3System):
        self.system = system

    def ensure_capacity(self):
        """(Re)allocate workspaces to match system capacities. Default: no-op."""
        pass

    def preprocess(self, V, F):
        pass

    def compute_energy(self, x: wp.array, energy: wp.array):
        """Compute the energy and add onto the energy array"""
        raise NotImplementedError

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        """
        ALWAYS compute the gradient and add onto the grad array;
        ALWAYS compute the hess_diag and add onto the hess_diag array;
        MAY also compute the hessian (blocks) and other needed buffers
            and store them into the member arrays of the energy calculator
        """
        raise NotImplementedError

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        """Compute hessian * dx and add onto the hess_dx array"""
        raise NotImplementedError


class DummyEnergyCalculator(EnergyCalculator):
    """
    Dummy energy calculator for testing.
    Always return energy E = 0.5 * x^T * A * x + b^T * x + c,
        where A is a random symmetric positive semi-definite matrix,
        b, c are random vectors.
    """

    name = "Dummy"

    def __init__(self, system: Stage3System):
        super().__init__(system)
        self.A_np = None
        self.b_np = None
        self.c_np = None
        self.ensure_capacity()

    def ensure_capacity(self):
        s = self.system
        c = s.config
        MP = max(int(s.cap_particles), int(c.max_particles) if not c.auto_capacity else 0)
        if MP <= 0:
            MP = int(c.max_particles)
        assert (
            MP <= 1024
        ), f"Dummy energy calculator only supports max_particles <= 1024, got max_particles = {MP}"
        if self.A_np is not None and self.A_np.shape[0] == MP * 3:
            return
        self.A_np = np.random.randn(MP * 3, MP * 3)
        self.A_np = self.A_np.T @ self.A_np
        self.b_np = np.random.randn(MP * 3)
        self.c_np = np.random.randn(1)

    def compute_energy(self, x: wp.array, energy: wp.array):
        x_np = x.numpy().reshape(-1)
        energy_np = (
            0.5 * np.dot(x_np, np.dot(self.A_np, x_np))
            + np.dot(self.b_np, x_np)
            + self.c_np
        )
        energy.assign(energy_np)

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        x_np = x.numpy().reshape(-1)
        grad_np = self.A_np @ x_np + self.b_np
        hess_diag_np = np.diag(self.A_np)

        grad.assign(grad_np.reshape(-1, 3) * grad_coeff)
        hess_diag.assign(hess_diag_np.reshape(-1, 3))

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        dx_np = dx.numpy().reshape(-1)
        hess_dx_np = self.A_np @ dx_np

        hess_dx.assign(hess_dx_np.reshape(-1, 3))


class Mesh2GTDistanceEnergyCalculator(EnergyCalculator):
    name = "M2GT"
    def __init__(self, system: Stage3System):
        super().__init__(system)
        self.target = None
        self.target_distance = None
        self._cap_particles = 0
        self.ensure_capacity()

    def ensure_capacity(self):
        s = self.system
        MP = max(int(s.cap_particles), 0)
        if MP <= 0:
            with wp.ScopedDevice(s.device):
                if self.target is None:
                    self.target = wp.zeros(0, dtype=wp.vec3)
                    self.target_distance = wp.zeros(0, dtype=wp.float32)
            return
        if self._cap_particles >= MP and self.target is not None:
            return
        with wp.ScopedDevice(s.device):
            self.target = wp.zeros(MP, dtype=wp.vec3)
            self.target_distance = wp.zeros(MP, dtype=wp.float32)
        self._cap_particles = MP

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        self.target_distance.fill_(1e9)
        self.target.fill_(1e9)
        wp.launch(
            kernel=update_min_pt_distance_kernel,
            dim=(s.n_particles, s.n_gt_triangles),
            inputs=[x, s.gt_vertices, s.gt_triangles],
            outputs=[self.target_distance],
            device=s.device,
        )
        wp.launch(
            kernel=update_closest_point_on_target_kernel,
            dim=(s.n_particles, s.n_gt_triangles),
            inputs=[x, s.gt_vertices, s.gt_triangles, self.target_distance],
            outputs=[self.target],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=mesh2gt_pp_distance_energy_kernel,
            dim=s.n_particles,
            inputs=[x, self.target, s.voronoi_areas, c.mesh2gt_dist_stiffness],
            outputs=[energy],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=mesh2gt_pp_distance_energy_diff_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.target,
                s.voronoi_areas,
                c.mesh2gt_dist_stiffness,
                grad_coeff,
            ],
            outputs=[grad, hess_diag],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=mesh2gt_pp_distance_energy_hess_dx_kernel,
            dim=s.n_particles,
            inputs=[s.voronoi_areas, c.mesh2gt_dist_stiffness, dx],
            outputs=[hess_dx],
            device=s.device,
        )
        
        
class Mesh2GTDistanceBvhEnergyCalculator(Mesh2GTDistanceEnergyCalculator):
    name = "M2GT"
    def __init__(self, system: Stage3System):
        super().__init__(system)

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        self.target.fill_(1e9)
        wp.launch(
            kernel=update_closest_point_on_target_bvh_kernel,
            dim=s.n_particles,
            inputs=[x, s.gt_mesh_bvh.id],
            outputs=[self.target],
            device=s.device,
        )


class GT2MeshDistanceEnergyCalculator(EnergyCalculator):
    """
    Distance between remeshed triangles and ground truth points.
    The ground truth points are uniformly sampled on the ground truth mesh.
    """
    name = "GT2M"

    def __init__(self, system: Stage3System):
        super().__init__(system)
        self.closest_tids = None
        self.pt_types = None
        self.d = None
        self.dd_dx = None
        self._cap_gt_samples = 0
        self.ensure_capacity()

    def ensure_capacity(self):
        s = self.system
        MS_GT = max(int(s.cap_gt_samples), 0)
        if MS_GT <= 0:
            with wp.ScopedDevice(s.device):
                if self.closest_tids is None:
                    self.closest_tids = wp.zeros(0, dtype=wp.int32)
                    self.pt_types = wp.zeros(0, dtype=wp.int32)
                    self.d = wp.zeros(0, dtype=wp.float32)
                    self.dd_dx = wp.zeros((0, 4), dtype=wp.vec3)
            return
        if self._cap_gt_samples >= MS_GT and self.closest_tids is not None:
            return
        with wp.ScopedDevice(s.device):
            self.closest_tids = wp.zeros(MS_GT, dtype=wp.int32)
            self.pt_types = wp.zeros(MS_GT, dtype=wp.int32)
            self.d = wp.zeros(MS_GT, dtype=wp.float32)
            self.dd_dx = wp.zeros((MS_GT, 4), dtype=wp.vec3)
        self._cap_gt_samples = MS_GT

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        self.d.fill_(1e9)
        wp.launch(
            kernel=update_min_pt_distance_kernel,
            dim=(s.n_gt_samples, s.n_triangles),
            inputs=[s.gt_samples, x, s.triangles],
            outputs=[self.d],
            device=s.device,
        )
        wp.launch(
            kernel=update_closest_triangle_on_target_kernel,
            dim=(s.n_gt_samples, s.n_triangles),
            inputs=[s.gt_samples, x, s.triangles, self.d],
            outputs=[self.closest_tids],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=gt2mesh_pt_distance_energy_kernel,
            dim=s.n_gt_samples,
            inputs=[
                x,
                s.gt_samples,
                self.closest_tids,
                s.triangles,
                s.gt_sample_weights,
                c.gt2mesh_dist_stiffness,
            ],
            outputs=[
                self.pt_types,
                self.d,
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=gt2mesh_pt_distance_energy_diff_kernel,
            dim=s.n_gt_samples,
            inputs=[
                x,
                s.gt_samples,
                self.closest_tids,
                s.triangles,
                s.gt_sample_weights,
                c.gt2mesh_dist_stiffness,
                self.pt_types,
                self.d,
                grad_coeff,
            ],
            outputs=[self.dd_dx, grad, hess_diag],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=gt2mesh_pt_distance_energy_hess_dx_kernel,
            dim=s.n_gt_samples,
            inputs=[
                self.closest_tids,
                s.triangles,
                s.gt_sample_weights,
                c.gt2mesh_dist_stiffness,
                self.dd_dx,
                dx,
            ],
            outputs=[hess_dx],
            device=s.device,
        )


class GT2MeshDistanceBvhEnergyCalculator(GT2MeshDistanceEnergyCalculator):
    """
    Distance between remeshed triangles and ground truth points.
    The ground truth points are uniformly sampled on the ground truth mesh.
    """
    name = "GT2M"

    def __init__(self, system: Stage3System):
        super().__init__(system)

    def update_target(self, x: wp.array):
        s = self.system
        c = s.config

        s.mesh_bvh.refit()

        wp.launch(
            kernel=update_closest_triangle_on_target_bvh_kernel,
            dim=s.n_gt_samples,
            inputs=[s.gt_samples, s.mesh_bvh.id],
            outputs=[self.closest_tids],
            device=s.device,
        )

class ElasticEnergyCalculator(EnergyCalculator):
    name = "Elas"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config

        self.mu = c.elas_young_modulus / (2 * (1 + c.elas_poisson_ratio))
        self.la = (
            c.elas_young_modulus
            * c.elas_poisson_ratio
            / ((1 + c.elas_poisson_ratio) * (1 - 2 * c.elas_poisson_ratio))
        )
        self.areas = None
        self.inv_Dm = None
        self._cap_faces = 0
        self.ensure_capacity()
        # Elastic: F = Ds * inv(Dm); forces from Piola stress on each triangle.

    def ensure_capacity(self):
        s = self.system
        MT = max(int(s.cap_faces), 0)
        if MT <= 0:
            with wp.ScopedDevice(s.device):
                if self.areas is None:
                    self.areas = wp.zeros(0, dtype=wp.float32)
                    self.inv_Dm = wp.zeros(0, dtype=wp.mat22)
            return
        if self._cap_faces >= MT and self.areas is not None:
            return
        with wp.ScopedDevice(s.device):
            self.areas = wp.zeros(MT, dtype=wp.float32)
            self.inv_Dm = wp.zeros(MT, dtype=wp.mat22)
        self._cap_faces = MT

    def preprocess(self, V, F):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_preprocess_kernel,
            dim=s.n_triangles,
            inputs=[
                s.q_rest,
                s.triangles,
            ],
            outputs=[
                self.areas,
                self.inv_Dm,
            ],
            device=s.device,
        )

        if c.debug:
            areas_np = self.areas.numpy()[: s.n_triangles]
            logger.debug(
                f"[ElasticEnergyCalculator] min area = {np.min(areas_np)}, max area = {np.max(areas_np)}"
            )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_energy_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.triangles,
                self.areas,
                self.inv_Dm,
                self.mu,
                self.la,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_diff_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.triangles,
                self.areas,
                self.inv_Dm,
                self.mu,
                self.la,
                grad_coeff,
            ],
            outputs=[
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=elastic_hess_dx_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.triangles,
                self.areas,
                self.inv_Dm,
                self.mu,
                self.la,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )


class LBCurvatureEnergyCalculator(EnergyCalculator):
    name = "Curv"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        self.LB_nnz = 0
        self.LB_indices = None
        self.LB_indptr = None
        self.LB_data = None
        self.curv_rest = None
        self.curv = None
        self._cap_particles = 0
        self._cap_lb_nnz = 0
        self.ensure_capacity()

    def ensure_capacity(self):
        s = self.system
        MP = max(int(s.cap_particles), 0)
        ME = max(int(s.cap_edges), 0)
        M_LB_nnz = 2 * ME + MP
        if MP <= 0:
            with wp.ScopedDevice(s.device):
                if self.LB_indices is None:
                    self.LB_indices = wp.zeros(0, dtype=wp.int32)
                    self.LB_indptr = wp.zeros(1, dtype=wp.int32)
                    self.LB_data = wp.zeros(0, dtype=wp.float32)
                    self.curv_rest = wp.zeros(0, dtype=wp.float32)
                    self.curv = wp.zeros(0, dtype=wp.vec3)
            return
        if (
            self._cap_particles >= MP
            and self._cap_lb_nnz >= M_LB_nnz
            and self.LB_indices is not None
        ):
            return
        with wp.ScopedDevice(s.device):
            self.LB_indices = wp.zeros(M_LB_nnz, dtype=wp.int32)
            self.LB_indptr = wp.zeros(MP + 1, dtype=wp.int32)
            self.LB_data = wp.zeros(M_LB_nnz, dtype=wp.float32)
            self.curv_rest = wp.zeros(MP, dtype=wp.float32)
            self.curv = wp.zeros(MP, dtype=wp.vec3)
        self._cap_particles = MP
        self._cap_lb_nnz = M_LB_nnz

    def preprocess(self, V, F):
        s = self.system
        c = s.config

        cotmatrix = igl.cotmatrix(V, F)
        massmatrix = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        voronoi_areas = massmatrix.diagonal()
        LB_sp: sp.csr_matrix = -(cotmatrix / voronoi_areas[:, None]).tocsr()  # [N, N]
        curv_rest = np.linalg.norm(LB_sp.dot(V), axis=1)

        NP = s.n_particles
        assert LB_sp.shape == (NP, NP), f"LB_sp shape {LB_sp.shape} != ({NP}, {NP})"
        self.LB_nnz = LB_sp.nnz

        wp_slice(self.LB_indices, 0, self.LB_nnz).assign(LB_sp.indices)
        wp_slice(self.LB_indptr, 0, NP + 1).assign(LB_sp.indptr)
        wp_slice(self.LB_data, 0, self.LB_nnz).assign(LB_sp.data)
        wp_slice(self.curv_rest, 0, NP).assign(curv_rest)

        # logger.info(f"[BendingEnergyCalculator] Mean curvature: \n{curv_rest}")
        # for i in range(100):
        #     logger.info(f"[BendingEnergyCalculator] Mean curvature [{i}]: {curv_rest[i]}")

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=lb_curvature_energy_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.LB_indices,
                self.LB_indptr,
                self.LB_data,
                self.curv_rest,
                c.curv_stiffness,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=lb_curvature_diff_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.LB_indices,
                self.LB_indptr,
                self.LB_data,
                self.curv_rest,
                c.curv_stiffness,
                grad_coeff,
            ],
            outputs=[
                self.curv,
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=lb_curvature_hess_dx_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                self.LB_indices,
                self.LB_indptr,
                self.LB_data,
                self.curv_rest,
                c.curv_stiffness,
                self.curv,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )


class HingeEnergyCalculator(EnergyCalculator):
    name = "Bend"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        self.rest_angles = None
        self.rest_elens = None
        self.blocks = None
        self.block_indices = None
        self._cap_edges = 0
        self.n_hinges = 0
        self.ensure_capacity()

    def ensure_capacity(self, min_hinges: int = 0):
        """Size hinge workspaces.

        Hinge count can exceed unique mesh edges on non-manifold meshes.
        Grow *only* this calculator's buffers — never mesh ``s.edges`` /
        edge-BVH storage (those are collision topology and are uploaded
        before preprocess).
        """
        s = self.system
        ME = max(int(s.cap_edges), int(min_hinges), 0)
        if ME <= 0:
            with wp.ScopedDevice(s.device):
                if self.rest_angles is None:
                    self.rest_angles = wp.zeros(0, dtype=wp.float32)
                    self.rest_elens = wp.zeros(0, dtype=wp.float32)
                    self.blocks = wp.zeros((0, 4, 4), dtype=wp.mat33)
                    self.block_indices = wp.zeros((0, 4), dtype=wp.int32)
            return
        if self._cap_edges >= ME and self.rest_angles is not None:
            return
        with wp.ScopedDevice(s.device):
            self.rest_angles = wp.zeros(ME, dtype=wp.float32)
            self.rest_elens = wp.zeros(ME, dtype=wp.float32)
            self.blocks = wp.zeros((ME, 4, 4), dtype=wp.mat33)
            self.block_indices = wp.zeros((ME, 4), dtype=wp.int32)
        self._cap_edges = ME

    def preprocess(self, V, F):
        """Discover hinges via sorted half-edge table (O(F log F)), not O(F^2)."""
        s = self.system
        c = s.config

        # Prefer mesh faces passed into preprocess; fall back to device triangles.
        if F is not None:
            tris = np.asarray(F, dtype=np.int32)
        else:
            tris = s.triangles.numpy()[: s.n_triangles]

        hinge_idx = build_hinge_indices(tris)
        n_hinges = int(hinge_idx.shape[0])
        self.n_hinges = n_hinges

        # Non-manifold opposite pairs can exceed unique-edge capacity. Grow
        # hinge buffers only — do not reallocate s.edges / edge BVH.
        if n_hinges > self._cap_edges:
            self.ensure_capacity(min_hinges=n_hinges)
        if n_hinges > self._cap_edges:
            raise RuntimeError(
                f"Hinge count {n_hinges} exceeds hinge capacity {self._cap_edges} "
                f"(n_edges={s.n_edges}, n_faces={s.n_triangles})"
            )

        if n_hinges != s.n_edges:
            # Boundary / open meshes: unique edges include non-hinge boundary edges.
            # Non-manifold meshes may also diverge. Launch dims use n_hinges.
            logger.warning(
                f"Hinge count {n_hinges} != unique edges {s.n_edges} "
                f"(boundary or non-manifold geometry); using n_hinges for Bend energy"
            )

        if n_hinges == 0:
            return

        wp_slice(self.block_indices, 0, n_hinges).assign(hinge_idx)
        wp.launch(
            kernel=hinge_fill_rest_kernel,
            dim=n_hinges,
            inputs=[
                s.q_rest,
                self.block_indices,
            ],
            outputs=[
                self.rest_angles,
                self.rest_elens,
            ],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config
        n_h = self.n_hinges
        if n_h <= 0:
            return

        wp.launch(
            kernel=hinge_energy_kernel,
            dim=n_h,
            inputs=[
                x,
                self.block_indices,
                self.rest_angles,
                self.rest_elens,
                c.hinge_stiffness,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config
        n_h = self.n_hinges
        if n_h <= 0:
            return
        
        wp.launch(
            kernel=hinge_diff_kernel,
            dim=n_h,
            inputs=[
                x,
                self.block_indices,
                self.rest_angles,
                self.rest_elens,
                c.hinge_stiffness,
                grad_coeff,
            ],
            outputs=[
                self.blocks,
                grad,
                hess_diag,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=block_spd_project_kernel,
            dim=n_h,
            inputs=[
                self.blocks,
                c.spd_max_iters,
            ],
            device=s.device,
        )
        
    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config
        n_h = self.n_hinges
        if n_h <= 0:
            return
        
        wp.launch(
            kernel=hinge_hess_dx_kernel,
            dim=n_h,
            inputs=[
                self.block_indices,
                self.blocks,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )

class CollisionEnergyCalculator(EnergyCalculator):
    name = "Coll"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        s = self.system
        with wp.ScopedDevice(s.device):
            self.contact_counter = wp.zeros(1, dtype=wp.int32)
        self.block_indices = None
        self.block_types = None
        self.d = None
        self.dd_dx = None
        self._cap_blocks = 0
        self.ensure_capacity()

    def ensure_capacity(self):
        s = self.system
        MB = max(int(s.cap_blocks), 0)
        if MB <= 0:
            with wp.ScopedDevice(s.device):
                if self.block_indices is None:
                    self.block_indices = wp.zeros((0, 4), dtype=wp.int32)
                    self.block_types = wp.zeros((0, 2), dtype=wp.int32)
                    self.d = wp.zeros(0, dtype=wp.float32)
                    self.dd_dx = wp.zeros((0, 4), dtype=wp.vec3)
            return
        if self._cap_blocks >= MB and self.block_indices is not None:
            return
        with wp.ScopedDevice(s.device):
            self.block_indices = wp.zeros((MB, 4), dtype=wp.int32)
            self.block_types = wp.zeros((MB, 2), dtype=wp.int32)
            self.d = wp.zeros(MB, dtype=wp.float32)
            self.dd_dx = wp.zeros((MB, 4), dtype=wp.vec3)
        self._cap_blocks = MB
            
    def preprocess(self, V, F):
        c = self.system.config
        self.radius = c.contact_detection_radius

    def _handle_contact_overflow(self, n_contacts: int, x: wp.array) -> bool:
        """Grow contact buffer if possible; else shrink radius. Returns True if retried."""
        s = self.system
        c = s.config
        cap = s.cap_blocks
        if n_contacts <= cap:
            return False

        logger.warning(
            f"Number of contacts ({n_contacts}) exceeds contact capacity ({cap})"
        )
        # Prefer growing the buffer (Phase B) over silently thinning the contact set.
        if cap < c.max_blocks:
            # Need at least n_contacts slots; geometric growth may overshoot.
            need = min(c.max_blocks, max(n_contacts, int(cap * c.capacity_growth) if cap > 0 else n_contacts))
            if need > cap:
                logger.info(f"Growing contact buffer {cap} -> {need}")
                s.ensure_capacity(n_blocks=need)
                self.detect_contact(x)
                return True

        self.radius /= 2.0
        logger.info(f"Retrying with smaller detection radius: {self.radius}")
        if self.radius < 1e-12:
            raise RuntimeError(
                f"Contact buffer full ({n_contacts} > {cap}) and detection radius "
                f"collapsed; raise max_blocks or reduce contact density"
            )
        self.detect_contact(x)
        return True

    def detect_contact(self, x: wp.array):
        s = self.system
        c = s.config
        MB = s.cap_blocks

        self.contact_counter.zero_()
        wp.launch(
            kernel=detect_pt_contact_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                self.contact_counter,
                MB,
                x,
                s.triangles,
                c.d_hat,
                self.radius,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=detect_ee_contact_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                self.contact_counter,
                MB,
                x,
                s.edges,
                c.d_hat,
                self.radius,
                c.ee_classify_thres,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )

        n_contacts = self.contact_counter.numpy()[0]
        logger.debug(f"Detected {n_contacts} contact pairs")
        if self._handle_contact_overflow(n_contacts, x):
            return

    def ccd(self, x: wp.array, v: wp.array, ccd_step: wp.array):
        s = self.system
        c = s.config
        MB = s.cap_blocks

        wp.launch(
            kernel=accd_kernel,
            dim=MB,
            inputs=[
                self.contact_counter,
                x,
                v,
                self.block_types,
                self.block_indices,
                c.ccd_slackness,
                c.ccd_thickness,
                c.ccd_max_iters,
                c.ee_classify_thres,
            ],
            outputs=[
                ccd_step,
            ],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config
        MB = s.cap_blocks

        wp.launch(
            kernel=collision_energy_kernel,
            dim=MB,
            inputs=[
                x,
                self.contact_counter,
                self.block_types,
                self.block_indices,
                c.coll_stiffness,
                c.d_hat,
                c.ee_classify_thres,
            ],
            outputs=[
                self.d,
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config
        MB = s.cap_blocks

        wp.launch(
            kernel=collision_diff_kernel,
            dim=MB,
            inputs=[
                x,
                self.contact_counter,
                self.block_types,
                self.block_indices,
                c.coll_stiffness,
                c.d_hat,
                self.d,
                grad_coeff,
            ],
            outputs=[
                self.dd_dx,
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config
        MB = s.cap_blocks

        wp.launch(
            kernel=collision_hess_dx_kernel,
            dim=MB,
            inputs=[
                self.contact_counter,
                self.block_indices,
                self.d,
                c.coll_stiffness,
                c.d_hat,
                self.dd_dx,
                dx,
            ],
            outputs=[hess_dx],
            device=s.device,
        )


class CollisionBvhEnergyCalculator(CollisionEnergyCalculator):
    name = "Coll"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)
        
    def preprocess(self, V, F):
        super().preprocess(V, F)

    def detect_contact(self, x: wp.array):
        s = self.system
        c = s.config
        MB = s.cap_blocks

        self.contact_counter.zero_()
        wp.launch(
            kernel=detect_pt_contact_bvh_kernel,
            dim=s.n_particles,
            inputs=[
                s.tri_bvh.id,
                self.contact_counter,
                MB,
                x,
                s.triangles,
                c.d_hat,
                self.radius,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=detect_ee_contact_bvh_kernel,
            dim=s.n_edges,
            inputs=[
                s.edge_bvh.id,
                self.contact_counter,
                MB,
                x,
                s.edges,
                c.d_hat,
                self.radius,
                c.ee_classify_thres,
            ],
            outputs=[
                self.block_types,
                self.block_indices,
            ],
            device=s.device,
        )

        n_contacts = self.contact_counter.numpy()[0]
        logger.debug(f"Detected {n_contacts} contact pairs")
        if self._handle_contact_overflow(n_contacts, x):
            return
            

class CollisionWoBufferEnergyCalculator(EnergyCalculator):
    name = "collision (wo buffer)"
    
    def __init__(self, system: Stage3System):
        super().__init__(system)

        s = self.system
        c = s.config
        
    def ccd(self, x: wp.array, v: wp.array, ccd_step: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=accd_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                v,
                s.triangles,
                c.ccd_slackness,
                c.ccd_thickness,
                c.contact_detection_radius,
                c.ccd_max_iters,
            ],
            outputs=[
                ccd_step,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=accd_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                v,
                s.edges,
                c.ccd_slackness,
                c.ccd_thickness,
                c.contact_detection_radius,
                c.ee_classify_thres,
                c.ccd_max_iters,
            ],
            outputs=[
                ccd_step,
            ],
            device=s.device,
        )

    def compute_energy(self, x: wp.array, energy: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_energy_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                s.triangles,
                c.coll_stiffness,
                c.d_hat,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=collision_energy_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                s.edges,
                c.coll_stiffness,
                c.ee_classify_thres,
                c.d_hat,
            ],
            outputs=[
                energy,
            ],
            device=s.device,
        )

    def compute_diff(
        self, x: wp.array, grad_coeff: float, grad: wp.array, hess_diag: wp.array
    ):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_diff_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                s.triangles,
                c.coll_stiffness,
                c.d_hat,
                grad_coeff,
            ],
            outputs=[
                grad,
                hess_diag,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=collision_diff_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                s.edges,
                c.coll_stiffness,
                c.d_hat,
                c.ee_classify_thres,
                grad_coeff,
            ],
            outputs=[
                grad,
                hess_diag,
            ],
            device=s.device,
        )

    def compute_hess_dx(self, x: wp.array, dx: wp.array, hess_dx: wp.array):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_hess_dx_wo_buffer_pt_kernel,
            dim=(s.n_particles, s.n_triangles),
            inputs=[
                x,
                s.triangles,
                c.d_hat,
                c.coll_stiffness,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )
        wp.launch(
            kernel=collision_hess_dx_wo_buffer_ee_kernel,
            dim=(s.n_edges, s.n_edges),
            inputs=[
                x,
                s.edges,
                c.d_hat,
                c.ee_classify_thres,
                c.coll_stiffness,
                dx,
            ],
            outputs=[
                hess_dx,
            ],
            device=s.device,
        )

