import numpy as np
import warp as wp
import trimesh
import igl
import scipy.sparse as sp
import logging

from .config import Stage3Config
from .energy import *
from .cg_solver import CGSolver
from .kernels.solver_kernels import line_search_kernel, clamp_p_kernel, restore_q_kernel
from .utils import stage3_logger as logger
from .utils import wp_slice
from .metrics import compute_igl_CD_HD
from .geometry import *


def _default_stage3_device():
    """Prefer the current torch CUDA device over hard-coded cuda:0."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        pass
    return "cuda:0"


class Stage3System:
    def __init__(self, config: Stage3Config = None, device=None):
        wp.init()

        if config is None:
            config = Stage3Config()
        self.config = config
        self.device = device if device is not None else _default_stage3_device()

        if config.debug:
            if config.use_cuda_graph:
                logger.warning(
                    "Debug mode is enabled but CUDA graph is also enabled; some debug information will not be generated"
                )
            if logger.level > logging.DEBUG:
                logger.warning(
                    "Debug mode is enabled but logger level is higher than DEBUG; some debug information will not be displayed"
                )

        self._init_counters()
        self._init_capacity_fields()
        self._init_scalar_arrays()
        # Energy calculators may be empty until first ensure/use when
        # config.lazy_calculators is True (default).
        self._init_energy_calcs()

        # CGSolver owns buffers that grow with ensure_capacity; construct now
        # (cheap under auto_capacity) so step() never has to special-case None.
        self.cg_solver = CGSolver(self)
        self.line_search_graph = None
        self.do_nothing = False

        # Legacy path: reserve full max_* ceilings at construction (~5 GiB).
        # Default auto_capacity=True defers allocation until register_mesh.
        if not config.auto_capacity:
            c = config
            MF = c.max_particles * 2 + 1024
            ME = MF // 2 * 3
            MF_GT = c.max_gt_particles * 2 + 1024
            self.ensure_capacity(
                n_particles=c.max_particles,
                n_faces=MF,
                n_edges=ME,
                n_gt_particles=c.max_gt_particles,
                n_gt_faces=MF_GT,
                n_gt_samples=c.max_gt_samples,
                n_blocks=c.max_blocks,
            )

    def _init_counters(self):
        self.n_particles = 0
        self.n_triangles = 0
        self.n_edges = 0
        self.n_gt_particles = 0
        self.n_gt_triangles = 0
        self.n_gt_samples = 0

    def _init_capacity_fields(self):
        """Allocated buffer capacities (not mesh counts). Start at 0 under auto_capacity."""
        self.cap_particles = 0
        self.cap_faces = 0
        self.cap_edges = 0
        self.cap_gt_particles = 0
        self.cap_gt_faces = 0
        self.cap_gt_samples = 0
        self.cap_blocks = 0

    def _init_scalar_arrays(self):
        """Tiny always-present buffers; heavy mesh/contact arrays come from ensure_capacity."""
        with wp.ScopedDevice(self.device):
            self.energy = wp.zeros(1, dtype=wp.float32)
            self.energy_prev = wp.zeros(1, dtype=wp.float32)
            self.ccd_step = wp.zeros(1, dtype=wp.float32)
            # Placeholders until first ensure_capacity (Warp allows length-0 arrays).
            self.q = wp.zeros(0, dtype=wp.vec3)
            self.q_prev_newton = wp.zeros(0, dtype=wp.vec3)
            self.q_prev_detection = wp.zeros(0, dtype=wp.vec3)
            self.q_rest = wp.zeros(0, dtype=wp.vec3)
            self.p = wp.zeros(0, dtype=wp.vec3)
            self.grad = wp.zeros(0, dtype=wp.vec3)
            self.hess_diag = wp.zeros(0, dtype=wp.vec3)
            self.voronoi_areas = wp.zeros(0, dtype=wp.float32)
            self.triangles = wp.zeros((0, 3), dtype=wp.int32)
            self.edges = wp.zeros((0, 2), dtype=wp.int32)
            self.gt_vertices = wp.zeros(0, dtype=wp.vec3)
            self.gt_triangles = wp.zeros((0, 3), dtype=wp.int32)
            self.gt_samples = wp.zeros(0, dtype=wp.vec3)
            self.gt_sample_weights = wp.zeros(0, dtype=wp.float32)
            self.edge_lowers = wp.zeros(0, dtype=wp.vec3)
            self.edge_uppers = wp.zeros(0, dtype=wp.vec3)
            self.tri_lowers = wp.zeros(0, dtype=wp.vec3)
            self.tri_uppers = wp.zeros(0, dtype=wp.vec3)

    @staticmethod
    def _grow_cap(need: int, old: int, growth: float, ceiling: int, name: str) -> int:
        """Return a capacity >= need, growing geometrically from old, capped by ceiling."""
        if need <= 0:
            return old
        if need <= old:
            return old
        if ceiling > 0 and need > ceiling:
            raise RuntimeError(
                f"Stage3 capacity overflow: need {need} for {name}, hard limit is {ceiling}"
            )
        if old <= 0:
            # First allocation: modest headroom so a slightly larger next mesh reuses.
            new = max(need + need // 4 + 64, need)
        else:
            new = old
            while new < need:
                nxt = max(int(new * growth), new + 1)
                if nxt <= new:
                    nxt = new + need
                new = nxt
        if ceiling > 0:
            new = min(new, ceiling)
        return new

    def ensure_capacity(
        self,
        n_particles: int = 0,
        n_faces: int = 0,
        n_edges: int = 0,
        n_gt_particles: int = 0,
        n_gt_faces: int = 0,
        n_gt_samples: int = 0,
        n_blocks: int = 0,
    ) -> bool:
        """Grow mesh/contact buffers if any requested size exceeds current capacity.

        Returns True if any buffer was reallocated (CUDA graphs are invalidated).
        """
        c = self.config
        growth = float(getattr(c, "capacity_growth", 2.0))

        # Face/edge hard limits derived from max_particles / max_gt_particles.
        face_ceiling = c.max_particles * 2 + 1024
        edge_ceiling = face_ceiling // 2 * 3
        gt_face_ceiling = c.max_gt_particles * 2 + 1024

        new_MP = self._grow_cap(n_particles, self.cap_particles, growth, c.max_particles, "particles")
        new_MF = self._grow_cap(n_faces, self.cap_faces, growth, face_ceiling, "faces")
        new_ME = self._grow_cap(n_edges, self.cap_edges, growth, edge_ceiling, "edges")
        new_MP_GT = self._grow_cap(
            n_gt_particles, self.cap_gt_particles, growth, c.max_gt_particles, "gt_particles"
        )
        new_MF_GT = self._grow_cap(
            n_gt_faces, self.cap_gt_faces, growth, gt_face_ceiling, "gt_faces"
        )
        new_MS_GT = self._grow_cap(
            n_gt_samples, self.cap_gt_samples, growth, c.max_gt_samples, "gt_samples"
        )
        new_MB = self._grow_cap(n_blocks, self.cap_blocks, growth, c.max_blocks, "blocks")

        grow_P = new_MP > self.cap_particles
        grow_F = new_MF > self.cap_faces
        grow_E = new_ME > self.cap_edges
        grow_P_GT = new_MP_GT > self.cap_gt_particles
        grow_F_GT = new_MF_GT > self.cap_gt_faces
        grow_S_GT = new_MS_GT > self.cap_gt_samples
        grow_B = new_MB > self.cap_blocks

        if not any((grow_P, grow_F, grow_E, grow_P_GT, grow_F_GT, grow_S_GT, grow_B)):
            return False

        logger.debug(
            "Stage3 ensure_capacity: "
            f"P {self.cap_particles}->{new_MP}, F {self.cap_faces}->{new_MF}, "
            f"E {self.cap_edges}->{new_ME}, P_GT {self.cap_gt_particles}->{new_MP_GT}, "
            f"F_GT {self.cap_gt_faces}->{new_MF_GT}, S_GT {self.cap_gt_samples}->{new_MS_GT}, "
            f"B {self.cap_blocks}->{new_MB}"
        )

        with wp.ScopedDevice(self.device):
            if grow_P:
                self.q = wp.zeros(new_MP, dtype=wp.vec3)
                self.q_prev_newton = wp.zeros(new_MP, dtype=wp.vec3)
                self.q_prev_detection = wp.zeros(new_MP, dtype=wp.vec3)
                self.q_rest = wp.zeros(new_MP, dtype=wp.vec3)
                self.p = wp.zeros(new_MP, dtype=wp.vec3)
                self.grad = wp.zeros(new_MP, dtype=wp.vec3)
                self.hess_diag = wp.zeros(new_MP, dtype=wp.vec3)
                self.voronoi_areas = wp.zeros(new_MP, dtype=wp.float32)
                self.cap_particles = new_MP

            if grow_F:
                self.triangles = wp.zeros((new_MF, 3), dtype=wp.int32)
                self.tri_lowers = wp.zeros(new_MF, dtype=wp.vec3)
                self.tri_uppers = wp.zeros(new_MF, dtype=wp.vec3)
                self.cap_faces = new_MF

            if grow_E:
                # Preserve uploaded edge topology / AABBs when growing after
                # register_mesh (zeroed realloc would poison collision detection).
                n_keep_e = min(int(self.n_edges), int(self.cap_edges))
                old_edges = self.edges
                old_edge_lowers = self.edge_lowers
                old_edge_uppers = self.edge_uppers
                self.edges = wp.zeros((new_ME, 2), dtype=wp.int32)
                self.edge_lowers = wp.zeros(new_ME, dtype=wp.vec3)
                self.edge_uppers = wp.zeros(new_ME, dtype=wp.vec3)
                if n_keep_e > 0:
                    wp_slice(self.edges, 0, n_keep_e).assign(
                        old_edges.numpy()[:n_keep_e]
                    )
                    wp.copy(self.edge_lowers, old_edge_lowers, count=n_keep_e)
                    wp.copy(self.edge_uppers, old_edge_uppers, count=n_keep_e)
                self.cap_edges = new_ME
                # BVH holds views of the old arrays; rebuild if one exists.
                if n_keep_e > 0 and getattr(self, "edge_bvh", None) is not None:
                    self.edge_bvh = wp.Bvh(
                        wp_slice(self.edge_lowers, 0, self.n_edges),
                        wp_slice(self.edge_uppers, 0, self.n_edges),
                    )

            if grow_P_GT:
                self.gt_vertices = wp.zeros(new_MP_GT, dtype=wp.vec3)
                self.cap_gt_particles = new_MP_GT

            if grow_F_GT:
                self.gt_triangles = wp.zeros((new_MF_GT, 3), dtype=wp.int32)
                self.cap_gt_faces = new_MF_GT

            if grow_S_GT:
                self.gt_samples = wp.zeros(new_MS_GT, dtype=wp.vec3)
                self.gt_sample_weights = wp.zeros(new_MS_GT, dtype=wp.float32)
                self.cap_gt_samples = new_MS_GT

            if grow_B:
                self.cap_blocks = new_MB

        # Dependent buffers (CG + energy calculators) and graph invalidation.
        if hasattr(self, "cg_solver") and self.cg_solver is not None:
            self.cg_solver.ensure_capacity()
            self.cg_solver.clear()
        self.line_search_graph = None

        # Only grow already-constructed calculators; lazy ones allocate on first use.
        if hasattr(self, "energy_calcs"):
            for ec in self.energy_calcs.values():
                if hasattr(ec, "ensure_capacity"):
                    ec.ensure_capacity()

        return True

    def clear(self):
        self._init_counters()
        self.cg_solver.clear()
        self.line_search_graph = None
        # Keep already-built calculators; ensure any newly enabled classes exist
        # only when needed (lazy) or rebuild missing entries for eager mode.
        if not getattr(self.config, "lazy_calculators", True):
            for k in self.config.energy_calcs:
                if k not in self.energy_calcs:
                    self.energy_calcs[k] = k(self)
                    if hasattr(self.energy_calcs[k], "ensure_capacity"):
                        self.energy_calcs[k].ensure_capacity()
        self.do_nothing = False

    def _ensure_energy_calculator(self, cls: type):
        """Instantiate calculator for cls if enabled and not yet built."""
        if cls is None or cls not in self.config.energy_calcs:
            return None
        ec = self.energy_calcs.get(cls)
        if ec is None:
            ec = cls(self)
            if hasattr(ec, "ensure_capacity"):
                ec.ensure_capacity()
            self.energy_calcs[cls] = ec
            logger.debug(f"Lazy-built energy calculator: {getattr(cls, 'name', cls)}")
        return ec

    def _get_energy_calculator(self, cls: type):
        if cls not in self.config.energy_calcs:
            return None
        if getattr(self.config, "lazy_calculators", True):
            return self._ensure_energy_calculator(cls)
        return self.energy_calcs.get(cls)

    def _refit_edge_bvh(self, x):
        with wp.ScopedDevice(self.device):
            compute_edge_bounds(
                self.n_edges,
                x,
                self.edges,
                self.edge_lowers,
                self.edge_uppers,
                0.0,
            )
            self.edge_bvh.refit()
        # self.edge_bvh = wp.Bvh(
        #     wp_slice(self.edge_lowers, 0, self.n_edges),
        #     wp_slice(self.edge_uppers, 0, self.n_edges),
        # )
    
    def _refit_tri_bvh(self, x):
        with wp.ScopedDevice(self.device):
            compute_tri_bounds(
                self.n_triangles,
                x,
                self.triangles,
                self.tri_lowers,
                self.tri_uppers,
                0.0,
            )
            # logger.debug(f"(tri_lowers, tri_uppers): {np.hstack((self.tri_lowers.numpy()[:10], self.tri_uppers.numpy()[:10]))}")
            self.tri_bvh.refit()
        # self.tri_bvh = wp.Bvh(
        #     wp_slice(self.tri_lowers, 0, self.n_triangles),
        #     wp_slice(self.tri_uppers, 0, self.n_triangles),
        # )
        
    # def _refit_gt_tri_bvh(self):
    #     compute_tri_bounds(
    #         self.n_gt_triangles,
    #         self.gt_vertices,
    #         self.gt_triangles,
    #         self.gt_tri_lowers,
    #         self.gt_tri_uppers,
    #     )
    #     self.gt_tri_bvh.refit()

    def get_vertices(self):
        return self.q.numpy()[: self.n_particles] / self.config.system_scale

    def register_mesh(
        self, V_gt: np.ndarray, F_gt: np.ndarray, V: np.ndarray, F: np.ndarray
    ):
        c = self.config

        V_gt = V_gt * c.system_scale
        V = V * c.system_scale

        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        gt_mesh = trimesh.Trimesh(vertices=V_gt, faces=F_gt, process=False)
        E: np.ndarray = mesh.edges_unique

        self.n_particles = V.shape[0]
        self.n_triangles = F.shape[0]
        self.n_edges = E.shape[0]
        self.n_gt_particles = V_gt.shape[0]
        self.n_gt_triangles = F_gt.shape[0]

        NP, NT, NE = self.n_particles, self.n_triangles, self.n_edges
        NP_GT, NT_GT = self.n_gt_particles, self.n_gt_triangles

        # Mesh-sized contact buffer: heuristic scales with edges; overflow
        # grows geometrically inside detect_contact. Clamp to max_blocks —
        # the hint is only initial capacity, not a hard requirement.
        bpe = int(getattr(c, "blocks_per_edge", 16))
        n_blocks_hint = max(
            int(getattr(c, "min_blocks", 1 << 16)),
            NE * bpe,
            NP * bpe,
        )
        n_blocks_hint = min(n_blocks_hint, int(c.max_blocks))
        self.ensure_capacity(
            n_particles=NP,
            n_faces=NT,
            n_edges=NE,
            n_gt_particles=NP_GT,
            n_gt_faces=NT_GT,
            n_gt_samples=c.max_gt_samples,
            n_blocks=n_blocks_hint,
        )

        wp_slice(self.q, 0, NP).assign(V)
        wp_slice(self.q_rest, 0, NP).assign(V)
        wp_slice(self.triangles, 0, NT).assign(F)
        wp_slice(self.edges, 0, NE).assign(E)

        # gt_samples = trimesh.sample.sample_surface(gt_mesh, c.max_gt_samples, seed=c.seed)[0]
        gt_samples = trimesh.sample.sample_surface_even(
            gt_mesh, c.max_gt_samples, seed=c.seed
        )[0]
        self.n_gt_samples = gt_samples.shape[0]
        NS_GT = self.n_gt_samples
        areas_gt = igl.doublearea(V_gt, F_gt) / 2

        wp_slice(self.gt_vertices, 0, NP_GT).assign(V_gt)
        wp_slice(self.gt_triangles, 0, NT_GT).assign(F_gt)
        wp_slice(self.gt_samples, 0, NS_GT).assign(gt_samples)
        wp_slice(self.gt_sample_weights, 0, NS_GT).fill_(areas_gt.sum() / NS_GT)

        areas = igl.doublearea(V, F) / 2
        logger.debug(f"min area = {np.min(areas)}, max area = {np.max(areas)}")
        if np.min(areas) < 1e-8:
            f_ids = np.where(areas < 1e-8)[0]
            # logger.warning(
            #     f"Triangle area is too small (id={f_ids}, area={areas[f_ids]}, indices={F[f_ids]}), may cause numerical instability"
            # )
            logger.warning(
                f"Triangle area is too small (min_area={np.min(areas)}), may cause numerical instability"
            )

        angles = igl.internal_angles(V, F)
        min_angle = np.min(angles) * 180 / np.pi
        logger.debug(f"min angle = {min_angle}")
        if min_angle <= c.min_angle_thres:
            self.do_nothing = True

        massmatrix = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)
        voronoi_areas = massmatrix.diagonal()
        wp_slice(self.voronoi_areas, 0, NP).assign(voronoi_areas)
        
        compute_edge_bounds(
            NE,
            self.q,
            self.edges,
            self.edge_lowers,
            self.edge_uppers,
            0.0,
        )
        self.edge_bvh = wp.Bvh(
            wp_slice(self.edge_lowers, 0, NE),
            wp_slice(self.edge_uppers, 0, NE),
        )
        compute_tri_bounds(
            NT,
            self.q,
            self.triangles,
            self.tri_lowers,
            self.tri_uppers,
            0.0,
        )
        self.tri_bvh = wp.Bvh(
            wp_slice(self.tri_lowers, 0, NT),
            wp_slice(self.tri_uppers, 0, NT),
        )
        # compute_tri_bounds(
        #     NT_GT,
        #     self.gt_vertices,
        #     self.gt_triangles,
        #     self.gt_tri_lowers,
        #     self.gt_tri_uppers,
        # )
        # self.gt_tri_bvh = wp.Bvh(
        #     wp_slice(self.gt_tri_lowers, 0, NT_GT),
        #     wp_slice(self.gt_tri_uppers, 0, NT_GT),
        # )
        
        self.mesh_bvh = wp.Mesh(
            wp_slice(self.q, 0, NP),
            wp_slice(self.triangles.reshape(-1), 0, NT * 3),
        )
        self.gt_mesh_bvh = wp.Mesh(
            wp_slice(self.gt_vertices, 0, NP_GT),
            wp_slice(self.gt_triangles.reshape(-1), 0, NT_GT * 3),
        )

        self._energy_preprocess(V, F)

    def _energy_preprocess(self, V, F):
        for k in self.config.energy_calcs:
            ec: EnergyCalculator = self._ensure_energy_calculator(k)
            if ec is not None:
                ec.preprocess(V, F)

    def _update_vertex_target(self):
        ec_mesh2gt: Mesh2GTDistanceEnergyCalculator = self._get_energy_calculator(
            Mesh2GTDistanceEnergyCalculator
        )
        if ec_mesh2gt is not None:
            ec_mesh2gt.update_target(self.q)
            
        ec_mesh2gt: Mesh2GTDistanceBvhEnergyCalculator = self._get_energy_calculator(
            Mesh2GTDistanceBvhEnergyCalculator
        )
        if ec_mesh2gt is not None:
            ec_mesh2gt.update_target(self.q)

        ec_gt2mesh: GT2MeshDistanceEnergyCalculator = self._get_energy_calculator(
            GT2MeshDistanceEnergyCalculator
        )
        if ec_gt2mesh is not None:
            ec_gt2mesh.update_target(self.q)
            
        ec_gt2mesh: GT2MeshDistanceBvhEnergyCalculator = self._get_energy_calculator(
            GT2MeshDistanceBvhEnergyCalculator
        )
        if ec_gt2mesh is not None:
            ec_gt2mesh.update_target(self.q)

    def _detect_contact(self):
        ec: CollisionEnergyCalculator = self._get_energy_calculator(
            CollisionEnergyCalculator
        )
        if ec is not None:
            ec.detect_contact(self.q)
        ec: CollisionBvhEnergyCalculator = self._get_energy_calculator(
            CollisionBvhEnergyCalculator
        )
        if ec is not None:
            self._refit_edge_bvh(self.q)
            self._refit_tri_bvh(self.q)
            ec.detect_contact(self.q)
            
        wp.copy(self.q_prev_detection, self.q, count=self.n_particles)

    def _init_energy_calcs(self):
        """Create calculator dict. Eager mode builds all; lazy mode defers to first use."""
        self.energy_calcs = {}
        if not getattr(self.config, "lazy_calculators", True):
            for ec in self.config.energy_calcs:
                self.energy_calcs[ec] = ec(self)

    def _iter_energy_calcs(self):
        """Yield enabled calculators in config order, constructing lazily if needed."""
        for k in self.config.energy_calcs:
            ec = self._ensure_energy_calculator(k)
            if ec is not None:
                yield ec

    def _compute_energy(self, verbose=False):
        self.energy.zero_()
        last_energy = 0.0
        energy_vals = {}
        for ec in self._iter_energy_calcs():
            ec: EnergyCalculator
            ec.compute_energy(self.q, self.energy)
            if verbose:
                new_energy = self.energy.numpy()[0]
                energy_vals[ec.name] = new_energy - last_energy
                last_energy = new_energy
        if verbose:
            logger.debug(
                f"Energy values: {', '.join(f'{k}: {v:.3e}' for k, v in energy_vals.items())}"
            )

    def _compute_diff(self):
        self.grad.zero_()
        self.hess_diag.zero_()
        for ec in self._iter_energy_calcs():
            ec.compute_diff(self.q, -1.0, self.grad, self.hess_diag)

    def _compute_hess_dx(self, dx: wp.array, hess_dx: wp.array):
        hess_dx.zero_()
        for ec in self._iter_energy_calcs():
            ec.compute_hess_dx(self.q, dx, hess_dx)

    def _clamp_p(self):
        c = self.config
        wp.launch(
            kernel=clamp_p_kernel,
            dim=self.n_particles,
            inputs=[
                self.p,
                self.q_prev_newton,
                self.q_prev_detection,
                c.contact_detection_radius,
            ],
            device=self.device,
        )

    def _ccd(self):
        self.ccd_step.fill_(1.0)

        ec = self._get_energy_calculator(CollisionEnergyCalculator)
        # assert ec is not None, "System does not have CollisionEnergyCalculator"
        ec: CollisionEnergyCalculator
        if ec is not None:
            ec.ccd(self.q, self.p, self.ccd_step)

        ec = self._get_energy_calculator(CollisionWoBufferEnergyCalculator)
        # assert ec is not None, "System does not have CollisionEnergyCalculator"
        ec: CollisionWoBufferEnergyCalculator
        if ec is not None:
            ec.ccd(self.q, self.p, self.ccd_step)
            
        ec = self._get_energy_calculator(CollisionBvhEnergyCalculator)
        # assert ec is not None, "System does not have CollisionEnergyCalculator"
        ec: CollisionBvhEnergyCalculator
        if ec is not None:
            ec.ccd(self.q, self.p, self.ccd_step)

        if self.config.debug:
            ccd_step_val = self.ccd_step.numpy()[0]
            logger.debug(f"CCD step: {ccd_step_val:.6f}")

    def _line_search(self):
        wp.copy(self.energy_prev, self.energy, count=1)
        for n_halves in range(self.config.n_line_search_iters):
            wp.launch(
                kernel=line_search_kernel,
                dim=self.n_particles,
                inputs=[
                    self.q_prev_newton,
                    self.p,
                    self.ccd_step,
                    n_halves,
                    self.energy_prev,
                    self.energy,
                ],
                outputs=[self.q],
            )
            self._compute_energy()

    def _reject_line_search_if_needed(self):
        """Restore q_prev_newton if no candidate decreased a finite objective.

        Must run on the host after the (possibly graph-captured) line search.
        Without this, the final half-step remains even when it increased energy.
        """
        e = float(self.energy.numpy()[0])
        e_prev = float(self.energy_prev.numpy()[0])
        if not (e == e) or not (e_prev == e_prev) or not (e < e_prev):
            wp.launch(
                kernel=restore_q_kernel,
                dim=self.n_particles,
                inputs=[self.q_prev_newton],
                outputs=[self.q],
            )
            wp.copy(self.energy, self.energy_prev, count=1)
            if self.config.debug:
                logger.warning(
                    f"Line search rejected: energy {e:.3e} vs prev {e_prev:.3e}; restored q"
                )

    def step(
        self,
        eval=False,
        eval_CDs=None,
        eval_HDs=None,
        gt_V=None,
        gt_F=None,
        eval_F=None,
    ):
        if self.do_nothing:
            logger.warning("Doing nothing in step() because of small internal angles")
            return

        """One Newton's iteration"""
        c = self.config
        
        self._update_vertex_target()
        if c.detect_contact_every == "step":
            self._detect_contact()

        for newton_iter in range(c.n_newton_iters):
            wp.copy(self.q_prev_newton, self.q, count=self.n_particles)

            # --------------------- Compute linear system --------------------- #
            # self._update_vertex_target()
            if c.detect_contact_every == "newton":
                self._detect_contact()
            self._compute_energy()
            self._compute_diff()

            # --------------------- CG linear solve --------------------- #
            self.cg_solver.solve()

            # --------------------- CCD --------------------- #
            self._clamp_p()
            self._ccd()

            # --------------------- Line search --------------------- #
            if c.use_cuda_graph:
                if not self.line_search_graph:
                    wp.capture_begin()
                    self._line_search()
                    self.line_search_graph = wp.capture_end()
                wp.capture_launch(self.line_search_graph)
            else:
                self._line_search()
            # Outside the CUDA graph: reject non-improving / non-finite steps.
            self._reject_line_search_if_needed()

            if c.debug:
                self._compute_energy(verbose=True)
                energy_val = self.energy.numpy()[0]
                energy_diff = energy_val - self.energy_prev.numpy()[0]
                logger.debug(
                    f"Newton iter {newton_iter}: energy = {energy_val:.3e}, energy_diff = {energy_diff:.3e}"
                )

            if eval:
                eval_V = self.get_vertices().astype(np.float64)
                cd, hd = compute_igl_CD_HD(gt_V, gt_F, eval_V, eval_F)
                eval_CDs.append(cd)
                eval_HDs.append(hd)
