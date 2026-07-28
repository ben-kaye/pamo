from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import Stage3System

import numpy as np
import warp as wp

from .kernels.solver_kernels import *
from .kernels.utils_kernels import *
from .utils import stage3_logger as logger
from .utils import wp_slice


class CGSolver:
    def __init__(self, system: Stage3System):
        self.system = system
        self.graph = None
        self.r = None
        self.v = None
        self.A_v = None
        self.v_A_v = None
        self.z = None
        self.z_r_last = None
        self.z_r = None
        self._cap_particles = 0
        self.ensure_capacity()

    def ensure_capacity(self):
        """(Re)allocate CG workspaces to match system.cap_particles."""
        MP = max(int(self.system.cap_particles), 0)
        if MP <= 0:
            # Keep scalar workspaces so early-exit paths can still touch them.
            with wp.ScopedDevice(self.system.device):
                if self.v_A_v is None:
                    self.v_A_v = wp.zeros(1, dtype=wp.float32)
                    self.z_r_last = wp.zeros(1, dtype=wp.float32)
                    self.z_r = wp.zeros(1, dtype=wp.float32)
                if self.r is None:
                    self.r = wp.zeros(0, dtype=wp.vec3)
                    self.v = wp.zeros(0, dtype=wp.vec3)
                    self.A_v = wp.zeros(0, dtype=wp.vec3)
                    self.z = wp.zeros(0, dtype=wp.vec3)
            return
        if self._cap_particles >= MP and self.r is not None:
            return

        with wp.ScopedDevice(self.system.device):
            self.r = wp.zeros(MP, dtype=wp.vec3)
            self.v = wp.zeros(MP, dtype=wp.vec3)
            self.A_v = wp.zeros(MP, dtype=wp.vec3)
            self.z = wp.zeros(MP, dtype=wp.vec3)
            if self.v_A_v is None:
                self.v_A_v = wp.zeros(1, dtype=wp.float32)
                self.z_r_last = wp.zeros(1, dtype=wp.float32)
                self.z_r = wp.zeros(1, dtype=wp.float32)
        self._cap_particles = MP
        self.graph = None

    def clear(self):
        self.graph = None

    def _launch_main_loop(self):
        s = self.system
        c = s.config
        NP = s.n_particles

        for cg_iter in range(c.n_cg_iters):
            self.A_v.zero_()
            s._compute_hess_dx(self.v, self.A_v)
            
            self.v_A_v.zero_()
            wp.launch(
                kernel=compute_dot_kernel,
                dim=NP,
                inputs=[self.v, self.A_v],
                outputs=[self.v_A_v],
            )
            
            if c.debug and not c.use_cuda_graph:
                v_A_v_val = self.v_A_v.numpy()[0]
                # logger.debug(f"v^T A v = {v_A_v_val}")
                if v_A_v_val < 0:
                    logger.warning(f"v^T A v = {v_A_v_val} < 0, Hessian is not positive definite")
            
            wp.copy(self.z_r_last, self.z_r, count=1)
            self.z_r.zero_()
            wp.launch(
                kernel=update_p_r_z_compute_zr_kernel,
                dim=NP,
                inputs=[
                    self.v,
                    self.A_v,
                    self.v_A_v,
                    self.z_r_last,
                    s.hess_diag,
                ],
                outputs=[
                    s.p,
                    self.r,
                    self.z,
                    self.z_r,
                ],
            )
            
            wp.launch(
                kernel=update_v_kernel,
                dim=NP,
                inputs=[self.z, self.z_r_last, self.z_r, self.v],
            )
            

    def solve(self):
        """
        Copy from self.system:
            - Gradient: self.system.grad => self.r
            - Diagonal of Hessian: self.system.hess_diag
        Call energy caculators to compute:
            - Hessian-vector product: => self.A_v
        Save the solution of A p = b into self.system.p
        """

        s = self.system
        c = s.config
        NP = s.n_particles

        self.ensure_capacity()
        
        if c.debug and not c.use_cuda_graph:
            hess_diag_np = s.hess_diag.numpy()
            if np.any(np.isnan(hess_diag_np)):
                nan_indices = np.where(np.isnan(hess_diag_np))[0]
                logger.error(f"NaN found in Hessian diagonal at indices: {nan_indices}")
            
            assert not np.isnan(s.grad.numpy()).any(), "Initial gradient is NaN"
            assert not np.isnan(s.hess_diag.numpy()).any(), "Initial Hessian diagonal is NaN"

        with wp.ScopedDevice(self.system.device):
            s.p.zero_()
            wp.copy(self.r, s.grad, count=NP)

            # ------------------------- Preconditioning ------------------------- #
            if c.cg_precond == "jacobi":
                wp.launch(
                    kernel=compute_block_diag_inv_kernel,
                    dim=NP,
                    inputs=[s.hess_diag, self.r],
                    outputs=[self.z],
                )
            else:
                wp.copy(self.z, self.r, count=NP)

            wp.copy(self.v, self.z, count=NP)

            self.z_r.zero_()
            wp.launch(
                kernel=compute_dot_kernel,
                dim=NP,
                inputs=[self.z, self.r],
                outputs=[self.z_r],
            )
            z_r_start = float(self.z_r.numpy()[0])

            if c.debug and not c.use_cuda_graph:
                assert not np.isnan(self.r.numpy()).any(), "Initial residual is NaN"
                assert not np.isnan(z_r_start), "Initial residual norm z_r is NaN"

            # Already solved or non-finite residual: leave p = 0.
            if not np.isfinite(z_r_start) or abs(z_r_start) < 1e-30:
                if c.debug:
                    logger.debug(
                        f"CG early exit: initial z^T r = {z_r_start:.3e}"
                    )
                return

            # ------------------------- Main loop ------------------------- #
            if c.use_cuda_graph and self.graph is None:
                wp.capture_begin()
                self._launch_main_loop()
                self.graph = wp.capture_end()

            if c.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._launch_main_loop()

            if c.debug:
                z_r_end = float(self.z_r.numpy()[0])
                ratio = z_r_end / z_r_start if abs(z_r_start) > 1e-30 else float("nan")
                logger.debug(
                    f"CG error from {z_r_start:.3e} to {z_r_end:.3e}, ratio: {ratio:.3e}"
                )
