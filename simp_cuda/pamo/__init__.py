import os
import copy
import warnings
import torch
from torch import nn
from torch.autograd import Function
import trimesh
import numpy
from pdmc import DMC
import time
import torch.nn.functional as F
from . import _C
import numpy as np
import trimesh
import pamo_safe_project
import torchcumesh2sdf


def _resolve_device(points):
    """Prefer the input tensor device; fall back to current CUDA device."""
    if isinstance(points, torch.Tensor) and points.is_cuda:
        return points.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    raise RuntimeError("PaMO requires a CUDA tensor or available CUDA device")


class PaMO(nn.Module):
    def __init__(self, input_mesh, use_stage1=True, use_stage3=True, device=None):
        super().__init__()
        pamo = _C.CUDSP_Free()

        self.use_stage1 = use_stage1
        self.use_stage3 = use_stage3
        # Optional sticky device; otherwise taken from points on each run().
        self.device = torch.device(device) if device is not None else None

        print("Stage1 : ", self.use_stage1)
        print("Stage3 : ", self.use_stage3)

        self.bbox = input_mesh.bounding_box.bounds
        diameter = np.abs(self.bbox[1] - self.bbox[0]).max()
        scale = 1.0 / diameter
        self.gt_mesh = copy.deepcopy(input_mesh)

        # Stage 3 system + stage 1 DMC are built on first use.
        self.config = None
        self.system = None
        self.vol2mesh = None
        self._stage1_device = None
        self._stage3_device = None

        class DSPFunction(Function):
            @staticmethod
            def forward(ctx, points, triangles, vertices_undo, num_vertices_undo, scale, threshold, is_stuck, init):
                verts, faces, verts_occ, verts_map, verts_undo = pamo.forward(
                    points, triangles, vertices_undo, num_vertices_undo,
                    scale, threshold, is_stuck, init,
                )
                ctx.points = points
                ctx.triangles = triangles
                return verts, faces, verts_occ, verts_map, verts_undo

        self.func = DSPFunction
        # mesh2vol params (band/margin recomputed when R changes for a run)
        self.R = 256
        self.band = 3 / self.R
        self.margin = self.band * 2 + 1
        self.target_faces = None

    def _set_stage1_resolution(self, R):
        """Keep band/margin consistent with Dual-MC grid resolution."""
        self.R = int(R)
        self.band = 3.0 / self.R
        self.margin = self.band * 2.0 + 1.0

    def _ensure_stage1(self, device):
        """Lazily construct Dual Marching Cubes on the requested device."""
        if not self.use_stage1:
            return
        dev = torch.device(device)
        if self.vol2mesh is None or self._stage1_device != dev:
            self.vol2mesh = DMC(dtype=torch.float32).to(dev)
            self._stage1_device = dev

    def _ensure_stage3(self, device):
        """Lazily construct Stage3System on the requested device."""
        if not self.use_stage3:
            return
        dev = torch.device(device)
        dev_str = str(dev)
        if self.config is None:
            self.config = pamo_safe_project.config.Stage3Config()
        if self.system is None or self._stage3_device != dev:
            self.system = pamo_safe_project.system.Stage3System(
                self.config, device=dev_str)
            self._stage3_device = dev

    def tri_area(self, v0, v1, v2):
        cross_prod = torch.cross(v1 - v0, v2 - v0)
        return 0.5 * torch.norm(cross_prod, dim=1)


    def preprocess_mesh(self, points, triangles, band, margin):
        tris = points[triangles]
        tris = tris.cpu().numpy()
        
        tris_mean = tris.mean(axis=1).mean(axis=0)
        tris = tris - tris_mean
        
        tris_min = tris.min(0).min(0)
        tris = tris - tris_min
        tris_max = tris.max()
        tris = (tris / tris_max + band) / margin
        
        return tris, tris_min, tris_max, tris_mean

    
    def remesh(self, tris, tris_min, tris_max, tris_mean, device):
        d = torchcumesh2sdf.get_sdf(tris, self.R, self.band)
        d = d - 0.9 / self.R
        
        v, f = self.vol2mesh(d, return_quads=False) #Dual MC

        v, f = v.cpu().numpy(), f.cpu().numpy()
        v = (((v * self.R +0.5)/(self.R+1)* self.margin - self.band) * tris_max + tris_min)
        
        v = torch.from_numpy(v).float().to(device)
        f = torch.from_numpy(f).int().to(device)
        
        return v, f

    def run(self, points, triangles, ratio, tolerance=4, threshold=1e-3, iter=1000000,
            min_faces=None, min_verts=None):
        # min_faces floors the *face* target (not vertex count). Default 0 so
        # ratio alone sets the target. min_verts is a deprecated alias.
        if min_faces is not None and min_verts is not None:
            raise TypeError(
                "PaMO.run() got both min_faces and min_verts; "
                "use min_faces only (min_verts is a deprecated alias)")
        if min_verts is not None:
            warnings.warn(
                "PaMO.run(min_verts=...) is deprecated; use min_faces=... "
                "(the value floors the face target, not vertex count)",
                DeprecationWarning,
                stacklevel=2,
            )
            min_faces = min_verts
        if min_faces is None:
            min_faces = 0
        self.target_faces = max(int(ratio * len(triangles)), min_faces)
        print("Target faces : {}".format(self.target_faces))

        device = self.device if self.device is not None else _resolve_device(points)
        if points.device != device:
            points = points.to(device)
        if triangles.device != device:
            triangles = triangles.to(device)

        # Choose stage-1 resolution and recompute band/margin *before* normalize.
        # Aggressive schedule for coarse targets: stage-1 remesh sets a floor
        # that stage-2 cannot get under (§3.3).
        if self.use_stage1:
            if self.target_faces <= 50:
                self._set_stage1_resolution(64)
            elif self.target_faces <= 1000:
                self._set_stage1_resolution(128)
            else:
                self._set_stage1_resolution(256)
        else:
            self._set_stage1_resolution(256)

        # scale the input mesh
        tris, tris_min, tris_max, tris_mean = self.preprocess_mesh(
            points, triangles, self.band, self.margin)
        tris = torch.tensor(tris, dtype=torch.float32, device=device)

        # stage1 (Remeshing)
        if self.use_stage1:
            self._ensure_stage1(device)
            start_stage1 = time.time()
            verts, faces = self.remesh(tris, tris_min, tris_max, tris_mean, device)
            end_stage1 = time.time()
            print(f"Time for Remeshing: {end_stage1 - start_stage1} sec")
        else:
            verts = points - torch.from_numpy(tris_mean).to(device)
            faces = triangles

        # stage2 (Simplification) — skip if already at/under target
        start_stage2 = time.time()
        if faces.shape[0] > self.target_faces and faces.shape[0] > 10:
            verts_undo = torch.empty(0, dtype=torch.int32, device=device)
            n_verts_undo = 0
            count = 0
            is_stuck = 0
            # Working threshold may relax when progress stalls (§3.3).
            thr = float(threshold)
            scale = max(
                max(verts[:, 0].max() - verts[:, 0].min(),
                    verts[:, 1].max() - verts[:, 1].min()),
                verts[:, 2].max() - verts[:, 2].min(),
            )
            init = True
            for it in range(iter):
                num_faces_prev = faces.shape[0]
                verts, faces, verts_occ, verts_map, verts_undo = self.func.apply(
                    verts, faces, verts_undo, n_verts_undo, scale, thr,
                    is_stuck, init)
                init = False
                n_verts_undo = verts_undo.shape[0]

                verts = verts[verts_occ.view(-1).bool()]
                faces = faces[faces[:, 0] >= 0]
                faces[:, 0] = verts_map[faces[:, 0].long()].view(-1)
                faces[:, 1] = verts_map[faces[:, 1].long()].view(-1)
                faces[:, 2] = verts_map[faces[:, 2].long()].view(-1)

                num_faces_current = faces.shape[0]

                if num_faces_current <= self.target_faces or num_faces_current <= 10:
                    break

                if num_faces_current == num_faces_prev:
                    count += 1
                else:
                    count = 0
                    is_stuck = 0

                if count >= 2:
                    is_stuck = 1

                if count == tolerance:
                    print("Not enough edges available to be collapsed.")
                    break

        end_stage2 = time.time()
        verts = verts.cpu().numpy() + tris_mean
        faces = faces.cpu().numpy()
        print(f"Time for Simplification: {end_stage2 - start_stage2} sec")
        
        # stage3 (Safe projection)
        if self.use_stage3:
            self._ensure_stage3(device)
            stage2_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            verts, faces = pamo_safe_project.process(
                self.gt_mesh.vertices,
                self.gt_mesh.faces,
                stage2_mesh.vertices,
                stage2_mesh.faces,
                5,
                system=self.system,  # if provided, reuse the same system
                config=self.config,
            )

        return verts, faces


class PaSP(nn.Module):
    def __init__(self):
        super().__init__()
        sp = _C.CUDSP()

        class PaSPFunction(Function):
            @staticmethod
            def forward(ctx, points, triangles, scale, threshold, init):
                verts, faces, verts_occ, verts_map = sp.forward(points, triangles, scale, threshold, init)
                ctx.points = points
                ctx.triangles = triangles
                return verts, faces, verts_occ, verts_map

        self.func = PaSPFunction

    def run(self, points, triangles, threshold=0.001, iter=1000):
        verts = points
        faces = triangles
        scale = max(max(verts[:,0].max()-verts[:,0].min(), verts[:,1].max()-verts[:,1].min()), verts[:,2].max()-verts[:,2].min())
        init = True
        for it in range(iter):
            num_faces = faces.shape[0]
            verts, faces, verts_occ, verts_map = self.func.apply(verts, faces, scale, threshold, init)
            verts = verts[verts_occ.view(-1).bool()]
            faces = faces[faces[:, 0] >= 0]
            faces[:,0] = verts_map[faces[:,0].long()].view(-1)
            faces[:,1] = verts_map[faces[:,1].long()].view(-1)
            faces[:,2] = verts_map[faces[:,2].long()].view(-1)
            init = False
            if faces.shape[0] == num_faces:
                print("Converged at iteration {}".format(it))
                break

        return verts, faces
