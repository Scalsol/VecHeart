import warnings
from typing import List

import torch
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from torchmetrics import Metric


def to_tensor(something, dtype=torch.float, device="cuda"):
    return torch.tensor(something, dtype=dtype, device=device)


class ChamferDistance(Metric):
    full_state_update = False

    def __init__(self, num_classes, num_points=30000):
        super(ChamferDistance, self).__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.num_classes = num_classes
        self.num_points = num_points
        self.add_state("steps", default=torch.zeros(1))
        self.add_state("chamdis", default=torch.zeros((num_classes,)))
        self.add_state("nc", default=torch.zeros((num_classes,)))

        self.instance_stats = []

    def update(self, pred_mesh, gt_mesh):
        self.steps += 1

        stats = self.compute_stats(pred_mesh, gt_mesh)
        self.chamdis += stats[0]
        self.nc += stats[1]

        self.instance_stats.append(stats)

    def compute(self):
        return self.chamdis / self.steps, self.nc / self.steps

    def compute_stats(self, pred_meshes, gt_meshes):
        """
        Every mesh is a PyTorch3D Mesh object, contains num_classes meshes for each sample.
        """
        if not isinstance(pred_meshes, Meshes):
            if not isinstance(pred_meshes, List):
                pred_meshes = [pred_meshes]
            vertices = [to_tensor(pred_mesh.vertices) for pred_mesh in pred_meshes]
            faces = [to_tensor(pred_mesh.faces, dtype=torch.int64) for pred_mesh in pred_meshes]
            pred_meshes = Meshes(vertices, faces)

        if not isinstance(gt_meshes, Meshes):
            if not isinstance(gt_meshes, List):
                gt_meshes = [gt_meshes]
            vertices = [to_tensor(gt_mesh.vertices) for gt_mesh in gt_meshes]
            faces = [to_tensor(gt_mesh.faces, dtype=torch.int64) for gt_mesh in gt_meshes]
            gt_meshes = Meshes(vertices, faces)

        pred_meshes, gt_meshes = pred_meshes[:self.num_classes], gt_meshes[:self.num_classes]

        chamfers = torch.zeros(self.num_classes, device=pred_meshes.device, dtype=torch.float32)
        ncs = torch.zeros(self.num_classes, device=pred_meshes.device, dtype=torch.float32)

        valid = pred_meshes.valid
        if not valid.all():
            warnings.warn("Has a mesh that has 0 vertices.")

        pred_pts, pred_normals = sample_points_from_meshes(pred_meshes, self.num_points, return_normals=True)
        gt_pts, gt_normals = sample_points_from_meshes(gt_meshes, self.num_points, return_normals=True)

        _chamfer, _nc = chamfer_distance(
            x=pred_pts[valid],
            y=gt_pts[valid],
            x_normals=pred_normals[valid],
            y_normals=gt_normals[valid],
            batch_reduction=None
        )
        _nc = 1 - _nc

        chamfers[valid] = _chamfer
        ncs[valid] = _nc

        return chamfers, ncs