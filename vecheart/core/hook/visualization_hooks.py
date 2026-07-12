import os
from typing import List

import numpy as np
import torch
import trimesh
from reimu import runner
from reimu.runner import HOOKS, Hook
from reimu.runner.dist_utils import master_only


@HOOKS.register_module()
class VisualizationHook(Hook):
    """
    Evaluate lambda expressions. It is really an ugly hook...
    """
    def __init__(self,
                 resolution: int = 128,
                 by_epoch: bool = False,
                 interval: int = 1,
                 return_gt: bool = True,
                 sax_only: bool = False,
                 out_dir=None) -> None:
        self.resolution = resolution

        self.by_epoch = by_epoch
        self.interval = interval
        self.return_gt = return_gt
        self.sax_only = sax_only

        self.out_dir = out_dir

    @master_only
    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = os.path.join(runner.work_dir, "Reconstruction_training")
        os.makedirs(self.out_dir, exist_ok=True)

    def _should_visualize(self, runner):
        if self.by_epoch:
            check_time = self.every_n_epochs
        else:
            check_time = self.every_n_iters

        if not check_time(runner, self.interval):
            return False

        return True

    def recon_mesh(self, runner, data):
        return runner.model.module.reconstruct_mesh(data)

    @master_only
    def _do_visualize(self, runner):
        data = runner.model.module.latest_data
        file_name = os.path.basename(data['sample_metas'][0]['sample_info']).replace('_sdf.npz', '')
        if 'parts' in data:
            parts = data['parts'][0].cpu().numpy()
        else:
            parts = [0]

        with torch.no_grad():
            data['surface'] = data['surface'][0:1]
            if 'slice_pts' in data:
                data['slice_pts'] = data['slice_pts'][0:1]
            data['resolution'] = self.resolution

            meshes = self.recon_mesh(runner, data)

        if not isinstance(meshes, List):
            meshes = [meshes]

        if 'slice_pts' in data:
            slice_pts = data['slice_pts'][0].cpu().numpy()

            slice_seps = runner.model.module.shape_cfgs.get(
                'slice_seps',
                [0, 7040, 10112, 12672, 13568, 14016]
            )
            slice_offsets = runner.model.module.shape_cfgs.get(
                'slice_offsets',
                [
                    [0, 1024, 1920, 7040],
                    [0, 512, 1024, 3072],
                    [0, 512, 512, 2560],
                    [0, 448, 896],
                    [0, 448, 448]
                ]
            )

            slice_pcs = []
            for part in parts:
                slice_pts_part = slice_pts[slice_seps[part]:slice_seps[part + 1]]
                slice_offsets_part = slice_offsets[part]
                if self.sax_only and part < 3:
                    slice_pts_part = slice_pts_part[slice_offsets_part[2]:slice_offsets_part[3]]
                slice_pcs.append(trimesh.points.PointCloud(slice_pts_part))
        else:
            slice_pcs = None

        if self.return_gt:
            gt_meshes = []

            for part in parts:
                gt_mesh_name = data['sample_metas'][0]['sample_info'].replace('_sdf.npz', f'_{part + 1}.obj')
                gt_mesh = trimesh.load(gt_mesh_name)

                shifts = np.array([13.59974129, 9.31580232, -1.01333806])
                gt_mesh.apply_translation(-shifts)
                gt_mesh.apply_scale(1 / 100)

                if 'R' in data:
                    rot_mat = np.eye(4)
                    rot_mat[:3, :3] = data['R'][0].cpu().numpy()
                    gt_mesh.apply_transform(rot_mat.T)

                gt_meshes.append(gt_mesh)

            return meshes, gt_meshes, file_name, parts, slice_pcs

        return meshes, None, file_name, parts, slice_pcs

    @master_only
    def after_train_epoch(self, runner: 'runner.BaseRunner'):
        if self.by_epoch and self._should_visualize(runner):
            meshes, gt_meshes, file_name, parts, slice_pcs = self._do_visualize(runner)

            for idx in range(len(meshes)):
                out_name = os.path.join(
                    self.out_dir,
                    f'epoch_{runner.epoch + 1:06d}_{file_name}_{parts[idx] + 1}_pred.obj'
                )
                meshes[idx].export(out_name)
            if gt_meshes is not None:
                for idx in range(len(gt_meshes)):
                    out_name = os.path.join(
                        self.out_dir,
                        f'epoch_{runner.epoch + 1:06d}_{file_name}_{parts[idx] + 1}_gt.obj'
                    )
                    gt_meshes[idx].export(out_name)
            if slice_pcs is not None:
                for idx in range(len(slice_pcs)):
                    out_name = os.path.join(
                        self.out_dir,
                        f'epoch_{runner.epoch + 1:06d}_{file_name}_{parts[idx] + 1}_slice_pts.obj'
                    )
                    slice_pcs[idx].export(out_name)

    @master_only
    def after_train_iter(self, runner: 'runner.BaseRunner'):
        if not self.by_epoch and self._should_visualize(runner):
            meshes, gt_meshes, file_name, parts, slice_pcs = self._do_visualize(runner)

            for idx in range(len(meshes)):
                out_name = os.path.join(
                    self.out_dir,
                    f'iter_{runner.iter + 1:06d}_{file_name}_{parts[idx] + 1}_pred.obj'
                )
                meshes[idx].export(out_name)
            if gt_meshes is not None:
                for idx in range(len(gt_meshes)):
                    out_name = os.path.join(
                        self.out_dir,
                        f'iter_{runner.iter + 1:06d}_{file_name}_{parts[idx] + 1}_gt.obj'
                    )
                    gt_meshes[idx].export(out_name)
            if slice_pcs is not None:
                for idx in range(len(slice_pcs)):
                    out_name = os.path.join(
                        self.out_dir,
                        f'iter_{runner.iter + 1:06d}_{file_name}_{parts[idx] + 1}_slice_pts.obj'
                    )
                    slice_pcs[idx].export(out_name)
