import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomTransform(object):
    def __init__(self, max_angles=30. / 360 * 2. * np.pi):
        self.max_angles = max_angles

    def __call__(self, results):
        points = results['points']
        surface = results['surface']

        roll = self.max_angles * (2 * torch.rand(1) - 1)
        yaw = self.max_angles * (2 * torch.rand(1) - 1)
        pitch = self.max_angles * (2 * torch.rand(1) - 1)

        tensor_0 = torch.zeros(1)
        tensor_1 = torch.ones(1)

        RX = torch.stack([
            torch.stack([tensor_1, tensor_0, tensor_0]),
            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3, 3)

        RY = torch.stack([
            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3, 3)

        RZ = torch.stack([
            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)

        points = torch.einsum('cpd,de->cpe', points, R).detach()
        surface = torch.einsum('cpd,de->cpe', surface, R).detach()

        results['points'] = points
        results['surface'] = surface
        results['R'] = R

        if 'slice_pts' in results:
            slice_pts = results['slice_pts']
            slice_pts = torch.mm(slice_pts, R).detach()
            results['slice_pts'] = slice_pts

        return results
