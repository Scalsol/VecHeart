import torch
import torch.nn as nn
from pytorch3d.loss import (mesh_edge_loss, mesh_laplacian_smoothing,
                            mesh_normal_consistency)

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def penalty_loss(pred, target, use_l2=False):
    """Smooth L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if use_l2:
        pred = (pred ** 2).sum(-1).sqrt()
    else:
        pred = pred.abs().sum(-1)

    return pred


@LOSSES.register_module()
class PenaltyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_l2=False):
        super(PenaltyLoss, self).__init__()

        self.loss_weight_exp = eval(loss_weight) if isinstance(loss_weight, str) \
            else lambda epoch: loss_weight
        self.loss_weight = None
        self.use_l2 = use_l2

    def forward(self, tensor, weight=None, avg_factor=None):
        """

        Args:
            tensor (torch.Tensor): shape: [B, N, C]
            weight:
            avg_factor:

        Returns:

        """
        return self.loss_weight * penalty_loss(tensor,
                                               None,
                                               weight,
                                               use_l2=self.use_l2,
                                               reduction='mean',
                                               avg_factor=avg_factor)

    def extra_repr(self):
        return f"loss_weight={self.loss_weight}"


@LOSSES.register_module()
class LatentLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_l2=False):
        super(LatentLoss, self).__init__()

        self.loss_weight = loss_weight
        self.use_l2 = use_l2

    def forward(self, tensor, weight=None, avg_factor=None):
        """

        Args:
            tensor (torch.Tensor): shape: [B, N, C]
            weight:
            avg_factor:

        Returns:

        """
        return self.loss_weight * penalty_loss(tensor,
                                               None,
                                               weight,
                                               use_l2=self.use_l2,
                                               reduction='mean',
                                               avg_factor=avg_factor)

    def extra_repr(self):
        return f"loss_weight={self.loss_weight}"


def border_penalty(tensor, border=1.0):
    return (tensor.abs().max() - border).relu().mean()


@LOSSES.register_module()
class BorderLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BorderLoss, self).__init__()

        self.loss_weight = loss_weight

    def forward(self, tensor, border=1.0):
        """

        Args:
            tensor (torch.Tensor): shape: [B, N, 3]
            border (float): the absolute border

        Returns:

        """
        return self.loss_weight * border_penalty(tensor, border)

    def extra_repr(self):
        return f"loss_weight={self.loss_weight}"


def points_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


@LOSSES.register_module()
class EikonalLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()

        self.loss_weight = loss_weight

    def forward(self, input, output):
        """
        Args:
            input (torch.Tensor): the input
            output (torch.Tensor): the output

        Returns:

        """
        grad = points_gradient(input, output)

        return self.loss_weight * (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()

    def extra_repr(self):
        return f"loss_weight={self.loss_weight}"


@LOSSES.register_module()
class MeshRegularizationLoss(nn.Module):
    def __init__(self, loss_weights=(0.1, 0.1, 1.0)):
        super(MeshRegularizationLoss, self).__init__()

        self.loss_weights = loss_weights

    def forward(self, meshes):
        loss = torch.tensor(0).float().cuda()

        if self.loss_weights[0] > 0:
            loss += self.loss_weights[0] * mesh_laplacian_smoothing(meshes, method="uniform")
        if self.loss_weights[1] > 0:
            loss += self.loss_weights[1] * mesh_normal_consistency(meshes)
        if self.loss_weights[2] > 0:
            loss += self.loss_weights[2] * mesh_edge_loss(meshes)

        return loss

    def extra_repr(self):
        return f'loss_weights={self.loss_weights}'
