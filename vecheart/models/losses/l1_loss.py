import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        if isinstance(target, float):
            target = torch.ones_like(pred) * target

        loss = self.loss_weight * l1_loss(pred, target, weight, avg_factor=avg_factor)
        return loss

    def extra_repr(self):
        return f"loss_weight={self.loss_weight}"
