import torch
from torchmetrics import Metric


def to_tensor(something, dtype=torch.float, device="cuda"):
    return torch.tensor(something, dtype=dtype, device=device)


class PointIoU(Metric):
    full_state_update = False

    def __init__(self, num_classes):
        super(PointIoU, self).__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.num_classes = num_classes
        self.add_state("steps", default=torch.zeros(1))
        self.add_state("iou", default=torch.zeros((num_classes,)))

        self.instance_stats = []

    def update(self, pred, label):
        self.steps += 1

        stats = self.compute_stats(pred, label)
        self.iou += stats

        self.instance_stats.append(stats)

    def compute(self):
        return self.iou / self.steps

    def compute_stats(self, pred, label):
        if pred.ndim == 2:
            pred = pred.unsqueeze(1)

        _label = torch.zeros_like(label)
        _label[label < 0] = 1

        _pred = torch.zeros_like(pred)
        _pred[pred < 0] = 1

        # accuracy = (_pred == _label).float().sum(dim=1) / _label.shape[1]
        # accuracy = accuracy.mean()
        intersection = (_pred * _label).sum(dim=2)
        union = (_pred + _label).gt(0).sum(dim=2) + 1e-5

        iou = intersection * 1.0 / union
        iou = iou.squeeze(0)

        return iou
