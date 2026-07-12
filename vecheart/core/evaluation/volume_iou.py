import numpy as np
import torch
from torchmetrics import Metric


class VolumeIoU(Metric):
    full_state_update = False

    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.num_classes = num_classes
        self.add_state("steps", default=torch.zeros(1))
        self.add_state("viou", default=torch.zeros((num_classes,)))

        self.instance_stats = []

    def update(self, pred, gt):
        self.steps += 1

        stats = self.compute_stats(pred, gt)
        self.viou += stats

        self.instance_stats.append(stats)

    def compute(self):
        return 100 * self.viou / self.steps

    def compute_stats(self, pred, gt):
        scores = torch.zeros(self.num_classes, device='cuda', dtype=torch.float32)

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).cuda()
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).cuda()

        for i in range(1, self.num_classes + 1):
            if (gt != i).all():
                # no foreground class
                scores[i - 1] += 1 if (pred != i).all() else 0
                continue
            tp, fn, fp = self.get_stats(pred, gt, i)
            denom = (tp + fp + fn).to(torch.float)
            score_cls = tp.to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores

    @staticmethod
    def get_stats(pred, gt, c):
        tp = torch.logical_and(pred == c, gt == c).sum()
        fn = torch.logical_and(pred != c, gt == c).sum()
        fp = torch.logical_and(pred == c, gt != c).sum()
        return tp, fn, fp
