import torch
from torch.utils.data import Dataset

from vecheart.core.evaluation import ChamferDistance, PointIoU, VolumeIoU
from .builder import DATASETS


@DATASETS.register_module()
class BaseDataset(Dataset):
    @staticmethod
    def evaluate(results, metrics="ChamDis", num_classes=1, **kwargs):
        if not isinstance(metrics, list):
            metrics = [metrics]
        eval_res = {}

        compute_std = kwargs.get("compute_std", False)

        _cham_dis = ChamferDistance(num_classes).to("cuda") if 'ChamDis' in metrics else None
        _point_iou = PointIoU(num_classes).to("cuda") if 'PointIoU' in metrics else None
        _volume_iou = VolumeIoU(num_classes).to("cuda") if 'VolumeIoU' in metrics else None

        for result in results:
            if _cham_dis is not None:
                _cham_dis.update(result["pred_meshes"], result["gt_meshes"])
            if _point_iou is not None:
                _point_iou.update(result["o"], result["label"])
            if _volume_iou is not None:
                _volume_iou.update(result['pred_volume'], result['gt_volume'])

        if _cham_dis is not None:
            cham_dis_scale = kwargs.get("cham_dis_scale", 1)  # is used to rescale to world unit (mm^2, 6400).
            cham_dis, nc = _cham_dis.compute()
            cham_dis = cham_dis * cham_dis_scale
            eval_res.update({
                "ChamDis": torch.mean(cham_dis).cpu().item(),
                "NC": torch.mean(nc).cpu().item()
            })

            try:
                for i in range(len(cham_dis)):
                    eval_res.update({f"CD{i + 1}": cham_dis[i].cpu().detach().item()})
                for i in range(len(nc)):
                    eval_res.update({f"NC{i + 1}": nc[i].cpu().detach().item()})

                if compute_std:
                    stats = _cham_dis.instance_stats
                    stats = [instance_stat[0] for instance_stat in stats]
                    stats = torch.stack(stats, dim=0) * cham_dis_scale
                    eval_res.update({f"CD_std": torch.std(stats, dim=0).cpu().numpy().tolist()})
            except:
                pass

        if _point_iou is not None:
            point_iou = _point_iou.compute()
            eval_res.update({"PointIoU": torch.mean(point_iou).cpu().item()})
            try:
                for i in range(len(point_iou)):
                    eval_res.update({f"IoU{i + 1}": point_iou[i].cpu().detach().item()})

                if compute_std:
                    stats = _point_iou.instance_stats
                    stats = torch.stack(stats, dim=0)
                    eval_res.update({f"PIoU_std": torch.std(stats, dim=0).cpu().numpy().tolist()})
            except:
                pass

        if _volume_iou is not None:
            volume_iou = _volume_iou.compute()
            eval_res.update({"VolumeIoU": torch.mean(volume_iou).cpu().item()})
            try:
                for i in range(len(volume_iou)):
                    eval_res.update({f"VIoU{i + 1}": volume_iou[i].cpu().detach().item()})

                if compute_std:
                    stats = _volume_iou.instance_stats
                    stats = torch.stack(stats, dim=0) * 100
                    eval_res.update({f"VIoU_std": torch.std(stats, dim=0).cpu().numpy().tolist()})
            except:
                pass

        return eval_res
