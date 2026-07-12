import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np
import pytorch3d.structures
import reimu
import torch
import torch.distributed as dist
from reimu.runner import get_dist_info


def get_sample_name(data):
    sample_info = data['sample_metas']._data[0][0]['sample_info']
    sample_name = os.path.basename(sample_info)
    sample_name = sample_name.replace(".nii.gz", "")
    sample_name = sample_name.replace("_sdf", "")
    sample_name = sample_name.replace(".npz", "")
    sample_name = sample_name.replace("_x.npy", "")

    return sample_name


def single_gpu_test(model,
                    data_loader,
                    return_label=False,
                    show=False,
                    show_dir=None,
                    **kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = reimu.ProgressBar(len(data_loader))
    if show_dir:
        os.makedirs(show_dir, exist_ok=True)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data["resolution"] = kwargs.get("resolution", None)
            if kwargs.get("return_volume", False):
                data["return_volume"] = True
            result = model(data, return_loss=False, return_label=return_label)

        batch_size = 1

        results.append(result)

        for _ in range(batch_size):
            prog_bar.update()

        if show and show_dir:
            sample_name = get_sample_name(data)

            # vecheart
            if "pred_meshes" in result:
                pred_meshes, gt_meshes = result['pred_meshes'], result.get('gt_meshes', [])
                for c, pred_mesh in enumerate(pred_meshes):
                    # pred_mesh.apply_scale(100)
                    # pred_mesh.apply_translation([13.59974129, 9.31580232, -1.01333806])
                    pred_mesh.export(os.path.join(show_dir, f"{sample_name}_{c + 1}_pred.obj"))
                for c, gt_mesh in enumerate(gt_meshes):
                    # gt_mesh.apply_scale(100)
                    # gt_mesh.apply_translation([13.59974129, 9.31580232, -1.01333806])
                    gt_mesh.export(os.path.join(show_dir, f"{sample_name}_{c + 1}_gt.obj"))

            if "pred_volume" in result:
                pred_volume, gt_volume = result['pred_volume'], result['gt_volume']

                pred_nii = nib.Nifti1Image(pred_volume.astype(np.int32), affine=np.eye(4))
                nib.save(pred_nii, os.path.join(show_dir, f"{sample_name}_volume_pred.nii.gz"))

                gt_nii = nib.Nifti1Image(gt_volume.astype(np.int32), affine=np.eye(4))
                nib.save(gt_nii, os.path.join(show_dir, f"{sample_name}_volume_gt.nii.gz"))

            if "slice_pcs" in result:
                slice_pcs = result['slice_pcs']
                for c, slice_pc in enumerate(slice_pcs):
                    slice_pc.export(os.path.join(show_dir, f"{sample_name}_{c + 1}_slice_pts.obj"))

    print()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, **kwargs):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = reimu.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data["resolution"] = kwargs.get("resolution", None)
            result = model(data, return_loss=False, return_label=True)

        results.append(result)

        if rank == 0:
            batch_size = len(result["preds"]) if "preds" in result else 1
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def move_to_device(obj, device=torch.device('cuda:0')):
    # is this a good workaround? I do not know xd.
    # but as the evaluation happens rarely, some overhead is acceptable.
    if isinstance(obj, (torch.Tensor, pytorch3d.structures.Meshes)):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            reimu.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        reimu.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    reimu.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(reimu.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        ordered_results = move_to_device(ordered_results)
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        ordered_results = move_to_device(ordered_results)
        return ordered_results
