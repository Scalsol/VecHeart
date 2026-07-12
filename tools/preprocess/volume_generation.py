import json
import multiprocessing as mp
import os
from multiprocessing import Pool
from pathlib import Path

import kaolin
import nibabel as nib
import numpy as np
import torch
import tqdm
import trimesh

from vecheart.utils import generate_grid


def to_tensor(something, dtype=torch.float, device="cuda"):
    return torch.tensor(something, dtype=dtype, device=device)


def load_geometries(geometry_paths):
    geometries = []
    for geometry_path in geometry_paths:
        geometry = trimesh.load(geometry_path)
        geometries.append(geometry)

    return geometries


def check_sign_grid(mesh, grid_pts):
    D, H, W = grid_pts.shape[:3]
    vertices, faces = mesh.vertices, mesh.faces

    grid_pts_flt = grid_pts.reshape(-1, 3)
    sign = kaolin.ops.mesh.check_sign(to_tensor(vertices).unsqueeze(0),
                                      to_tensor(faces, torch.int64),
                                      to_tensor(grid_pts_flt).unsqueeze(0),
                                      hash_resolution=512)

    return sign.reshape(D, H, W).cpu().numpy()


def process_single_instance(instance):
    resolution = 128

    print('processing', instance)
    mesh_files = sorted(list(instance.glob(f"*.obj")))
    meshes = load_geometries(mesh_files)

    shifts = np.array([13.59974129, 9.31580232, -1.01333806])
    for mesh in meshes:
        mesh.apply_translation(-shifts)
        mesh.apply_scale(1 / 100.0)

    grid_pts = generate_grid(resolution - 1).numpy()
    signs = [check_sign_grid(mesh, grid_pts) for mesh in meshes]

    image_arr = np.zeros((resolution, resolution, resolution))
    for part in range(len(signs)):
        image_arr[signs[part] == 1] = part + 1

    image_arr = image_arr.astype(np.int32)

    nii = nib.Nifti1Image(image_arr, affine=np.eye(4))
    nib.save(nii, str(mesh_files[0]).replace("_1.obj", "_volume.nii.gz"))

    np.savez_compressed(
        str(mesh_files[0]).replace("_1.obj", "_volume.npz"),
        volume=image_arr
    )


if __name__ == "__main__":
    vhmesh_path = Path("data/VHMesh")

    all_instances = sorted(list(vhmesh_path.glob("*")))
    all_instances = [instance for instance in all_instances if os.path.isdir(instance)]

    # when using CUDA with multiprocessing, remember to use SPAWN
    with mp.get_context("spawn").Pool(16) as p:
        res = p.map(process_single_instance, all_instances)