import json
import os
from multiprocessing import Pool

import numpy as np
import point_cloud_utils as pcu
import trimesh

N_vol = 50000
N_near = 25000

from pathlib import Path


def process(model_filename):
    mesh = trimesh.load(model_filename, skip_materials=True, process=True, force='mesh')
    print('loading successfully', model_filename)

    v, f = mesh.vertices, mesh.faces
    resolution = 50_000

    if not mesh.is_watertight:
        print(model_filename)

    vw, fw = v, f
    bbox_max = np.array([109.37135966, 77.16720654, 93.35775865])
    bbox_min = np.array([-82.17187708, -58.5356019, -95.38443476])

    shifts = (bbox_max + bbox_min) / 2
    vw = vw - shifts
    scale = 1 / 100
    vw *= scale

    fid, bc = pcu.sample_mesh_random(vw, fw, N_near)
    surface_points = pcu.interpolate_barycentric_coords(fw, fid, bc, vw)

    vol_points = np.random.rand(N_vol, 3) * 2 - 1

    vol_sdf, _, _ = pcu.signed_distance_to_mesh(vol_points, vw, fw)

    near_points = [
        surface_points + np.random.normal(scale=0.005, size=(N_near, 3)),
        surface_points + np.random.normal(scale=0.05, size=(N_near, 3)),
    ]
    near_points = np.concatenate(near_points)
    near_sdf, _, _ = pcu.signed_distance_to_mesh(near_points, vw, fw)

    geo_data = dict(
        shifts=shifts,
        scale=scale,
        vol_points=vol_points.astype(np.float32),
        vol_sdf=vol_sdf.astype(np.float32),
        near_points=near_points.astype(np.float32),
        near_sdf=near_sdf.astype(np.float32),
        surface_points=surface_points.astype(np.float32),
    )

    return geo_data


def process_single_instance(instance):
    print('processing', instance)
    mesh_files = sorted(list(instance.glob(f"*.obj")))
    geo_datas = [process(mesh_file) for mesh_file in mesh_files]

    save_name = str(mesh_files[0]).replace("_1.obj", "_sdf.npz")
    np.savez(
        save_name,
        num_parts=len(geo_datas),
        geo_datas=geo_datas
    )


if __name__ == '__main__':
    vhmesh_path = Path("data/VHMesh")

    all_instances = sorted(list(vhmesh_path.glob("*")))
    all_instances = [instance for instance in all_instances if os.path.isdir(instance)]

    print(len(all_instances))

    with Pool(16) as p:
        res = p.map_async(process_single_instance, all_instances)
        _ = res.get()
