import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree  # noqa
from sklearn.decomposition import PCA


def load_geometries(geometry_paths):
    geometries = []
    for geometry_path in geometry_paths:
        geometry = trimesh.load(geometry_path)
        geometries.append(geometry)

    return geometries


def centroid(pts):
    return np.mean(pts, axis=0)


def lv_long_axis(lv_points):
    pca = PCA(n_components=3)
    pca.fit(lv_points)
    # principal direction
    axis = pca.components_[0]  # unit vector
    # sign of the direction: we want axis to point base -> apex or apex -> base, which can be decided from z or the point projection
    return axis / np.linalg.norm(axis)


def find_apex(lv_points, axis):
    # pick a reference origin, e.g., lv centroid
    c = centroid(lv_points)
    proj = (lv_points - c) @ axis
    # apex is the point with minimal (or maximal) proj depending orientation
    idx = np.argmin(proj)  # choose minimal; if wrong, switch to argmax
    return lv_points[idx]


def valve_center_between(A_points, B_points, dist_thresh=5.0):
    # Build kdtree B
    tree = cKDTree(B_points)
    dists, idxs = tree.query(A_points, k=1)
    close_mask = dists < dist_thresh  # mm units; adjust to your coordinate system
    if np.sum(close_mask) < 10:
        # if the threshold is too small and yields too few points, take the nearest few-percent of points
        k = max(10, int(0.01 * len(A_points)))
        idx_sorted = np.argsort(dists)[:k]
        pts = A_points[idx_sorted]
    else:
        pts = A_points[close_mask]
    return centroid(pts)


def plane_from_3pts(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)  # noqa
    n = n / np.linalg.norm(n)
    return n, p1


def generate_sax_planes(axis, lv_centroid, apex, mitral, margin_mm=2.0):
    # axis: unit LAX direction pointing roughly base<- ->apex (either direction)
    # we cut n_slices at equal spacing along the LAX from the base (base_near_mitral) to the apex
    # base point: choose point near mitral (mitral) projected onto axis
    # apex is known. compute scalar coordinates along axis (with origin at lv_centroid)
    origin = lv_centroid
    proj_apex = np.dot(apex - origin, axis)
    proj_mitral = np.dot(mitral - origin, axis)
    # define basal and apical positions: ensure base > apex in projection (adjust accordingly)
    z_base = proj_mitral + margin_mm  # extend a bit beyond the base
    z_apex = proj_apex - margin_mm

    if z_base < z_apex:
        z_base, z_apex = z_apex, z_base

    z = z_apex
    planes = []
    while z < z_base:
        point_on_plane = origin + axis * z
        normal = axis  # SAX plane normal == LAX direction
        planes.append((normal, point_on_plane))

        z += 10.0

    return planes


def do_section(p, n, all_meshes, all_names=['myo', 'lv', 'rv', 'la', 'ra'], skipped=[], slice_name='sax'):
    slice_pts_all_classes = dict()
    all_empty = True
    for mesh_name, mesh in zip(all_names, all_meshes):
        if mesh_name in skipped:
            slice_pts_all_classes[mesh_name] = []
            continue

        sec_3d = mesh.section(plane_origin=p, plane_normal=n)
        if sec_3d is not None:
            slice_pts_all_classes[mesh_name] = sec_3d.vertices
            all_empty = False
        else:
            slice_pts_all_classes[mesh_name] = []
            if 'SAX' not in slice_name:
                print(f'No intersection for {slice_name} {mesh_name}!')

    return slice_pts_all_classes, all_empty


def process_single_instance(instance):
    print('processing', instance)
    mesh_files = sorted(list(instance.glob(f"*.obj")))
    meshes = load_geometries(mesh_files)

    myo_pts, lv_pts, rv_pts, la_pts, ra_pts = [mesh.vertices for mesh in meshes]

    lv_c = centroid(lv_pts)
    la_c = centroid(la_pts)
    ra_c = centroid(ra_pts)
    rv_c = centroid(rv_pts)

    axis = lv_long_axis(lv_pts)
    apex = find_apex(lv_pts, axis)
    mitral = valve_center_between(la_pts, lv_pts, dist_thresh=4.0)
    tricuspid = valve_center_between(ra_pts, rv_pts, dist_thresh=4.0)

    proj = (lv_pts - lv_c) @ axis
    # take top K basal points
    K = max(30, int(0.01*len(lv_pts)))
    basal_idx = np.argsort(proj)[-K:]
    aortic_approx = centroid(lv_pts[basal_idx])

    # 4CH plane
    n4ch, p4ch = plane_from_3pts(apex, mitral, tricuspid)

    # 3CH plane: apex, mitral, aortic_approx
    n3ch, p3ch = plane_from_3pts(apex, mitral, aortic_approx)

    # 2CH plane
    v = mitral - apex
    n2ch = np.cross(v, n4ch)  # noqa
    n2ch = n2ch / np.linalg.norm(n2ch)
    p2ch = apex

    # SAX planes
    sax_planes = generate_sax_planes(axis, lv_c, apex, mitral)

    # 4CH pts
    slice_pts_4ch, _ = do_section(p4ch, n4ch, meshes, skipped=[], slice_name='4CH')

    # 3CH pts
    slice_pts_3ch, _ = do_section(p3ch, n3ch, meshes, skipped=['ra'], slice_name='3CH')

    # 2CH pts
    slice_pts_2ch, _ = do_section(p2ch, n2ch, meshes, skipped=['rv', 'ra'], slice_name='2CH')

    # SAX pts
    slice_pts_sax = []
    for plane in sax_planes:
        slice_pts_sax_slice, all_empty = do_section(plane[1], plane[0], meshes, skipped=['la', 'ra'], slice_name='SAX')
        if not all_empty:
            slice_pts_sax.append(slice_pts_sax_slice)
        else:
            print('No intersection.')

    slice_pts = {
        '4CH': slice_pts_4ch,
        '3CH': slice_pts_3ch,
        '2CH': slice_pts_2ch,
        'SAX': slice_pts_sax,
    }

    save_name = str(mesh_files[0]).replace("_1.obj", "_slice_pts.npz")
    np.savez(save_name, slice_pts=slice_pts, axis=axis, lv_center=lv_c)

    vis_folder = mesh_files[0].parent / 'slice_pts'
    os.makedirs(vis_folder, exist_ok=True)
    for slice_name in ['4CH', '3CH', '2CH']:
        pts_all = []
        pts_by_class = slice_pts[slice_name]
        for part in ['myo', 'lv', 'rv', 'la', 'ra']:
            if len(pts_by_class[part]) > 0:
                pts_all.append(pts_by_class[part])
        pts_all = np.concatenate(pts_all)

        trimesh.PointCloud(pts_all).export(str(vis_folder / (slice_name + '.obj')))

    pts_all = []
    pts_by_class = slice_pts['SAX']
    for pts_slice in pts_by_class:
        for part in ['myo', 'lv', 'rv', 'la', 'ra']:
            if len(pts_slice[part]) > 0:
                pts_all.append(pts_slice[part])
    pts_all = np.concatenate(pts_all)

    trimesh.PointCloud(pts_all).export(str(vis_folder / 'SAX.obj'))

    # get some statistic
    output = dict()
    for slice_name in ['4CH', '2CH']:
        pts_by_class = slice_pts[slice_name]
        for part in ['myo', 'lv', 'rv', 'la', 'ra']:
            if len(pts_by_class[part]) > 0:
                output[f'{part}_{slice_name}'] = len(pts_by_class[part])
            else:
                output[f'{part}_{slice_name}'] = 0

    for slice_name in ['SAX']:
        pts_by_class = slice_pts[slice_name]
        for part in ['myo', 'lv', 'rv']:
            output[f'{part}_{slice_name}'] = 0
            for pts_slice in pts_by_class:
                if len(pts_slice[part]) > 0:
                    output[f'{part}_{slice_name}'] += len(pts_slice[part])

    return output


if __name__ == '__main__':
    vhmesh_path = Path("data/VHMesh")

    all_instances = sorted(list(vhmesh_path.glob("*")))
    all_instances = [instance for instance in all_instances if os.path.isdir(instance)]

    with Pool(16) as p:
        res = p.map_async(process_single_instance, all_instances)
        outputs = res.get()
