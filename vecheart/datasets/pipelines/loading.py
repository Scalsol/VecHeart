import numpy as np
import torch

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadSDFFromFile(object):
    def __init__(self,
                 load_all=True,
                 parts=None,
                 mode='training',
                 sdf_sampling=True,
                 sdf_size=4096,
                 surface_sampling=True,
                 surface_size=8192):
        self.load_all = load_all
        self.parts = parts

        self.mode = mode

        self.sdf_sampling = sdf_sampling
        self.sdf_size = sdf_size
        self.surface_sampling = surface_sampling
        self.surface_size = surface_size

    def process_single_part(self, data):
        vol_points = data['vol_points']
        vol_sdf = data['vol_sdf']
        near_points = data['near_points']
        near_sdf = data['near_sdf']
        surface = data['surface_points']

        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.surface_size, replace=False)
            surface = surface[ind]
        surface = torch.from_numpy(surface)

        if self.sdf_sampling:
            ### make sure balanced sampling, maybe not necessary when doing sdf regression
            pos_vol_id = vol_sdf < 0

            if pos_vol_id.sum() > self.sdf_size // 2:
                ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size // 2, replace=False)
                pos_vol_points = vol_points[pos_vol_id][ind]
                pos_vol_sdf = vol_sdf[pos_vol_id][ind]
            else:
                pos_vol_id = near_sdf < 0
                if pos_vol_id.sum() > self.sdf_size // 2:
                    ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size // 2, replace=False)
                    pos_vol_points = near_points[pos_vol_id][ind]
                    pos_vol_sdf = near_sdf[pos_vol_id][ind]
                else:
                    ind = np.random.default_rng().choice(vol_points.shape[0], self.sdf_size // 2, replace=False)
                    pos_vol_points = vol_points[ind]
                    pos_vol_sdf = vol_sdf[ind]

            neg_vol_id = vol_sdf >= 0

            ind = np.random.default_rng().choice(neg_vol_id.sum(), self.sdf_size // 2, replace=False)
            neg_vol_points = vol_points[neg_vol_id][ind]
            neg_vol_sdf = vol_sdf[neg_vol_id][ind]

            vol_points = np.concatenate([pos_vol_points, neg_vol_points], axis=0)
            vol_sdf = np.concatenate([pos_vol_sdf, neg_vol_sdf], axis=0)

            ind = np.random.default_rng().choice(near_points.shape[0], self.sdf_size, replace=False)
            near_points = near_points[ind]
            near_sdf = near_sdf[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_sdf = torch.from_numpy(vol_sdf).float()

        if self.mode == 'train':
            near_points = torch.from_numpy(near_points)
            near_sdf = torch.from_numpy(near_sdf).float()

            points = torch.cat([vol_points, near_points], dim=0)
            sdf = torch.cat([vol_sdf, near_sdf], dim=0)
        else:
            near_points = torch.from_numpy(near_points)
            near_sdf = torch.from_numpy(near_sdf).float()

            points = near_points
            sdf = near_sdf

        return points, sdf, surface

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['sample_info']
        with np.load(filename, allow_pickle=True) as data:
            num_parts = data['num_parts']
            geo_datas = data['geo_datas']

        if self.load_all:
            parts = np.arange(num_parts)
        else:
            parts = np.array(self.parts)

        points_list, sdf_list, surface_list = [], [], []
        for part in parts:
            points, sdf, surface = self.process_single_part(geo_datas[part])
            points_list.append(points)
            sdf_list.append(sdf)
            surface_list.append(surface)

        points = torch.stack(points_list, dim=0)
        sdf = torch.stack(sdf_list, dim=0)
        surface = torch.stack(surface_list, dim=0)

        results['parts'] = parts
        results['points'] = points
        results['sdf'] = sdf
        results['surface'] = surface

        return results


@PIPELINES.register_module()
class LoadSlicePtsFromFile(object):
    def __init__(
        self,
        parts=[0, 1, 2, 3, 4],
        sigma=4.0,
        use_3d_motion=False,
        use_2d_motion=False,
        only_central=False,
        pts_num=dict(
            myo={'4CH': 1024, '2CH': 896, 'SAX': 5120},
            lv={'4CH': 512, '2CH': 512, 'SAX': 2048},
            rv={'4CH': 512, 'SAX': 2048},
            la={'4CH': 448, '2CH': 448},
            ra={'4CH': 448}
        )
    ):
        self.parts_name = ['myo', 'lv', 'rv', 'la', 'ra']
        self.parts = parts
        self.parts_in_use = [self.parts_name[part] for part in parts]

        self.pts_num = pts_num

        self.sigma = sigma
        self.use_3d_motion = use_3d_motion
        self.use_2d_motion = use_2d_motion

    def sample_pts(self, points, size):
        if size <= points.shape[0]:
            ind = np.random.default_rng().choice(points.shape[0], size, replace=False)
            sampled_points = points[ind]
        else:
            sampled_points = points
            while sampled_points.shape[0] < size:
                ind = np.random.default_rng().choice(
                    points.shape[0],
                    min(points.shape[0], size - sampled_points.shape[0]),
                    replace=False
                )
                sampled_points = np.concatenate([sampled_points, points[ind]], axis=0)

        return sampled_points

    def generate_motion(self, axis_z, dims=['x', 'y', 'z']):
        sigma = self.sigma * np.random.rand(1)

        shift_x, shift_y, shift_z = sigma * (np.random.rand(3) - 0.5) * 2

        axis_x = np.array([0, 1, 0])
        axis_x = axis_x - np.dot(axis_x, axis_z) * axis_z
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = np.cross(axis_z, axis_x)  # noqa
        axis_y = axis_y / np.linalg.norm(axis_y)

        shift = np.array([0.0, 0.0, 0.0])
        if 'x' in dims:
            shift = shift + shift_x * axis_x
        if 'y' in dims:
            shift = shift + shift_y * axis_y
        if 'z' in dims:
            shift = shift + shift_z * axis_z

        return shift

    def __call__(self, results):
        filename = results['slice_info']
        with np.load(filename, allow_pickle=True) as data:
            slice_pts = data['slice_pts'].item()
            axis = data['axis']

        slice_pts_4ch = slice_pts.get('4CH', None)
        slice_pts_2ch = slice_pts.get('2CH', None)
        slice_pts_sax = slice_pts.get('SAX', None)

        # motion for 2CH
        if self.use_3d_motion:
            motion_2ch = self.generate_motion(axis)
        else:
            motion_2ch = np.array([0.0, 0.0, 0.0])

        # motion for SAXs
        if self.use_3d_motion:
            motion_sax_global = self.generate_motion(axis, dims=['z'])
        else:
            motion_sax_global = np.array([0.0, 0.0, 0.0])
        motion_saxs = []
        if self.use_2d_motion:
            for i_slice in range(len(slice_pts_sax)):
                motion_sax = self.generate_motion(axis, dims=['x', 'y'])
                motion_saxs.append(motion_sax + motion_sax_global)

        slice_pts = []
        for part in self.parts_in_use:
            part_pts = []
            if '4CH' in self.pts_num[part] and slice_pts_4ch is not None:
                sampled_pts = self.sample_pts(slice_pts_4ch[part], self.pts_num[part]['4CH'])
                part_pts.append(sampled_pts)
            if '2CH' in self.pts_num[part] and slice_pts_2ch is not None:
                sampled_pts = self.sample_pts(slice_pts_2ch[part], self.pts_num[part]['2CH'])
                if self.use_3d_motion:
                    sampled_pts = sampled_pts + motion_2ch
                part_pts.append(sampled_pts)
            if 'SAX' in self.pts_num[part] and slice_pts_sax is not None:
                sax_pts = []
                for i_slice, slice_pts_sax_single in enumerate(slice_pts_sax):
                    pts = slice_pts_sax_single[part]
                    if len(pts) > 0:
                        if self.use_2d_motion:
                            pts = pts + motion_saxs[i_slice]
                        sax_pts.append(pts)

                sax_pts = np.concatenate(sax_pts, axis=0)
                part_pts.append(self.sample_pts(sax_pts, self.pts_num[part]['SAX']))

            part_pts = np.concatenate(part_pts, axis=0)
            slice_pts.append(part_pts)

        slice_pts = np.concatenate(slice_pts, axis=0)

        # post transformation
        shifts = np.array([13.59974129, 9.31580232, -1.01333806])
        slice_pts = (slice_pts - shifts) * (1 / 100)

        results['slice_pts'] = torch.from_numpy(slice_pts).float()

        return results
