# Slice-to-3D with view dropout + motion corruption (per-slice 2D shifts at train).
# Only train_pipeline changes vs. vecheart_slice_vd.py (motion is a train-time aug);
# a pipeline is a list, so the config system replaces it wholesale rather than
# merging, hence it is redefined in full here.
_base_ = ['./vecheart_slice_vd.py']

sdf_size = 1024
surface_size = 8192
parts = [0, 1, 2, 3, 4]

train_pipeline = [
    dict(
        type='LoadSDFFromFile',
        load_all=False,
        parts=parts,
        mode='train',
        sdf_sampling=True,
        sdf_size=sdf_size,
        surface_sampling=True,
        surface_size=surface_size
    ),
    dict(
        type='LoadSlicePtsFromFile',
        sigma=6.0,
        # use_3d_motion=True,
        use_2d_motion=True
    ),
    dict(type='RandomTransform'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['points', 'sdf', 'surface', 'R', 'parts', 'slice_pts']),
]
data = dict(train=dict(pipeline=train_pipeline))
