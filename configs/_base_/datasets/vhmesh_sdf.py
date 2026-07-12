# VHMesh dataset for 3D auto-encoding (SDF only, no slice contours).
# sdf_size / surface_size are the per-part sampling budgets and must match the
# model's shape_cfgs.pts_layout (see _base_/models/vecheart.py).
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
    dict(type='RandomTransform'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['points', 'sdf', 'surface', 'R', 'parts']),
]
val_pipeline = [
    dict(
        type='LoadSDFFromFile',
        load_all=False,
        parts=parts,
        mode='val',
        sdf_sampling=False,
        sdf_size=sdf_size,
        surface_sampling=True,
        surface_size=surface_size
    ),
    dict(type='ToTensor', keys=['points', 'sdf', 'surface', 'parts']),
    dict(type='Collect', keys=['points', 'sdf', 'surface', 'parts'])
]
dataset_type = 'SDFDataset'
data_root = 'data/VHMesh/'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=False,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        pipeline=val_pipeline,
    ),
    train_dataloader=dict(),
    val_dataloader=dict(),
    test_dataloader=dict()
)
