# Base slice-to-3D model: VecHeartSlice. Reuses the frozen pretrained 3D decoder
# (from the ACM auto-encoder) and trains only the parallel slice encoder.
# `pts_layout` MUST match the dataset sampling sizes (see
# _base_/datasets/vhmesh_slice.py: sdf_size=1024, surface_size=8192).
num_classes = 5
dim = 512
num_latents = 256

model = dict(
    type='VecHeartSlice',
    pretrained='work_dirs/vecheart_acm/latest.pth',
    bottleneck=dict(
        type='NormalizedBottleneck',
        dim=dim,
        latent_dim=32
    ),
    loss=dict(
        vol=dict(
            type='L1Loss',
            loss_weight=1.0
        ),
        near=dict(
            type='L1Loss',
            loss_weight=10.0
        ),
        surface=dict(
            type='L1Loss',
            loss_weight=1.0
        ),
        eikonal=dict(
            type='EikonalLoss',
            loss_weight=0.001
        ),
        recon=dict(
            type='L2Loss',
            loss_weight=0.0
        )
    ),
    encoder_depth=0,
    decoder_depth=7,
    dim=dim,
    output_dim=1,
    num_latents=num_latents,
    dim_head=64,
    num_classes=num_classes,
    parts=[0, 1, 2, 3, 4],
    shape_cfgs=dict(
        pts_layout=[1024, 1024, 8192],
        global_layers=[1, 3, 5]
    ),
    slice_cfgs=dict(
        slice_seps=[0, 7040, 10112, 12672, 13568, 14016],
        use_pretrained=True
    )
)
