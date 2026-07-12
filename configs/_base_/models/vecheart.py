# Base 3D auto-encoder model: VecHeart (VecSet-style per-part latent SDF decoder).
# `pts_layout` below is inlined as [sdf_size, sdf_size, surface_size] and MUST
# match the dataset sampling sizes (see _base_/datasets/vhmesh_sdf.py:
# sdf_size=1024, surface_size=8192).
num_classes = 5
dim = 512
num_latents = 256

model = dict(
    type='VecHeart',
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
        global_layers=[1, 3, 5],
        shifts=[13.59974129, 9.31580232, -1.01333806],
        scale=1.0 / 100
    )
)
