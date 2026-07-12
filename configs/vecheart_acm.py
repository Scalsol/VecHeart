# ACM auto-encoder = base VecHeart + part-masking (MAE-style pretraining).
# This is the checkpoint the slice configs pretrain from.
_base_ = ['./vecheart.py']

model = dict(
    mask_cfgs=dict(
        mask_parts=2,
        random_mask=True,
        test_mask_parts=0,
    )
)
