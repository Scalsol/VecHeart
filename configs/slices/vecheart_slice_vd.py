# Slice-to-3D with random view dropout (robustness to missing 4CH/2CH/SAX views).
_base_ = ['./vecheart_slice.py']

model = dict(
    slice_cfgs=dict(
        slice_offsets=[
            [0, 1024, 1920, 7040],
            [0, 512, 1024, 3072],
            [0, 512, 512, 2560],
            [0, 448, 896],
            [0, 448, 448]
        ],
        view_dropout=True,
        dropout_ratio=[0.5, 0.5, 0.5],
        viu_train=['4CH', '2CH', 'SAX'],
        viu_test=['4CH', '2CH', 'SAX'],
    )
)
