# Slice-to-3D, clean full-view baseline (no view dropout, no motion corruption).
# Base slice config the other slice variants inherit from.
_base_ = [
    '../_base_/models/vecheart_slice.py',
    '../_base_/datasets/vhmesh_slice.py',
    '../_base_/default_schedule.py',
    '../_base_/default_runtime.py',
]
