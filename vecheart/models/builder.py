# Copyright (c) OpenMMLab. All rights reserved.
from reimu.models import MODELS as REIMU_MODELS
from reimu.utils import Registry

MODELS = Registry('models', parent=REIMU_MODELS)

RECONSTRUCTORS = MODELS
COMPONENTS = MODELS
LOSSES = MODELS


def build_model(cfg):
    return MODELS.build(cfg)


def build_reconstructor(cfg):
    return RECONSTRUCTORS.build(cfg)


def build_component(cfg):
    if cfg is None:
        return None
    return COMPONENTS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    if cfg is None:
        return None
    return LOSSES.build(cfg)
