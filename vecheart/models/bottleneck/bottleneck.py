# Copyright (c) 2025, Biao Zhang.

import torch
import torch.nn as nn

from ..builder import COMPONENTS
from ..layers import DiagonalGaussianDistribution


@COMPONENTS.register_module()
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def pre(self, x):
        return {'x': x}

    def post(self, x):
        return x


@COMPONENTS.register_module()
class KLBottleneck(nn.Module):
    def __init__(self, dim, latent_dim, kl_weight):
        super().__init__()

        self.kl_weight = kl_weight

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def pre(self, x):
        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        if self.training:
            x = posterior.sample()
        else:
            x = posterior.mode()
        kl = posterior.kl()

        return {'x': x, 'kl': self.kl_weight * kl}

    def post(self, x):
        x = self.proj(x)
        return x


## LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion
## https://openreview.net/forum?id=72OSO38a2z
@COMPONENTS.register_module()
class NormalizedBottleneck(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()

        self.post_bottleneck_proj = nn.Linear(latent_dim, dim)

        self.pre_bottleneck_proj = nn.Linear(dim, latent_dim)
        self.pre_bottleneck_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

        self.gamma = nn.Parameter(torch.ones(latent_dim))
        self.beta = nn.Parameter(torch.zeros(latent_dim))

    def pre(self, x):
        x = self.pre_bottleneck_norm(self.pre_bottleneck_proj(x))
        return {'x': x}

    def post(self, x):
        x = x * self.gamma + self.beta

        x = self.post_bottleneck_proj(x)

        return x
