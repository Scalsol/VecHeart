import torch


def generate_grid(resolution):
    d = torch.linspace(-1, 1, resolution + 1)
    xs, ys, zs = torch.meshgrid((d, d, d), indexing="ij")
    grid = torch.stack((xs, ys, zs), -1)

    return grid
