# Copyright (c) 2025, Biao Zhang.
import math
from collections import OrderedDict

import mcubes
import numpy as np
import torch
import torch.nn as nn
import trimesh
from einops import rearrange, repeat
from reimu.runner import auto_fp16

from vecheart.utils import generate_grid
from ..builder import RECONSTRUCTORS, build_component, build_loss
from ..layers import Attention, FeedForward, PointEmbed, PreNorm


@RECONSTRUCTORS.register_module()
class VecHeart(nn.Module):
    def __init__(
            self,
            bottleneck,
            loss,
            encoder_depth=0,
            decoder_depth=6,
            dim=512,
            output_dim=1,
            num_latents=512,
            dim_head=64,
            num_classes=5,
            parts=[0, 1, 2, 3, 4],
            shape_cfgs=None,
            mask_cfgs=None,
            pretrained=None
    ):
        super().__init__()

        queries_dim = dim

        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_latents = num_latents
        self.num_classes = num_classes
        self.parts = parts

        self.class_latents = nn.ModuleList([nn.Embedding(num_latents, dim) for _ in range(self.num_classes)])

        self.point_embed = PointEmbed(dim=dim)

        # encoder
        self.encoder_cross_attn = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=dim // dim_head, dim_head=dim_head)),
            PreNorm(dim, FeedForward(dim))
        ])

        self.encoder_layers = nn.ModuleList([])
        for i in range(encoder_depth):
            self.encoder_layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=dim // dim_head, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim))
            ]))

        # bottleneck
        self.bottleneck = build_component(bottleneck)

        # decoder
        self.decoder_layers = nn.ModuleList([])

        for i in range(decoder_depth):
            self.decoder_layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=dim // dim_head, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim))
            ]))

        # query
        self.query_cross_attn = PreNorm(queries_dim,
                                        Attention(queries_dim, dim, heads=dim // dim_head, dim_head=dim_head))

        self.to_outputs = nn.Sequential(
            nn.LayerNorm(queries_dim),
            nn.Linear(queries_dim, output_dim)
        )

        nn.init.zeros_(self.to_outputs[1].weight)
        nn.init.zeros_(self.to_outputs[1].bias)

        # build loss
        self.loss_vol = build_loss(loss.vol)
        self.loss_near = build_loss(loss.near)
        self.loss_surface = build_loss(loss.surface)
        self.loss_eikonal = build_loss(loss.eikonal)

        self.shape_cfgs = shape_cfgs

        # mask related (None -> no masking, behaves as a plain auto-encoder)
        self.mask_cfgs = mask_cfgs

        self._calc_layout()

        # for training sample visualization
        self.latest_data = None

        # this is very important for reimu fp16 training to enable!!!
        self.fp16_enabled = False

        if pretrained:
            self.load_pretrained(pretrained)

    def _calc_layout(self):
        self.layouts = {
            'vol': {'size': self.shape_cfgs['pts_layout'][0]},
            'near': {'size': self.shape_cfgs['pts_layout'][1]},
            'surface': {'size': self.shape_cfgs['pts_layout'][2]}
        }
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start

    def get_layout(self, feats, name):
        if name not in self.layouts:
            return None

        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]]

    def encode(self, pc, all_masks=None, parts=None):
        B, C, N, _ = pc.shape

        all_latents = torch.stack([latents.weight for latents in self.class_latents], dim=0)
        if parts is not None:
            all_latents = all_latents[parts]
        x = repeat(all_latents, 'c n d -> b c n d', b=B)

        # drop masked parts before encoding; with all_masks=None this is the plain flatten
        if all_masks is None or C < self.num_classes:
            x = rearrange(x, 'b c n d -> (b c) n d')
            pc = rearrange(pc, 'b c n d -> (b c) n d')
        else:
            x = x[~all_masks]
            pc = pc[~all_masks]

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.encoder_cross_attn

        x = cross_attn(x, context=pc_embeddings, mask=None) + x
        x = cross_ff(x) + x

        for i_layer, (self_attn, self_ff) in enumerate(self.encoder_layers):
            x = self_attn(x) + x
            x = self_ff(x) + x

        bottleneck = self.bottleneck.pre(x)
        return bottleneck

    def part_to_whole(self, x_part, all_masks=None):
        _, N, D = x_part.shape
        B = all_masks.shape[0]

        # visible
        x_full = torch.zeros((B, self.num_classes, N, D)).to(x_part)
        x_full[~all_masks] = x_part

        # masked: fall back to the learnable per-class latents
        all_latents = torch.stack([latents.weight for latents in self.class_latents], dim=0)
        all_latents = repeat(all_latents, 'c n d -> b c n d', b=B)
        x_full[all_masks] = all_latents[all_masks]

        return x_full

    def decode(self, x, all_masks=None):
        x = self.bottleneck.post(x)

        if all_masks is not None:
            # restore masked parts, so the global layers see the full num_classes set
            x = self.part_to_whole(x, all_masks)
            x = rearrange(x, 'b c n d -> (b c) n d')
            c = self.num_classes
        else:
            c = len(self.parts)

        global_layers = self.shape_cfgs.get('global_layers', [1, 3, 5])
        for i_layer, (self_attn, self_ff) in enumerate(self.decoder_layers):
            if i_layer in global_layers:
                x = rearrange(x, '(b c) n d -> b (c n) d', c=c)

            x = self_attn(x) + x
            x = self_ff(x) + x

            if i_layer in global_layers:
                x = rearrange(x, 'b (c n) d -> (b c) n d', c=c)

        return x

    def query(self, x, queries):
        queries_embeddings = self.point_embed(queries)
        latents = self.query_cross_attn(queries_embeddings, context=x)

        return self.to_outputs(latents)

    def _generate_mask(self, pc, training=True):
        B, C, N, _ = pc.shape
        all_masks = torch.zeros((B, self.num_classes)).to(pc)

        if training:
            mask_parts = self.mask_cfgs.get('mask_parts', 0)
            random_mask = self.mask_cfgs.get('random_mask', False)
        else:
            # test: fixed count, may differ from training (falls back to mask_parts if unset)
            mask_parts = self.mask_cfgs.get('test_mask_parts', self.mask_cfgs.get('mask_parts', 0))
            random_mask = False

        for b in range(B):
            n_mask = np.random.randint(0, mask_parts + 1) if random_mask else mask_parts
            if n_mask <= 0:
                continue
            mask = np.concatenate([np.zeros(self.num_classes - n_mask), np.ones(n_mask)])
            if training:
                np.random.shuffle(mask)
            all_masks[b, :] = torch.from_numpy(mask)

        all_masks = all_masks.to(torch.bool)

        return all_masks

    def forward_train(self, pc, queries, sdf, block_size=100000):
        B, C, N, _ = pc.shape

        losses = {}

        # mask_cfgs=None -> all_masks=None -> plain auto-encoding
        all_masks = self._generate_mask(pc) if self.mask_cfgs is not None else None

        bottleneck = self.encode(pc, all_masks)
        x = self.decode(bottleneck['x'], all_masks)
        x = rearrange(x, '(b c) n d -> b c n d', b=B)

        for c in range(self.num_classes):
            queries_c, sdf_c = queries[:, c], sdf[:, c]
            x_c = x[:, c]

            if queries_c.shape[1] > block_size:
                N = block_size
                os = []
                for block_idx in range(math.ceil(queries_c.shape[1] / N)):
                    o = self.query(x_c, queries_c[:, block_idx * N:(block_idx + 1) * N, :]).squeeze(-1)
                    os.append(o)
                o = torch.cat(os, dim=1)
            else:
                o = self.query(x_c, queries_c).squeeze(-1)

            losses.update(
                {f"loss_vol_{c}": self.loss_vol(self.get_layout(o, 'vol'), self.get_layout(sdf_c, 'vol'))})
            losses.update(
                {f"loss_near_{c}": self.loss_near(self.get_layout(o, 'near'), self.get_layout(sdf_c, 'near'))})
            losses.update(
                {f"loss_surface_{c}": self.loss_surface(self.get_layout(o, 'surface'), 0.0)})
            losses.update({f"loss_eikonal_{c}": self.loss_eikonal(queries_c, o)})

        return losses

    def forward_test(self, pc, queries, block_size=100000):
        B, C, N, _ = pc.shape

        if self.mask_cfgs is not None:
            all_masks = self._generate_mask(pc, training=False)
            bottleneck = self.encode(pc, all_masks, parts=self.parts)
        else:
            all_masks = None
            bottleneck = self.encode(pc)

        x = self.decode(bottleneck['x'], all_masks)
        x = rearrange(x, '(b c) n d -> b c n d', b=B)

        o_list = []
        for c in range(queries.shape[1]):
            queries_c = queries[:, c]
            x_c = x[:, c]

            if queries_c.shape[1] > block_size:
                N = block_size
                os = []
                for block_idx in range(math.ceil(queries_c.shape[1] / N)):
                    o = self.query(x_c, queries_c[:, block_idx * N:(block_idx + 1) * N, :]).squeeze(-1)
                    os.append(o)
                o = torch.cat(os, dim=1)
            else:
                o = self.query(x_c, queries_c).squeeze(-1)
            o_list.append(o)

        o = torch.stack(o_list, dim=1)

        return o

    def reconstruct_mesh(self, data):
        resolution = data['resolution']
        gap = 2. / resolution
        grid = generate_grid(resolution).reshape(-1, 3).cuda()
        grid = repeat(grid, 'n d -> b c n d', b=1, c=self.num_classes)

        outputs = self.forward_test(data['surface'], grid)
        volume = outputs.view(self.num_classes, resolution + 1, resolution + 1, resolution + 1).cpu().numpy() * (-1)

        meshes = []
        for c in range(self.num_classes):
            verts, faces = mcubes.marching_cubes(volume[c], 0)
            verts *= gap
            verts -= 1.
            mesh = trimesh.Trimesh(verts, faces)
            meshes.append(mesh)

        if 'return_volume' in data:
            arr = np.zeros((resolution + 1, resolution + 1, resolution + 1))
            for c in range(self.num_classes):
                arr[volume[c] > 0] = c + 1
            arr = arr.astype(np.int32)

            return meshes, arr

        return meshes

    def load_mesh(self, filename):
        mesh = trimesh.load(filename)

        shifts = np.array(self.shape_cfgs.get('shifts', [13.59974129, 9.31580232, -1.01333806]))
        mesh.apply_translation(-shifts)
        mesh.apply_scale(self.shape_cfgs.get('scale', 1.0 / 100))

        return mesh

    def load_volume(self, filename):
        volume = np.load(filename)['volume']

        return volume

    @auto_fp16()
    def forward(self, data, return_loss=True, return_label=False):
        if return_loss:
            self.latest_data = data

            # [B, C, P1, 3], [B, C, P1], [B, C, P2, 3]
            points, sdf, surface = data['points'], data['sdf'], data['surface']
            points = points.requires_grad_(True)
            points_all = torch.cat([points, surface], dim=2)
            return self.forward_train(surface, points_all, sdf)
        else:
            points, sdf, surface = data['points'], data['sdf'], data['surface']
            results = {}

            # this means we will also do marching cube here
            if data.get('resolution', None) is not None:
                output = self.reconstruct_mesh(data)

                if 'return_volume' not in data:
                    results.update({'pred_meshes': output})
                else:
                    results.update({'pred_meshes': output[0]})
                    results.update({'pred_volume': output[1]})

            if return_label:
                results.update({'o': self.forward_test(surface, points)})
                results.update({'label': data['sdf']})

                gt_meshes = []
                parts = data['parts'][0].cpu().numpy()
                for part in parts:
                    gt_mesh_name = data['sample_metas'][0]['sample_info'].replace('_sdf.npz', f'_{part + 1}.obj')
                    gt_mesh = self.load_mesh(gt_mesh_name)

                    gt_meshes.append(gt_mesh)
                results.update({"gt_meshes": gt_meshes})

                if 'return_volume' in data:
                    gt_volume_name = data['sample_metas'][0]['sample_info'].replace('_sdf.npz', '_volume.npz')
                    results.update({'gt_volume': self.load_volume(gt_volume_name)})
            return results

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        losses = self(data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data["points"].shape[0])

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data["points"].shape[0])

        return outputs

    def load_pretrained(self, pretrained):
        if isinstance(pretrained, str):
            pretrained = dict(type="Pretrained", checkpoint=pretrained)
            from reimu.models import initialize

            initialize(self, pretrained)
