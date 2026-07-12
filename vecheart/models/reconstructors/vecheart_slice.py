# Copyright (c) 2025, Biao Zhang.
import math

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
from .vecheart import VecHeart

# canonical view order; index aligns with slice_cfgs['dropout_ratio'] and per-class slice_offsets
_VIEWS = ('4CH', '2CH', 'SAX')


@RECONSTRUCTORS.register_module()
class VecHeartSlice(VecHeart):
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
            slice_cfgs=None,
            pretrained=None
    ):
        super().__init__(
            bottleneck,
            loss,
            encoder_depth,
            decoder_depth,
            dim,
            output_dim,
            num_latents,
            dim_head,
            num_classes,
            parts,
            shape_cfgs
        )

        # slice-branch settings (separate from the shared 3D-decoder shape_cfgs)
        self.slice_cfgs = slice_cfgs

        # slice encoder
        self.class_latents_slice = nn.ModuleList([nn.Embedding(num_latents, dim) for _ in range(self.num_classes)])
        self.point_embed_slice = PointEmbed(dim=dim)
        self.encoder_cross_attn_slice = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=dim // dim_head, dim_head=dim_head)),
            PreNorm(dim, FeedForward(dim))
        ])
        self.encoder_layers_slice = nn.ModuleList([])
        for i in range(encoder_depth):
            self.encoder_layers_slice.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=dim // dim_head, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim))
            ]))
        self.bottleneck_slice = build_component(bottleneck)

        self.loss_recon = build_loss(loss.recon)

        self.slice_seps = self.slice_cfgs['slice_seps']
        if pretrained:
            self.load_pretrained(pretrained)
            # load
            if self.slice_cfgs.get('use_pretrained', False):
                self.class_latents_slice.load_state_dict(self.class_latents.state_dict())
                self.point_embed_slice.load_state_dict(self.point_embed.state_dict())
                self.encoder_cross_attn_slice.load_state_dict(self.encoder_cross_attn.state_dict())
                self.bottleneck_slice.load_state_dict(self.bottleneck.state_dict())
            if self.slice_cfgs.get('freeze_model', True):
                self._freeze_stages()

    def encode_pc(self, pc):
        B, C, N, _ = pc.shape

        all_latents = torch.stack([latents.weight for latents in self.class_latents], dim=0)
        x = repeat(all_latents, 'c n d -> (b c) n d', b=B)

        pc = rearrange(pc, 'b c n d -> (b c) n d')
        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.encoder_cross_attn

        x = cross_attn(x, context=pc_embeddings, mask=None) + x
        x = cross_ff(x) + x

        for i_layer, (self_attn, self_ff) in enumerate(self.encoder_layers):
            x = self_attn(x) + x
            x = self_ff(x) + x

        bottleneck = self.bottleneck.pre(x)
        return bottleneck

    def _view_keep_mask(self, B, device):
        """Return a [B, 3] bool tensor over `_VIEWS`; True = that view is kept for that sample.

        Train: each in-use view is kept with prob (1 - dropout_ratio[v]); at least one in-use
        view is guaranteed per sample.
        Test: deterministic. views listed in `viu_test` are kept, the rest dropped. 
        Views absent from `viu_train`/`viu_test` are never kept.
        """
        key = 'viu_train' if self.training else 'viu_test'
        view_in_use = self.slice_cfgs.get(key, list(_VIEWS))
        usable = torch.tensor([v in view_in_use for v in _VIEWS], device=device)  # [3]

        if self.training:
            ratio = torch.tensor(self.slice_cfgs['dropout_ratio'], device=device)  # [3]
            keep = (torch.rand(B, 3, device=device) >= ratio) & usable
            # guarantee >= 1 in-use view per sample
            none = ~keep.any(dim=1)
            if none.any():
                idx = usable.nonzero(as_tuple=True)[0]
                pick = idx[torch.randint(len(idx), (int(none.sum()),), device=device)]
                keep[torch.where(none)[0], pick] = True
        else:
            keep = usable.unsqueeze(0).expand(B, -1).clone()
        return keep

    def encode_slice(self, slice_pts):
        B, N, _ = slice_pts.shape

        all_latents = torch.stack([latents.weight for latents in self.class_latents_slice], dim=0)
        x = repeat(all_latents, 'c n d -> b c n d', b=B)

        pc_embeddings = self.point_embed_slice(slice_pts)
        cross_attn, cross_ff = self.encoder_cross_attn_slice

        # keep[b, v] == True means view v is observed for sample b (None -> keep everything)
        keep = self._view_keep_mask(B, pc_embeddings.device) \
            if self.slice_cfgs.get('view_dropout', False) else None

        x_all = []
        for c in range(self.num_classes):
            x_c = x[:, c]
            pc_embeddings_c = pc_embeddings[:, self.slice_seps[c]:self.slice_seps[c + 1]]
            keep_c = torch.ones((B, pc_embeddings_c.shape[1]), dtype=torch.bool, device=pc_embeddings_c.device)
            if keep is not None:
                offsets = self.slice_cfgs['slice_offsets'][c]
                for v in range(len(offsets) - 1):
                    drop = ~keep[:, v]
                    if drop.any():
                        keep_c[drop, offsets[v]:offsets[v + 1]] = False

            # For FA Attention mask, True means valid, False means invalid.
            # So mask is keep.
            x_c_padded = torch.zeros_like(x_c)
            all_masks = torch.any(keep_c, dim=1)
            if torch.any(all_masks):
                x_c_masked = cross_attn(
                    x_c[all_masks], context=pc_embeddings_c[all_masks], mask=keep_c[all_masks]) + x_c[all_masks]
                x_c_masked = cross_ff(x_c_masked) + x_c_masked
                x_c_padded[all_masks] = x_c_masked
            x_c_padded[~all_masks] = x_c[~all_masks]

            x_all.append(x_c_padded)

        x = torch.stack(x_all, dim=1)
        x = rearrange(x, 'b c n d -> (b c) n d')

        for i_layer, (self_attn, self_ff) in enumerate(self.encoder_layers_slice):
            x = self_attn(x) + x
            x = self_ff(x) + x

        bottleneck = self.bottleneck_slice.pre(x)
        return bottleneck

    def forward_train(self, slice_pts, queries, sdf, pc, block_size=100000):
        B, C, N, _ = pc.shape

        # with torch.no_grad():
        #     bottleneck_pc = self.encode_pc(pc)
        bottleneck_slice = self.encode_slice(slice_pts)
        x = self.decode(bottleneck_slice['x'])
        x = rearrange(x, '(b c) n d -> b c n d', b=B)

        losses = {}
        # losses.update({"loss_recon": self.loss_recon(bottleneck_slice['x'], bottleneck_pc['x'])})

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

    def forward_test(self, slice_pts, queries, block_size=100000):
        B, N, _ = slice_pts.shape
        bottleneck = self.encode_slice(slice_pts)

        x = self.decode(bottleneck['x'])
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

        if data.get('slice_seps', None) is not None:
            old_slice_seps = self.slice_seps
            self.slice_seps = data['slice_seps']

        outputs = self.forward_test(data['slice_pts'], grid)
        volume = outputs.view(self.num_classes, resolution + 1, resolution + 1, resolution + 1).cpu().numpy() * (-1)

        meshes = []
        for c in range(self.num_classes):
            verts, faces = mcubes.marching_cubes(volume[c], 0)
            verts *= gap
            verts -= 1.
            mesh = trimesh.Trimesh(verts, faces)
            meshes.append(mesh)

        if data.get('slice_seps', None) is not None:
            self.slice_seps = old_slice_seps

        if 'return_volume' in data:
            arr = np.zeros((resolution + 1, resolution + 1, resolution + 1))
            for c in range(self.num_classes):
                arr[volume[c] > 0] = c + 1
            arr = arr.astype(np.int32)

            return meshes, arr

        return meshes

    @auto_fp16()
    def forward(self, data, return_loss=True, return_label=False):
        if return_loss:
            self.latest_data = data

            # [B, C, P1, 3], [B, C, P1], [B, C, P2, 3], [B, P_all, 3]
            points, sdf, surface, slice_pts = data['points'], data['sdf'], data['surface'], data['slice_pts']
            points = points.requires_grad_(True)
            points_all = torch.cat([points, surface], dim=2)
            return self.forward_train(slice_pts, points_all, sdf, surface)
        else:
            points, sdf, surface, slice_pts = data['points'], data['sdf'], data['surface'], data['slice_pts']
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
                results.update({'o': self.forward_test(slice_pts, points)})
                results.update({'label': data['sdf']})

                gt_meshes = []
                parts = data['parts'][0].cpu().numpy()
                for part in parts:
                    gt_mesh_name = data['sample_metas'][0]['sample_info'].replace('_sdf.npz', f'_{part + 1}.obj')
                    gt_mesh = self.load_mesh(gt_mesh_name)

                    gt_meshes.append(gt_mesh)
                results.update({"gt_meshes": gt_meshes})

                slice_pcs = []
                slice_pts = data['slice_pts'][0].cpu().numpy()
                view_in_use = self.slice_cfgs.get('viu_test', list(_VIEWS))
                # slice_offsets is only needed to drop per-view points; configs without
                # view dropout omit it -> keep every point.
                slice_offsets = self.slice_cfgs.get('slice_offsets', None)
                for part in parts:
                    slice_pts_part = slice_pts[self.slice_seps[part]:self.slice_seps[part + 1]]

                    keep = np.ones(slice_pts_part.shape[0], dtype=bool)
                    if slice_offsets is not None:
                        offsets = slice_offsets[part]
                        for v in range(len(offsets) - 1):
                            if _VIEWS[v] not in view_in_use:
                                keep[offsets[v]:offsets[v + 1]] = False

                    slice_pcs.append(trimesh.points.PointCloud(slice_pts_part[keep]))
                results.update({"slice_pcs": slice_pcs})

                if 'return_volume' in data:
                    gt_volume_name = data['sample_metas'][0]['sample_info'].replace('_sdf.npz', '_volume.npz')
                    results.update({'gt_volume': self.load_volume(gt_volume_name)})
            return results

    def _freeze_stages(self):
        for param in self.class_latents.parameters():
            param.requires_grad = False
        for param in self.point_embed.parameters():
            param.requires_grad = False
        for param in self.encoder_cross_attn.parameters():
            param.requires_grad = False
        for param in self.encoder_layers.parameters():
            param.requires_grad = False
        for param in self.bottleneck.parameters():
            param.requires_grad = False
        for param in self.decoder_layers.parameters():
            param.requires_grad = False
        for param in self.query_cross_attn.parameters():
            param.requires_grad = False
        for param in self.to_outputs.parameters():
            param.requires_grad = False
