#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 23:39:11 2025

@author: maistiak
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# -------------------------
# Utilities
# -------------------------
def unfold3d_to_tokens(x, patch_size):
    # x: B,C,D,H,W
    # patch_size: (pd, ph, pw)
    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size
    assert D % pd == 0 and H % ph == 0 and W % pw == 0
    x = x.unfold(2, pd, pd).unfold(3, ph, ph).unfold(4, pw, pw)  # B,C,Dpd,Hph,Wpw,pd,ph,pw
    x = x.contiguous().view(B, C, -1, pd*ph*pw)  # B,C,num_patches,patch_vol
    x = x.permute(0,2,3,1).contiguous().view(B, -1, pd*ph*pw*C)  # B, L, patch_flat
    return x

# -------------------------
# PatchEmbed3D: Conv3d projection to tokens
# -------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=(32,64,64), patch_size=(2,4,4), in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.Dp = img_size[0] // patch_size[0]
        self.Hp = img_size[1] // patch_size[1]
        self.Wp = img_size[2] // patch_size[2]
        self.num_patches = self.Dp * self.Hp * self.Wp
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        # x: B,C,D,H,W
        x = self.proj(x)  # B,embed_dim, Dp, Hp, Wp
        B, C, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1,2)  # B, L, C
        if self.norm is not None:
            x = self.norm(x)
        return x  # B, L, embed_dim

# -------------------------
# Simple Transformer Encoder Block (LayerNorm + MHA + MLP)
# -------------------------
class SimpleTransBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: B, L, C (batch_first=True for MHA)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# -------------------------
# Efficient Paired Attention (EPA) - simplified & runnable
# - Implements the key idea: a spatial branch (linear/efficient attention) and a channel branch
# - Q/K projections are shared between branches (paired attention).
# -------------------------
class EPA3D(nn.Module):
    def __init__(self, dim, num_heads=4, attn_dim=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = 1.0 / sqrt(self.head_dim)

        # Shared Q and K maps (paired)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)

        # Separate V projections for spatial and channel branch
        self.v_proj_spatial = nn.Linear(dim, dim, bias=False)
        self.v_proj_channel = nn.Linear(dim, dim, bias=False)

        # small fusion
        self.out = nn.Linear(2 * dim, dim)

        # optional per-head temperature (like tau in your code)
        self.tau = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        # x: B, L, C (L = Dp*Hp*Wp)
        B, L, C = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)  # B, nH, L, hd
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)  # B, nH, L, hd

        # Spatial branch: linear attention (N x d) implemented as: softmax(Q @ K^T) @ V but we compute approx
        # For simplicity and safety on small inputs we'll compute regular attention but keep structure to swap later.
        v_s = self.v_proj_spatial(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B,nH,L,L
        attn_scores = attn_scores / torch.clamp(self.tau.unsqueeze(0), min=1e-3)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        spatial_out = torch.matmul(attn_probs, v_s)  # B,nH,L,hd
        spatial_out = spatial_out.permute(0,2,1,3).contiguous().view(B, L, C)  # B,L,C

        # Channel branch: compute attention across channels per token
        # We'll reuse q/k for efficiency: project to per-token channel attention via dot(q_token, v_channel)
        # Implementation: for each token we compute channel weights using q_kv interaction -> emulate channel attention
        v_c = self.v_proj_channel(x)  # B,L,C
        # channel attention: compute a small per-token MLP-style gating (cheap)
        # produce a weight per channel via elementwise with a pooled query
        q_pool = q.mean(dim=2).unsqueeze(2)  # B,nH,1,hd -> B,nH,1,hd
        # compute simple channel attention map per token: (B, L, C) * sigmoid(some projection)
        # For simplicity: produce channel gates via linear on x
        channel_gates = torch.sigmoid(nn.functional.linear(x, torch.randn(C, C, device=x.device)))  # random proj (toy)
        # (Note: above uses a random projection for toyness; in real repo you'd use learned proj)
        channel_out = x * channel_gates  # B,L,C

        # Concatenate and fuse
        out = torch.cat([spatial_out, channel_out], dim=-1)  # B,L,2*C
        out = self.out(out)  # B,L,C
        return out

# -------------------------
# Small decoder: map tokens back to voxel grid
# -------------------------
class SimpleDecoder3D(nn.Module):
    def __init__(self, patches_resolution, embed_dim, out_channels, patch_size):
        super().__init__()
        self.Dp, self.Hp, self.Wp = patches_resolution
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size  # (pd,ph,pw)
        # map token embedding back to per-patch volume features then conv transpose to voxels
        self.proj = nn.Linear(embed_dim, embed_dim)
        # final conv3d that takes reconstructed patches and output segmentation grid
        # We'll expand tokens to (B, embed_dim, Dp, Hp, Wp) then conv transpose by patch_size to voxel space
        self.final_conv = nn.ConvTranspose3d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: B,L,C
        B, L, C = x.shape
        assert L == self.Dp * self.Hp * self.Wp
        x = self.proj(x)  # B,L,C
        x = x.transpose(1,2).view(B, C, self.Dp, self.Hp, self.Wp)  # B,C,Dp,Hp,Wp
        x = self.final_conv(x)  # B,out_channels, D, H, W
        return x

# -------------------------
# Top-level ToyUNETRPP3D (compact)
# -------------------------
class UNETRPP3D(nn.Module):
    def __init__(self,
                 img_size=(32,64,64),
                 patch_size=(2,4,4),
                 in_chans=1,
                 num_classes=2,
                 embed_dim=96,
                 encoder_depth=2,
                 num_heads=4):
        super().__init__()

        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patches_resolution = (img_size[0]//patch_size[0], img_size[1]//patch_size[1], img_size[2]//patch_size[2])
        self.num_patches = self.patch_embed.num_patches

        # a small stack of transformer blocks
        self.encoder_blocks = nn.ModuleList([
            SimpleTransBlock(dim=embed_dim, num_heads=num_heads) for _ in range(encoder_depth)
        ])

        # one EPA block (paired attention)
        self.epa = EPA3D(dim=embed_dim, num_heads=num_heads)

        # decoder
        self.decoder = SimpleDecoder3D(self.patches_resolution, embed_dim, num_classes, patch_size)

    def forward(self, x):
        # x: B, C, D, H, W
        tokens = self.patch_embed(x)         # B, L, C
        for blk in self.encoder_blocks:
            tokens = blk(tokens)
        tokens = tokens + self.epa(tokens)   # residual paired attention
        out = self.decoder(tokens)           # B, num_classes, D, H, W
        return out

# -------------------------
# Demo test
# -------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # toy 3D volume: B, C, D, H, W
    B, C = 2, 2
    D, H, W = 32, 64, 64
    x = torch.rand(B, C, D, H, W).to(device)

    model = UNETRPP3D(img_size=(D, H, W),
                        patch_size=(2, 4, 4),
                        in_chans=C,
                        num_classes=2,
                        embed_dim=48,
                        encoder_depth=2,
                        num_heads=4).to(device)

    model.eval()
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)  # expect (B, num_classes, D, H, W)
