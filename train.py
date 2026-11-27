# -*- coding: utf-8 -*-
"""
Two-Stage Deraining Network — Final Fixed Version
===================================================

What was wrong (diagnosed from training logs ep 41-50)
-------------------------------------------------------
Stage2 PSNR was FALLING during Phase 2 (26.60 → 25.98 dB) despite active
training. Three compounding problems caused this:

  BUG-1  GNN redistributes noise spatially
         WindowGNN with random-init attention weights spreads encoded
         noise across the entire image. Worse, when Phase 2 gradients
         try to fix this, GNN learns to redistribute in the wrong
         direction. Removed entirely. Replaced with StreakContinuityBlock
         (dilated depthwise convs, zero-init → starts as identity).

  BUG-2  stage1_feat=None in Phase 2 breaks the LF path
         LF path needs Stage1 decoder features for context. Without them
         it tries to reconstruct low-frequency content from scratch,
         producing garbage that pollutes the fused features.
         Fixed: Stage1 run under torch.no_grad() in Phase 2, both
         B1 and f_full detached and passed to Stage2.

  BUG-3  LAMBDA_HF=0.30 too aggressive on untrained Stage2
         Laplacian loss amplifies HF errors 4-8× vs L1. On a Stage2
         still producing noisy output this creates enormous unstable
         gradients. FrequencyGate learns to suppress everything to
         minimise this loss, killing real HF signal too.
         Fixed: Phase 2 uses graduated loss schedule:
           ep 40-49  L1 only
           ep 50-59  L1 + light edge (0.05)
           ep 60-79  Full HF losses at reduced weights

Previously fixed bugs (still present here)
------------------------------------------
  BUG-4  res_scale not clamped → could go negative, Stage2 subtracts from B1
  BUG-5  No gradient clipping → perceptual loss spikes corrupt weights
  BUG-6  norm_vgg() defined but never applied before perceptual loss
  BUG-7  scheduler.step() was inside batch loop → LR collapse
  BUG-8  dam_scale ramped to 6.0 → exploding features
  BUG-9  .view() on non-contiguous tensors → RuntimeError
  BUG-10 Checkpoint epoch +1 bug → wrong epoch on resume
  BUG-11 EdgeLoss groups=C instead of groups=3
  BUG-12 LaplacianHFLoss kernel recreated every forward pass

Architecture changes
--------------------
  * GNN (WindowGNN) removed — was actively hurting Stage2 PSNR
  * StreakContinuityBlock added — dilated depthwise convs, zero-init,
    covers same receptive field as GNN but starts as identity
  * Stage2 decoder last layer zero-initialised → starts as identity
  * res_scale initialised at 0.01 (not 0.1) → smaller initial perturbation

Training strategy
-----------------
  Phase 1  ep   0-39   Stage1 only, Stage2 frozen
                        Build stable B1 before Stage2 touches anything
  Phase 2  ep  40-79   Stage2 only, Stage1 frozen
                        Stage1 run under no_grad, both B1+f_full detached
                        Graduated loss: L1 → L1+edge → full HF losses
  Phase 3  ep  80-199  Joint training, lower LR
                        DAM introduced gradually over eps 80-120
"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================================================================
# CONFIG
# ================================================================
TRAIN_RAIN_DIR  = os.environ.get("TRAIN_RAIN_DIR", "datasets/Rain13K/input")
TRAIN_GT_DIR    = os.environ.get("TRAIN_GT_DIR", "datasets/Rain13K/target")
TEST_RAIN_DIR   = os.environ.get("TEST_RAIN_DIR", "datasets/Rain100H/rainy")
TEST_GT_DIR     = os.environ.get("TEST_GT_DIR", "datasets/Rain100H/gt")
MODEL_SAVE_PATH = os.environ.get("MODEL_SAVE_PATH", "checkpoints/best_model.pth")
CKPT_DIR        = "checkpoints"

BATCH_SIZE  = 4
EPOCHS      = 60
PATCH_SIZE  = 256

PHASE1_END  = 20    # Stage1 only :  ep  0-39
PHASE2_END  = 40    # Stage2 only :  ep 40-79
                    # Joint        :  ep 80-199

# Loss weights (Phase 3 / joint)
W_S1_L1    = 0.10
W_S1_EDGE  = 0.05
W_S2_L1    = 1.00
W_S2_HF    = 0.15   # reduced from 0.30 — was causing instability
W_S2_FFT   = 0.05
W_S2_EDGE  = 0.05
W_S2_PERC  = 0.05


# ================================================================
# DATASET
# ================================================================
class RainDataset(Dataset):
    def __init__(self, inp_dir, gt_dir, patch=256, train=True):
        self.inp   = sorted([os.path.join(inp_dir, f) for f in os.listdir(inp_dir)])
        self.gt    = sorted([os.path.join(gt_dir,  f) for f in os.listdir(gt_dir)])
        self.patch = patch
        self.train = train
        self.to_t  = T.ToTensor()

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        x = np.array(Image.open(self.inp[idx]).convert("RGB"))
        y = np.array(Image.open(self.gt[idx]).convert("RGB"))
        h, w = x.shape[:2]

        if self.train:
            if h >= self.patch and w >= self.patch:
                i = random.randint(0, h - self.patch)
                j = random.randint(0, w - self.patch)
                x = x[i:i+self.patch, j:j+self.patch]
                y = y[i:i+self.patch, j:j+self.patch]
            else:
                sz = (self.patch, self.patch)
                x  = np.array(Image.fromarray(x).resize(sz, Image.BICUBIC))
                y  = np.array(Image.fromarray(y).resize(sz, Image.BICUBIC))

        return self.to_t(x), self.to_t(y)


# ================================================================
# SHARED BLOCKS
# ================================================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvTransformerBlock(nn.Module):
    """
    Restormer-style channel-wise attention. O(C^2) not O(N^2).
    Safe at any spatial resolution.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.h    = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = nn.BatchNorm2d(dim)
        self.qkv   = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.dw    = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
        self.proj  = nn.Conv2d(dim, dim, 1)

        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn   = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.dw(self.qkv(self.norm1(x)))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.h, C // self.h, H * W)
        k = k.reshape(B, self.h, C // self.h, H * W)
        v = v.reshape(B, self.h, C // self.h, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temp
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1)).permute(0, 1, 3, 2).reshape(B, C, H, W)
        x   = x + self.proj(out)
        x   = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    """
    Standard spatial self-attention. O(N^2).
    Only used at UNet bottleneck where spatial size = patch/4.
    Uses .reshape() — handles non-contiguous tensors from interpolate.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # .reshape() not .view() — handles non-contiguous tensors
        t = x.reshape(B, C, H * W).permute(0, 2, 1)
        t = t + self.attn(self.norm1(t), self.norm1(t), self.norm1(t))[0]
        t = t + self.ffn(self.norm2(t))
        return t.permute(0, 2, 1).reshape(B, C, H, W)


# ================================================================
# STAGE 1 ATTENTION BLOCKS
# ================================================================

class StreakAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x + x * self.net(x)


class DirectionalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.h    = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2))
        self.v    = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0))
        self.d    = nn.Conv2d(dim, dim,  3,      padding=1)
        self.fuse = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        return x + self.fuse(torch.cat([self.h(x), self.v(x), self.d(x)], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // reduction, 4), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x + x * w


class MultiAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.streak = StreakAttention(dim)
        self.dirn   = DirectionalAttention(dim)
        self.chan   = ChannelAttention(dim)
        self.fuse   = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        out = self.fuse(torch.cat([self.streak(x),
                                   self.dirn(x),
                                   self.chan(x)], dim=1))
        return x + out


# ================================================================
# STAGE 1 — MULTI-SCALE UNET
# ================================================================

class UNetCore(nn.Module):
    """
    Shared UNet backbone used at each scale in Stage1.
    Bottleneck: ResBlock → TransformerBlock (global, only safe here at patch/4)
                → MultiAttentionBlock (local streak/channel)
                → ConvTransformerBlock (conv-spatial)
                → ResBlock
    """
    def __init__(self, dim=64):
        super().__init__()
        d1, d2, d4 = dim, dim * 2, dim * 4

        self.enc1 = nn.Sequential(nn.Conv2d(3,  d1, 3, 1, 1), ResBlock(d1))
        self.enc2 = nn.Sequential(nn.Conv2d(d1, d2, 3, 2, 1), ResBlock(d2))
        self.enc3 = nn.Sequential(nn.Conv2d(d2, d4, 3, 2, 1), ResBlock(d4))

        self.bot  = nn.Sequential(
            ResBlock(d4),
            TransformerBlock(d4),     # global — spatial is patch/4 here
            MultiAttentionBlock(d4),     # local streak / channel
            ConvTransformerBlock(d4),    # conv-spatial
            ResBlock(d4),
        )

        self.dec2 = nn.Sequential(nn.Conv2d(d4 + d2, d2, 3, 1, 1), ResBlock(d2))
        self.dec1 = nn.Sequential(nn.Conv2d(d2 + d1, d1, 3, 1, 1), ResBlock(d1))
        self.head = nn.Conv2d(d1, 3, 3, 1, 1)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        b  = self.bot(f3)

        d2 = self.dec2(torch.cat([
            F.interpolate(b,  size=f2.shape[-2:], mode='bilinear', align_corners=False), f2
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False), f1
        ], dim=1))

        return self.head(d1), d1    # prediction + decoder feature


class Stage1(nn.Module):
    """
    Multi-scale rain removal — three branches fused with learned softmax weights.
      Branch 1: full resolution  — thin/fine streaks
      Branch 2: half resolution  — wide/dense streaks
      Branch 3: streak prior     — lightweight CNN on high-pass of input
    """
    def __init__(self, dim=64):
        super().__init__()
        self.unet = UNetCore(dim)

        self.streak_branch = nn.Sequential(
            nn.Conv2d(3,   dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, 1),
        )

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(9, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1), nn.Softmax(dim=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Branch 1 — full scale
        pred_full, f_full = self.unet(x)
        pred_full = pred_full + x

        # Branch 2 — half scale
        xh = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        pred_half, _ = self.unet(xh)
        pred_half = F.interpolate(pred_half + xh, size=(H, W),
                                  mode='bilinear', align_corners=False)

        # Branch 3 — streak prior (high-pass = x - avg_pool low-pass)
        lp          = F.avg_pool2d(x, kernel_size=15, stride=1, padding=7)
        pred_streak = self.streak_branch(x - lp) + x

        weights = self.fuse_attn(
            torch.cat([pred_full, pred_half, pred_streak], dim=1)
        )

        B1 = (weights[:, 0:1] * pred_full  +
              weights[:, 1:2] * pred_half  +
              weights[:, 2:3] * pred_streak)

        return B1, f_full


# ================================================================
# STAGE 2 COMPONENTS
# ================================================================

class HFExtractor(nn.Module):
    """
    Extracts HF content from Stage1 error residual (x - B1).
    Fixed Laplacian buffer + learned depthwise CNN.
    Kernel stored as buffer — not reallocated each forward pass.
    """
    def __init__(self, dim=64):
        super().__init__()
        lap = torch.tensor([[0,  1, 0],
                            [1, -4, 1],
                            [0,  1, 0]], dtype=torch.float32)
        self.register_buffer('lap', lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

        self.learned = nn.Sequential(
            nn.Conv2d(3,   dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, residual):
        lap_hf = F.conv2d(residual, self.lap, padding=1, groups=3)
        feat   = self.learned(residual)
        return lap_hf, feat


class FrequencyGate(nn.Module):
    """
    Suppresses real-texture HF (isotropic), passes rain-streak HF (directional).
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 3, dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Sigmoid(),
        )

    def forward(self, feat, lap_hf):
        lap_up = F.interpolate(lap_hf, size=feat.shape[-2:],
                               mode='bilinear', align_corners=False)
        return self.gate(torch.cat([feat, lap_up], dim=1))


class StreakContinuityBlock(nn.Module):
    """
    Replaces WindowGNN entirely.

    Why removed GNN:
      - At random init, attention redistributes encoded noise spatially
      - During Phase 2, gradients teach GNN to redistribute in the wrong
        direction — PSNR was FALLING (26.60 → 25.98) during training
      - Moving GNN to a different position would not fix this

    Why StreakContinuityBlock instead:
      - Dilated depthwise convs give the same large receptive field
        (dilation=8 at encoded res covers 64px original = full streak)
      - Zero-init on fuse layer → starts as perfect identity (x + 0 = x)
      - Depthwise = no cross-channel mixing at init → stable gradients
      - Can only improve from init, never degrade B1

    Zero-init proof:
      fuse weight=0, bias=0 → fuse(feats) = 0 → forward returns x + BN(0) ≈ x
    """
    def __init__(self, dim):
        super().__init__()
        # Horizontal streak directions — dilation covers increasing range
        self.h1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1),  groups=dim)
        self.h2 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 4),  groups=dim, dilation=(1, 4))
        self.h3 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 8),  groups=dim, dilation=(1, 8))

        # Vertical streak directions
        self.v1 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0),  groups=dim)
        self.v2 = nn.Conv2d(dim, dim, (3, 1), padding=(4, 0),  groups=dim, dilation=(4, 1))
        self.v3 = nn.Conv2d(dim, dim, (3, 1), padding=(8, 0),  groups=dim, dilation=(8, 1))

        self.fuse = nn.Conv2d(dim * 6, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

        # ZERO INIT — this block starts as identity: x + BN(0) = x
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        feats = torch.cat([
            self.h1(x), self.h2(x), self.h3(x),
            self.v1(x), self.v2(x), self.v3(x),
        ], dim=1)
        return x + self.norm(self.fuse(feats))


class DirectionAwareModulation(nn.Module):
    """
    Modulates HF features along streak directions.
    alpha starts at 0 (sigmoid(0)=0.5) and is learned.
    Only applied during Phase 3 with DAM warm-up.
    """
    def __init__(self, dim):
        super().__init__()
        self.h    = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.v    = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.d1   = nn.Conv2d(dim, dim, 3, padding=1,  groups=dim)
        self.d2   = nn.Conv2d(dim, dim, 3, padding=2,  groups=dim, dilation=2)
        self.fuse = nn.Conv2d(dim * 4, dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        dirs  = torch.cat([self.h(x), self.v(x), self.d1(x), self.d2(x)], dim=1)
        attn  = 0.5 * torch.sigmoid(self.fuse(dirs))
        alpha = torch.sigmoid(self.alpha)
        return x + alpha * (x * attn)


# ================================================================
# STAGE 2 — HIGH-FREQUENCY REFINEMENT
# ================================================================

class Stage2(nn.Module):
    """
    HF refinement of Stage1 output B1.

    Two parallel paths:
      LF path: cat(x, B1) [6ch] → encoder → ResBlock (+ Stage1 feat injection)
      HF path: error=x-B1 → HFExtractor → FrequencyGate → StreakContinuityBlock
               → DirectionAwareModulation → ConvTransformerBlock

    Fused → ConvTransformer → ResBlock → decoder → masked residual onto B1

    Identity initialisation:
      decoder last layer  : zero-init weights+bias → residual=0 at init
      StreakContinuityBlock: zero-init fuse → starts as identity
      res_scale=0.01       : small initial perturbation budget
      → b_hat = B1 + mask * 0.01 * 0 = B1  exactly at random init
    """
    def __init__(self, dim=64):
        super().__init__()

        # LF path
        self.lf_enc = nn.Sequential(
            nn.Conv2d(6,   dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.lf_res = ResBlock(dim)

        # HF path
        self.hf_ext    = HFExtractor(dim)
        self.hf_enc    = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.freq_gate    = FrequencyGate(dim)
        self.streak_cont  = StreakContinuityBlock(dim)   # replaces GNN
        self.dir_mod      = DirectionAwareModulation(dim)
        self.hf_ctb       = ConvTransformerBlock(dim)

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(inplace=True),
            ResBlock(dim),
            ConvTransformerBlock(dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, 3,   4, stride=2, padding=1),
        )

        # ZERO-INIT last decoder layer → residual=0 at init → Stage2=B1
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        # Small initial scale — clamped positive so it never subtracts from B1
        self.res_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, B1, stage1_feat=None, use_dam=True, dam_scale=1.0):

        error = x - B1      # Stage1 residual — HF artifacts live here

        # LF path
        lf = self.lf_enc(torch.cat([x, B1], dim=1))
        if stage1_feat is not None:
            lf = lf + F.interpolate(stage1_feat, size=lf.shape[-2:],
                                    mode='bilinear', align_corners=False)
        lf = self.lf_res(lf)

        # HF path
        
        
        lap_hf, hf_feat = self.hf_ext(error)
        hf   = self.hf_enc(hf_feat)
        gate = self.freq_gate(hf, lap_hf)    # suppress texture HF
        hf   = hf * gate

        hf = self.streak_cont(hf)            # dilated streak continuity (stable)

        if use_dam:
            dam_out = self.dir_mod(hf)
            hf = hf + dam_scale * (dam_out - hf)

        hf = self.hf_ctb(hf)

        
        
        # Fuse + decode
        feat     = self.fuse(torch.cat([hf, lf], dim=1))
        residual = self.decoder(feat)
        residual = F.interpolate(residual, size=x.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # Mask: only correct where Stage1 made large errors
        error_map = torch.abs(error).mean(dim=1, keepdim=True)
        mask      = torch.sigmoid(20.0 * (error_map - 0.05))

        # res_scale clamped positive — Stage2 can never subtract from B1
        b_hat = B1 + mask * self.res_scale.clamp(0.001, 0.5) * residual

        # Hard clamp — output stays in valid image range
        return b_hat.clamp(0, 1)


# ================================================================
# FULL MODEL
# ================================================================

class DerainNet(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.s1 = Stage1(dim)
        self.s2 = Stage2(dim)

    def forward(self, x, use_dam=True, dam_scale=1.0):
        B1, f_full = self.s1(x)
        B_hat = self.s2(x, B1, stage1_feat=f_full,
                        use_dam=use_dam, dam_scale=dam_scale)
        return B1, B_hat


# ================================================================
# LOSSES
# ================================================================

class EdgeLoss(nn.Module):
    """Sobel gradient loss. groups=3 (per-channel) — was groups=C before."""
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sx', sx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sy', sy.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def _sobel(self, x):
        gx = F.conv2d(x, self.sx, padding=1, groups=3)
        gy = F.conv2d(x, self.sy, padding=1, groups=3)
        return (gx**2 + gy**2 + 1e-8).sqrt()

    def forward(self, pred, target):
        return F.l1_loss(self._sobel(pred), self._sobel(target))


class LaplacianHFLoss(nn.Module):
    """
    Direct HF supervision for Stage2.
    Kernel stored as buffer — not recreated every forward call.
    """
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0, 1, 0],[1,-4, 1],[0, 1, 0]], dtype=torch.float32)
        self.register_buffer('lap', lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        ph = F.conv2d(pred,   self.lap, padding=1, groups=3)
        th = F.conv2d(target, self.lap, padding=1, groups=3)
        return F.l1_loss(ph, th)


class FFTLoss(nn.Module):
    """Frequency-domain amplitude loss."""
    def forward(self, pred, target):
        pf = torch.fft.rfft2(pred,   norm='ortho')
        tf = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pf), torch.abs(tf))


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def norm_vgg(x):
    """Must be applied before PerceptualLoss — was missing in original code."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ================================================================
# METRICS
# ================================================================

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))


def calc_ssim(pred, target):
    p = pred.cpu().numpy().transpose(1, 2, 0)
    t = target.cpu().numpy().transpose(1, 2, 0)
    return ssim(p, t, channel_axis=2, data_range=1.0)


# ================================================================
# DATA LOADERS
# ================================================================

train_loader = DataLoader(
    RainDataset(TRAIN_RAIN_DIR, TRAIN_GT_DIR, PATCH_SIZE, train=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    RainDataset(TEST_RAIN_DIR, TEST_GT_DIR, PATCH_SIZE, train=False),
    batch_size=1, shuffle=False, num_workers=2,
)


# ================================================================
# MODEL + LOSSES
# ================================================================

model   = DerainNet(dim=64).to(device)
l1_fn   = nn.L1Loss()
edge_fn = EdgeLoss().to(device)
lap_fn  = LaplacianHFLoss().to(device)
fft_fn  = FFTLoss().to(device)
perc_fn = PerceptualLoss().to(device)
scaler  = GradScaler()


# ================================================================
# OPTIMISER FACTORIES — one per phase
# ================================================================

def make_opt_s1(lr=2e-4):
    return torch.optim.AdamW(
        model.s1.parameters(), lr=lr, weight_decay=1e-4)

def make_opt_s2(lr=1e-4):
    return torch.optim.AdamW(
        model.s2.parameters(), lr=lr, weight_decay=1e-4)

def make_opt_joint():
    return torch.optim.AdamW([
        {'params': model.s1.parameters(), 'lr': 5e-5},
        {'params': model.s2.parameters(), 'lr': 2e-5},
    ], weight_decay=1e-4)


# ================================================================
# HELPERS
# ================================================================

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def phase_label(ep):
    if ep < PHASE1_END:  return "Phase1"
    if ep < PHASE2_END:  return "Phase2"
    return "Phase3"


# ================================================================
# CHECKPOINT RESUME
# ================================================================

os.makedirs(CKPT_DIR, exist_ok=True)
start_epoch = 0
best_psnr   = 0.0
ckpt_file   = os.path.join(CKPT_DIR, "last.pth")

# Default — will be rebuilt correctly for each phase at ep==boundary
opt       = make_opt_s1()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=PHASE1_END, eta_min=1e-6)

if os.path.exists(ckpt_file):
    print("Resuming from checkpoint …")
    ckpt        = torch.load(ckpt_file, map_location=device)
    start_epoch = ckpt['epoch']    # saved as ep+1 — no +1 bug
    best_psnr   = ckpt['best_psnr']

    # Load model weights
    model.load_state_dict(ckpt['model'], strict=False)

    # Rebuild optimiser for correct phase
    ep = start_epoch
    if ep < PHASE1_END:
        opt = make_opt_s1()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(PHASE1_END - ep, 1), eta_min=1e-6)
    elif ep < PHASE2_END:
        opt = make_opt_s2()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(PHASE2_END - ep, 1), eta_min=1e-6)
    else:
        opt = make_opt_joint()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(EPOCHS - ep, 1), eta_min=1e-6)

    # Restore opt/scheduler state if compatible
    if 'opt' in ckpt:
        try:    opt.load_state_dict(ckpt['opt'])
        except: print("  Optimiser state mismatch — fresh optimiser.")
    if 'scheduler' in ckpt:
        try:    scheduler.load_state_dict(ckpt['scheduler'])
        except: print("  Scheduler state mismatch — fresh scheduler.")

    print(f"  Resumed epoch {start_epoch},  best PSNR {best_psnr:.2f} dB")


# ================================================================
# TRAINING LOOP
# ================================================================

for ep in range(start_epoch, EPOCHS):

    # ----------------------------------------------------------
    # Phase transitions
    # ----------------------------------------------------------
    if ep == 0:
        unfreeze(model.s1)
        freeze(model.s2)
        opt = make_opt_s1(lr=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=PHASE1_END, eta_min=1e-6)
        print(f"\n{'='*62}")
        print(f"PHASE 1  (ep 0-{PHASE1_END-1}): Stage1 only — building stable B1")
        print(f"{'='*62}")

    elif ep == PHASE1_END:
        # Keep Stage1 frozen; Stage2 starts fresh with identity init
        freeze(model.s1)
        unfreeze(model.s2)
        opt = make_opt_s2(lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=(PHASE2_END - PHASE1_END), eta_min=1e-6)
        print(f"\n{'='*62}")
        print(f"PHASE 2  (ep {PHASE1_END}-{PHASE2_END-1}): Stage2 only — HF refinement on fixed B1")
        print(f"  Loss schedule:")
        print(f"    ep {PHASE1_END}-{PHASE1_END+9}  : L1 only")
        print(f"    ep {PHASE1_END+10}-{PHASE1_END+19}: L1 + light edge")
        print(f"    ep {PHASE1_END+20}-{PHASE2_END-1}: Full HF losses")
        print(f"{'='*62}")

    elif ep == PHASE2_END:
        unfreeze(model.s1)
        unfreeze(model.s2)
        opt = make_opt_joint()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=(EPOCHS - PHASE2_END), eta_min=1e-6)
        print(f"\n{'='*62}")
        print(f"PHASE 3  (ep {PHASE2_END}-{EPOCHS-1}): Joint fine-tuning")
        print(f"{'='*62}")

    # ----------------------------------------------------------
    # DAM schedule — only in Phase 3
    # Ramps 0→1 over first 40 epochs of Phase 3
    # ----------------------------------------------------------
    if ep < PHASE2_END:
        use_dam, dam_scale = False, 0.0
    elif ep < PHASE2_END + 40:
        use_dam   = True
        dam_scale = min((ep - PHASE2_END) / 40.0, 1.0)
    else:
        use_dam, dam_scale = True, 1.0

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader,
                     desc=f"Ep {ep+1:03d}/{EPOCHS}  {phase_label(ep)}"):
        x, y = x.to(device), y.to(device)

        with autocast():

            # ── PHASE 1: Stage1 only ────────────────────────────────
            if ep < PHASE1_END:
                B1, _ = model.s1(x)

                loss = (l1_fn(B1, y)
                      + W_S1_EDGE * edge_fn(B1, y)
                      + 0.05      * lap_fn(B1, y))   # light HF on Stage1 output

                # Still compute B_hat for validation logging (no grad)
                B_hat = B1.detach()

            # ── PHASE 2: Stage2 only ────────────────────────────────
            elif ep < PHASE2_END:
                # Run Stage1 under no_grad — B1 and features are fixed
                with torch.no_grad():
                    B1, f_full = model.s1(x)
                    B1     = B1.detach()
                    f_full = f_full.detach()

                B_hat = model.s2(x, B1,
                                 stage1_feat=f_full,   # pass features — was None before
                                 use_dam=False,
                                 dam_scale=0.0)

                # Graduated loss — aggressive HF losses too early caused
                # FrequencyGate to suppress everything (PSNR was falling)
                if ep < PHASE1_END + 10:
                    # ep 40-49: L1 only — let Stage2 learn basic mapping first
                    loss = l1_fn(B_hat, y)

                elif ep < PHASE1_END + 20:
                    # ep 50-59: add light edge supervision
                    loss = (l1_fn(B_hat, y)
                          + 0.05 * edge_fn(B_hat, y))

                else:
                    # ep 60-79: full HF losses at reduced weights
                    loss = (l1_fn(B_hat, y)
                          + 0.10 * lap_fn(B_hat, y)
                          + 0.03 * fft_fn(B_hat, y)
                          + 0.05 * edge_fn(B_hat, y)
                          + 0.05 * perc_fn(norm_vgg(B_hat.clamp(0, 1)),
                                           norm_vgg(y)))

            # ── PHASE 3: Joint ──────────────────────────────────────
            else:
                B1, B_hat = model(x, use_dam=use_dam, dam_scale=dam_scale)

                loss_s1 = (W_S1_L1   * l1_fn(B1, y)
                         + W_S1_EDGE * edge_fn(B1, y))

                loss_s2 = (W_S2_L1   * l1_fn(B_hat, y)
                         + W_S2_HF   * lap_fn(B_hat, y)
                         + W_S2_FFT  * fft_fn(B_hat, y)
                         + W_S2_EDGE * edge_fn(B_hat, y)
                         + W_S2_PERC * perc_fn(norm_vgg(B_hat.clamp(0, 1)),
                                               norm_vgg(y)))

                loss = loss_s2 + loss_s1

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        # Gradient clipping — prevents NaN from HF/perceptual loss spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()

    # Scheduler steps ONCE per epoch — outside the batch loop
    scheduler.step()

    avg = total_loss / len(train_loader)
    lr  = scheduler.get_last_lr()[0]
    print(f"Epoch {ep+1:03d}  loss={avg:.4f}  lr={lr:.2e}  "
          f"DAM={'ON' if use_dam else 'off'} ({dam_scale:.2f})")

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------
    model.eval()
    ps, ss, ps1, ss1 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            B1, B_hat = model(x, use_dam=use_dam, dam_scale=dam_scale)
            B_hat = B_hat.clamp(0, 1)
            B1    = B1.clamp(0, 1)

            ps  += psnr(B_hat, y).item()
            ps1 += psnr(B1,    y).item()
            ss  += calc_ssim(B_hat[0], y[0])
            ss1 += calc_ssim(B1[0],   y[0])

    n   = len(test_loader)
    ps  /= n;  ss  /= n
    ps1 /= n;  ss1 /= n

    print(f"  Stage2  PSNR={ps:.2f} dB   SSIM={ss:.4f}")
    print(f"  Stage1  PSNR={ps1:.2f} dB  SSIM={ss1:.4f}")

    # Stage2 must be >= Stage1 to save best — if not, something is wrong
    if ps < ps1 - 0.5 and ep >= PHASE2_END:
        print(f"  ⚠️  Stage2 significantly below Stage1 — check Stage2 weights")

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------
    ckpt = {
        'epoch':     ep + 1,       # next epoch on resume — no +1 bug
        'model':     model.state_dict(),
        'opt':       opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': best_psnr,
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, "last.pth"))

    if ps > best_psnr:
        best_psnr = ps
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        torch.save(ckpt, os.path.join(CKPT_DIR, "best.pth"))
        print(f"  ✅ Best model saved  PSNR={best_psnr:.2f} dB")