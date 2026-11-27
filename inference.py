# -*- coding: utf-8 -*-
"""
DerainNet — Inference / Sample Generation Script
==================================================
Loads a trained DerainNet checkpoint, runs it on a test set,
saves output images, and reports per-image + aggregate PSNR/SSIM.

Output structure
----------------
  OUTPUT_DIR/
    derained/          ← Stage2 final output images
    stage1/            ← Stage1 intermediate output images
    results.csv        ← per-image PSNR/SSIM for both stages
    summary.txt        ← aggregate mean/best/worst statistics

Usage
-----
  python inference.py

  Adjust the CONFIG section below for your paths.
"""

import os
import csv
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================================================================
# CONFIG — edit these paths
# ================================================================
TEST_RAIN_DIR   = os.environ.get("TEST_RAIN_DIR", "datasets/Test100/input")
TEST_GT_DIR     = os.environ.get("TEST_GT_DIR", "datasets/Test100/target")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pth")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "outputs/inference")

SAVE_STAGE1     = True    # also save Stage1 intermediate outputs
SAVE_INPUT      = True    # also save original rainy input alongside output


# ================================================================
# PASTE YOUR FULL MODEL DEFINITION HERE
# (all classes exactly as in your training script)
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
            nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Conv2d(dim * 4, dim, 1),
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
        out  = (attn @ v.transpose(-2, -1)).permute(0, 1, 3, 2).reshape(B, C, H, W)
        x    = x + self.proj(out)
        x    = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.reshape(B, C, H * W).permute(0, 2, 1)
        t = t + self.attn(self.norm1(t), self.norm1(t), self.norm1(t))[0]
        t = t + self.ffn(self.norm2(t))
        return t.permute(0, 2, 1).reshape(B, C, H, W)


class StreakAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 1), nn.Sigmoid(),
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
            nn.Linear(dim, max(dim // reduction, 4)), nn.ReLU(inplace=True),
            nn.Linear(max(dim // reduction, 4), dim), nn.Sigmoid(),
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
        out = self.fuse(torch.cat([self.streak(x), self.dirn(x), self.chan(x)], dim=1))
        return x + out


class UNetCore(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        d1, d2, d4 = dim, dim * 2, dim * 4
        self.enc1 = nn.Sequential(nn.Conv2d(3,  d1, 3, 1, 1), ResBlock(d1))
        self.enc2 = nn.Sequential(nn.Conv2d(d1, d2, 3, 2, 1), ResBlock(d2))
        self.enc3 = nn.Sequential(nn.Conv2d(d2, d4, 3, 2, 1), ResBlock(d4))
        self.bot  = nn.Sequential(
            ResBlock(d4), TransformerBlock(d4), MultiAttentionBlock(d4),
            ConvTransformerBlock(d4), ResBlock(d4),
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
            F.interpolate(b,  size=f2.shape[-2:], mode='bilinear', align_corners=False), f2], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False), f1], dim=1))
        return self.head(d1), d1


class Stage1(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.unet = UNetCore(dim)
        self.streak_branch = nn.Sequential(
            nn.Conv2d(3,   dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, 1),
        )
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(9, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1), nn.Softmax(dim=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pred_full, f_full = self.unet(x)
        pred_full = pred_full + x
        xh = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        pred_half, _ = self.unet(xh)
        pred_half = F.interpolate(pred_half + xh, size=(H, W),
                                  mode='bilinear', align_corners=False)
        lp          = F.avg_pool2d(x, kernel_size=15, stride=1, padding=7)
        pred_streak = self.streak_branch(x - lp) + x
        weights = self.fuse_attn(torch.cat([pred_full, pred_half, pred_streak], dim=1))
        B1 = (weights[:, 0:1] * pred_full  +
              weights[:, 1:2] * pred_half  +
              weights[:, 2:3] * pred_streak)
        return B1, f_full


class HFExtractor(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        lap = torch.tensor([[0,  1, 0],[1, -4, 1],[0,  1, 0]], dtype=torch.float32)
        self.register_buffer('lap', lap.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.learned = nn.Sequential(
            nn.Conv2d(3,   dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )
    def forward(self, residual):
        lap_hf = F.conv2d(residual, self.lap, padding=1, groups=3)
        feat   = self.learned(residual)
        return lap_hf, feat


class FrequencyGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 3, dim, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.Sigmoid(),
        )
    def forward(self, feat, lap_hf):
        lap_up = F.interpolate(lap_hf, size=feat.shape[-2:],
                               mode='bilinear', align_corners=False)
        return self.gate(torch.cat([feat, lap_up], dim=1))


class StreakContinuityBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.h1 = nn.Conv2d(dim, dim, (1,3), padding=(0,1),  groups=dim)
        self.h2 = nn.Conv2d(dim, dim, (1,3), padding=(0,4),  groups=dim, dilation=(1,4))
        self.h3 = nn.Conv2d(dim, dim, (1,3), padding=(0,8),  groups=dim, dilation=(1,8))
        self.v1 = nn.Conv2d(dim, dim, (3,1), padding=(1,0),  groups=dim)
        self.v2 = nn.Conv2d(dim, dim, (3,1), padding=(4,0),  groups=dim, dilation=(4,1))
        self.v3 = nn.Conv2d(dim, dim, (3,1), padding=(8,0),  groups=dim, dilation=(8,1))
        self.fuse = nn.Conv2d(dim * 6, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
    def forward(self, x):
        feats = torch.cat([self.h1(x), self.h2(x), self.h3(x),
                           self.v1(x), self.v2(x), self.v3(x)], dim=1)
        return x + self.norm(self.fuse(feats))


class DirectionAwareModulation(nn.Module):
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


class Stage2(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.lf_enc = nn.Sequential(
            nn.Conv2d(6,   dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.lf_res = ResBlock(dim)
        self.hf_ext   = HFExtractor(dim)
        self.hf_enc   = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.freq_gate   = FrequencyGate(dim)
        self.streak_cont = StreakContinuityBlock(dim)
        self.dir_mod     = DirectionAwareModulation(dim)
        self.hf_ctb      = ConvTransformerBlock(dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1), nn.ReLU(inplace=True),
            ResBlock(dim), ConvTransformerBlock(dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, 3,   4, stride=2, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, B1, stage1_feat=None, use_dam=True, dam_scale=1.0):
        error = x - B1
        lf = self.lf_enc(torch.cat([x, B1], dim=1))
        if stage1_feat is not None:
            lf = lf + F.interpolate(stage1_feat, size=lf.shape[-2:],
                                    mode='bilinear', align_corners=False)
        lf = self.lf_res(lf)
        lap_hf, hf_feat = self.hf_ext(error)
        hf   = self.hf_enc(hf_feat)
        gate = self.freq_gate(hf, lap_hf)
        hf   = hf * gate
        hf   = self.streak_cont(hf)
        if use_dam:
            dam_out = self.dir_mod(hf)
            hf = hf + dam_scale * (dam_out - hf)
        hf = self.hf_ctb(hf)
        feat     = self.fuse(torch.cat([lf, hf], dim=1))
        residual = self.decoder(feat)
        residual = F.interpolate(residual, size=x.shape[-2:],
                                 mode='bilinear', align_corners=False)
        error_map = torch.abs(error).mean(dim=1, keepdim=True)
        mask      = torch.sigmoid(20.0 * (error_map - 0.05))
        b_hat = B1 + mask * self.res_scale.clamp(0.001, 0.5) * residual
        return b_hat.clamp(0, 1)


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
# METRICS
# ================================================================

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))

def calc_ssim(pred, target):
    p = pred.cpu().numpy().transpose(1, 2, 0)
    t = target.cpu().numpy().transpose(1, 2, 0)
    return ssim(p, t, channel_axis=2, data_range=1.0)

def tensor_to_pil(t):
    """Convert (C,H,W) float tensor [0,1] to PIL Image."""
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ================================================================
# LOAD MODEL
# ================================================================

model = DerainNet(dim=64).to(device)

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

# Checkpoint may be a full training dict or just state_dict
if isinstance(ckpt, dict) and 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
    print(f"  Loaded from training checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"best PSNR {ckpt.get('best_psnr', '?')})")
else:
    model.load_state_dict(ckpt)
    print("  Loaded raw state_dict")

model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params/1e6:.2f}M")


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================

dir_derained = os.path.join(OUTPUT_DIR, "derained")
dir_stage1   = os.path.join(OUTPUT_DIR, "stage1")
dir_input    = os.path.join(OUTPUT_DIR, "input")

os.makedirs(dir_derained, exist_ok=True)
if SAVE_STAGE1: os.makedirs(dir_stage1, exist_ok=True)
if SAVE_INPUT:  os.makedirs(dir_input,  exist_ok=True)


# ================================================================
# GET SORTED FILE LISTS
# ================================================================

rain_files = sorted(os.listdir(TEST_RAIN_DIR))
gt_files   = sorted(os.listdir(TEST_GT_DIR))

assert len(rain_files) == len(gt_files), \
    f"Mismatch: {len(rain_files)} rainy vs {len(gt_files)} GT images"

print(f"\nTest set  : {TEST_RAIN_DIR}")
print(f"GT set    : {TEST_GT_DIR}")
print(f"Images    : {len(rain_files)}")
print(f"Output dir: {OUTPUT_DIR}\n")

to_tensor = T.ToTensor()


# ================================================================
# INFERENCE LOOP
# ================================================================

results   = []   # per-image results for CSV
ps2_all   = []   # Stage2 PSNR per image
ss2_all   = []   # Stage2 SSIM per image
ps1_all   = []   # Stage1 PSNR per image
ss1_all   = []   # Stage1 SSIM per image
time_all  = []   # inference time per image

with torch.no_grad():
    for rain_fname, gt_fname in tqdm(zip(rain_files, gt_files),
                                     total=len(rain_files),
                                     desc="Generating"):

        # Load images
        rain_img = Image.open(os.path.join(TEST_RAIN_DIR, rain_fname)).convert("RGB")
        gt_img   = Image.open(os.path.join(TEST_GT_DIR,   gt_fname)).convert("RGB")
        rain_img = rain_img.resize((480, 320))
        gt_img = gt_img.resize((480, 320))

        
        

        x = to_tensor(rain_img).unsqueeze(0).to(device)   # (1,3,H,W)
        y = to_tensor(gt_img).unsqueeze(0).to(device)

        # Inference
        t0 = time.time()
        B1, B_hat = model(x, use_dam=True, dam_scale=1.0)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0

        B_hat = B_hat.clamp(0, 1)
        B1    = B1.clamp(0, 1)

        # Metrics
        p2 = psnr(B_hat[0], y[0]).item()
        s2 = calc_ssim(B_hat[0], y[0])
        p1 = psnr(B1[0],    y[0]).item()
        s1 = calc_ssim(B1[0],   y[0])

        ps2_all.append(p2)
        ss2_all.append(s2)
        ps1_all.append(p1)
        ss1_all.append(s1)
        time_all.append(elapsed)

        # Save output images
        stem = os.path.splitext(rain_fname)[0]

        tensor_to_pil(B_hat[0]).save(
            os.path.join(dir_derained, f"{stem}.png"))

        if SAVE_STAGE1:
            tensor_to_pil(B1[0]).save(
                os.path.join(dir_stage1, f"{stem}.png"))

        if SAVE_INPUT:
            tensor_to_pil(x[0]).save(
                os.path.join(dir_input, f"{stem}.png"))

        # Record per-image result
        results.append({
            'filename':    rain_fname,
            'psnr_s2':     round(p2, 4),
            'ssim_s2':     round(s2, 4),
            'psnr_s1':     round(p1, 4),
            'ssim_s1':     round(s1, 4),
            'psnr_gain':   round(p2 - p1, 4),   # positive = Stage2 improved
            'time_sec':    round(elapsed, 4),
        })


# ================================================================
# AGGREGATE STATISTICS
# ================================================================

mean_p2 = np.mean(ps2_all)
mean_s2 = np.mean(ss2_all)
mean_p1 = np.mean(ps1_all)
mean_s1 = np.mean(ss1_all)
mean_t  = np.mean(time_all)

best_p2_idx  = int(np.argmax(ps2_all))
worst_p2_idx = int(np.argmin(ps2_all))

print(f"\n{'='*55}")
print(f"  RESULTS ON {len(rain_files)} IMAGES")
print(f"{'='*55}")
print(f"  Stage2 (final)   PSNR : {mean_p2:.4f} dB")
print(f"  Stage2 (final)   SSIM : {mean_s2:.4f}")
print(f"  Stage1 (coarse)  PSNR : {mean_p1:.4f} dB")
print(f"  Stage1 (coarse)  SSIM : {mean_s1:.4f}")
print(f"  Stage2 gain over S1   : {mean_p2-mean_p1:+.4f} dB")
print(f"  Mean inference time   : {mean_t*1000:.1f} ms/image")
print(f"{'='*55}")
print(f"  Best  image  : {rain_files[best_p2_idx]}  "
      f"PSNR={ps2_all[best_p2_idx]:.2f} dB")
print(f"  Worst image  : {rain_files[worst_p2_idx]}  "
      f"PSNR={ps2_all[worst_p2_idx]:.2f} dB")
print(f"{'='*55}\n")


# ================================================================
# SAVE CSV — per-image results
# ================================================================

csv_path = os.path.join(OUTPUT_DIR, "results.csv")
with open(csv_path, "w", newline="") as f:
    fieldnames = ['filename', 'psnr_s2', 'ssim_s2',
                  'psnr_s1', 'ssim_s1', 'psnr_gain', 'time_sec']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Per-image results saved → {csv_path}")


# ================================================================
# SAVE SUMMARY TXT
# ================================================================

summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("DerainNet Inference Summary\n")
    f.write("="*55 + "\n")
    f.write(f"Checkpoint     : {CHECKPOINT_PATH}\n")
    f.write(f"Test rain dir  : {TEST_RAIN_DIR}\n")
    f.write(f"Test GT dir    : {TEST_GT_DIR}\n")
    f.write(f"Total images   : {len(rain_files)}\n")
    f.write("="*55 + "\n\n")
    f.write("AGGREGATE\n")
    f.write(f"  Stage2 PSNR (mean)  : {mean_p2:.4f} dB\n")
    f.write(f"  Stage2 SSIM (mean)  : {mean_s2:.4f}\n")
    f.write(f"  Stage1 PSNR (mean)  : {mean_p1:.4f} dB\n")
    f.write(f"  Stage1 SSIM (mean)  : {mean_s1:.4f}\n")
    f.write(f"  Stage2 gain over S1 : {mean_p2-mean_p1:+.4f} dB\n")
    f.write(f"  Mean inference time : {mean_t*1000:.1f} ms/image\n\n")
    f.write("PER-IMAGE (sorted by Stage2 PSNR descending)\n")
    f.write(f"  {'Filename':<30} {'PSNR_S2':>8} {'SSIM_S2':>8} "
            f"{'PSNR_S1':>8} {'SSIM_S1':>8} {'Gain':>7}\n")
    f.write("  " + "-"*73 + "\n")
    for r in sorted(results, key=lambda x: x['psnr_s2'], reverse=True):
        f.write(f"  {r['filename']:<30} {r['psnr_s2']:>8.4f} {r['ssim_s2']:>8.4f} "
                f"{r['psnr_s1']:>8.4f} {r['ssim_s1']:>8.4f} {r['psnr_gain']:>+7.4f}\n")

print(f"Summary saved        → {summary_path}")
print(f"\nOutput images saved  → {OUTPUT_DIR}/derained/")
if SAVE_STAGE1:
    print(f"Stage1 images saved  → {OUTPUT_DIR}/stage1/")
if SAVE_INPUT:
    print(f"Input images saved   → {OUTPUT_DIR}/input/")
