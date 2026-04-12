#!/usr/bin/env python
"""
Fast REN training with pixel loss only (no PoseNet/SegNet).
Much faster on CPU since it skips the large distortion models.

Usage:
  python train_ren_fast.py [--epochs 50] [--batch-size 4] [--lr 1e-3]
"""
import os, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cpu')


class REN(nn.Module):
    def __init__(self, features=32):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


def decode_frames_subsampled(video_path, subsample, target_w=None, target_h=None, lanczos=False):
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % subsample != 0:
            continue
        t = yuv420_to_rgb(frame)
        if target_w and target_h and (t.shape[0] != target_h or t.shape[1] != target_w):
            if lanczos:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                t = torch.from_numpy(np.array(pil))
            else:
                t = F.interpolate(
                    t.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(target_h, target_w), mode='bicubic', align_corners=False
                ).clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
        frames.append(t)
    container.close()
    return frames


class FramePairDataset(Dataset):
    def __init__(self, comp_frames, gt_frames):
        assert len(comp_frames) == len(gt_frames)
        self.comp = comp_frames
        self.gt = gt_frames

    def __len__(self):
        return len(self.comp)

    def __getitem__(self, idx):
        return self.comp[idx].permute(2, 0, 1).float(), self.gt[idx].permute(2, 0, 1).float()


def train(args):
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(1234)
    np.random.seed(1234)

    W, H = camera_size

    archive_path = os.path.join(HERE, 'archive/0.mkv')
    if not os.path.exists(archive_path):
        print("ERROR: No compressed archive found. Run compress.sh first.")
        sys.exit(1)

    SUBSAMPLE = args.subsample
    print(f"Loading compressed frames (subsample={SUBSAMPLE})...", flush=True)
    comp_frames = decode_frames_subsampled(archive_path, SUBSAMPLE, target_w=W, target_h=H, lanczos=True)
    print(f"  {len(comp_frames)} frames", flush=True)

    gt_path = os.path.join(ROOT, 'videos/0.mkv')
    print(f"Loading GT frames (subsample={SUBSAMPLE})...", flush=True)
    gt_frames = decode_frames_subsampled(gt_path, SUBSAMPLE)
    print(f"  {len(gt_frames)} frames", flush=True)

    assert len(comp_frames) == len(gt_frames)

    split = len(comp_frames) * 5 // 6
    train_ds = FramePairDataset(comp_frames[:split], gt_frames[:split])
    val_ds = FramePairDataset(comp_frames[split:], gt_frames[split:])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = REN(features=args.features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    save_path = os.path.join(HERE, 'ren_model.pt')
    best_val = float('inf')

    # Use Charbonnier loss (robust L1) for better edge preservation
    def charbonnier_loss(pred, target, eps=1e-3):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

    print(f"\n  Training for {args.epochs} epochs...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        n_batches = 0

        for comp, gt in train_loader:
            comp = comp.to(DEVICE)
            gt = gt.to(DEVICE)

            optimizer.zero_grad()
            out = model(comp)
            loss = charbonnier_loss(out, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for comp, gt in val_loader:
                    comp = comp.to(DEVICE)
                    gt = gt.to(DEVICE)
                    out = model(comp)
                    val_loss += charbonnier_loss(out, gt).item()
                    n_val += 1

            val_loss /= max(n_val, 1)
            marker = ''
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), save_path)
                marker = '  <- saved'

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}", flush=True)

    size_kb = os.path.getsize(save_path) / 1024
    print(f"\n  Best val_loss: {best_val:.4f}")
    print(f"  Saved: {save_path} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--subsample', type=int, default=12)
    args = parser.parse_args()
    train(args)
