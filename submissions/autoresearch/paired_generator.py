#!/usr/bin/env python
"""
Paired mask-to-frame generator for video compression challenge.
Key insight: PoseNet evaluates PAIRS of frames for temporal consistency.
This generator takes TWO masks and outputs TWO RGB frames jointly,
ensuring temporal coherence.
"""
import os, sys, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
SEG_H, SEG_W = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.conv(x)


class PairGenerator(nn.Module):
    """
    Takes two segmentation masks (each 384x512, 5 classes) and generates
    two RGB frames (each 3x384x512) with temporal consistency.

    Architecture: Shared encoder for masks -> cross-attention for temporal
    consistency -> independent decoders for each frame.
    """
    def __init__(self, base_ch=64):
        super().__init__()
        # Mask encoder (shared): one-hot mask -> features
        self.mask_enc = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, base_ch, 7, stride=2, padding=3),  # 192x256
            nn.ReLU(inplace=True),
            ResBlock(base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),  # 96x128
            nn.ReLU(inplace=True),
            ResBlock(base_ch * 2),
        )

        # Cross-frame attention: exchange information between frame features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=base_ch * 2, num_heads=4, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(base_ch * 2)

        # Shared body with residual blocks
        self.body = nn.Sequential(
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2),
            ResBlock(base_ch * 2),
        )

        # Decoder: upsample back to 384x512
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),  # 192x256
            nn.ReLU(inplace=True),
            ResBlock(base_ch),
            nn.ConvTranspose2d(base_ch, base_ch // 2, 4, stride=2, padding=1),  # 384x512
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode_mask(self, mask):
        """mask: (B, H, W) long -> features (B, C, H/4, W/4)"""
        onehot = F.one_hot(mask.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
        return self.mask_enc(onehot)

    def cross_attend(self, feat1, feat2):
        """Exchange temporal information between frame features."""
        B, C, H, W = feat1.shape
        # Flatten spatial dims for attention
        f1_flat = feat1.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        f2_flat = feat2.flatten(2).permute(0, 2, 1)

        # Cross-attention: frame1 attends to frame2 and vice versa
        f1_attn, _ = self.cross_attn(f1_flat, f2_flat, f2_flat)
        f2_attn, _ = self.cross_attn(f2_flat, f1_flat, f1_flat)

        f1_out = self.cross_norm(f1_flat + f1_attn).permute(0, 2, 1).view(B, C, H, W)
        f2_out = self.cross_norm(f2_flat + f2_attn).permute(0, 2, 1).view(B, C, H, W)
        return f1_out, f2_out

    def forward(self, mask1, mask2):
        """
        mask1, mask2: (B, 384, 512) long tensors (class indices 0-4)
        Returns: rgb1, rgb2: (B, 3, 384, 512) float tensors in [0, 255]
        """
        # Encode both masks with shared encoder
        feat1 = self.encode_mask(mask1)
        feat2 = self.encode_mask(mask2)

        # Cross-attention for temporal consistency
        feat1, feat2 = self.cross_attend(feat1, feat2)

        # Shared body
        feat1 = self.body(feat1)
        feat2 = self.body(feat2)

        # Decode to RGB
        rgb1 = self.decoder(feat1) * 255.0
        rgb2 = self.decoder(feat2) * 255.0
        return rgb1, rgb2


def extract_masks(video_path, segnet, device):
    """Extract segmentation masks from video using SegNet."""
    container = av.open(video_path)
    masks = []
    for frame in container.decode(video=0):
        rgb = yuv420_to_rgb(frame)
        x = rgb.permute(2, 0, 1).unsqueeze(0).float().to(device)
        x = F.interpolate(x, size=(SEG_H, SEG_W), mode='bilinear')
        with torch.no_grad():
            mask = segnet(x).argmax(dim=1).squeeze(0).cpu()
        masks.append(mask)
    container.close()
    return torch.stack(masks)


class PairDataset(Dataset):
    def __init__(self, masks, gt_frames):
        assert len(masks) == len(gt_frames)
        self.masks = masks
        self.gt = gt_frames

    def __len__(self):
        return len(self.masks) - 1

    def __getitem__(self, idx):
        return (self.masks[idx], self.masks[idx + 1],
                self.gt[idx].permute(2, 0, 1).float(),
                self.gt[idx + 1].permute(2, 0, 1).float())


def train(args):
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(42)

    # Load evaluation models
    print("Loading DistortionNet...", flush=True)
    distortion_net = DistortionNet().to(DEVICE).eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in distortion_net.parameters():
        p.requires_grad_(False)
    segnet = distortion_net.segnet
    posenet = distortion_net.posenet

    # Extract masks
    video_path = os.path.join(ROOT, 'videos/0.mkv')
    print(f"Extracting masks...", flush=True)
    masks = extract_masks(video_path, segnet, DEVICE)
    print(f"  {len(masks)} masks", flush=True)

    # Load GT frames
    print(f"Loading GT frames...", flush=True)
    container = av.open(video_path)
    gt_frames = [yuv420_to_rgb(f) for f in container.decode(video=0)]
    container.close()
    gt_frames = torch.stack(gt_frames)
    print(f"  {len(gt_frames)} frames", flush=True)

    # Split
    split = 1000
    train_ds = PairDataset(masks[:split], gt_frames[:split])
    val_ds = PairDataset(masks[split:], gt_frames[split:])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create paired generator
    generator = PairGenerator(base_ch=args.base_ch).to(DEVICE)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f"  Generator: {n_params:,} params", flush=True)

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_path = os.path.join(HERE, 'pair_generator.pt')
    best_val = float('inf')

    print(f"\n  Training for {args.epochs} epochs...\n", flush=True)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        train_loss = 0
        train_seg = 0
        train_pose = 0
        n_batches = 0

        for mask1, mask2, gt1, gt2 in train_loader:
            mask1 = mask1.to(DEVICE)
            mask2 = mask2.to(DEVICE)
            gt1 = gt1.to(DEVICE)
            gt2 = gt2.to(DEVICE)

            # Generate frame pair
            fake1, fake2 = generator(mask1, mask2)

            # SegNet loss on BOTH frames (eval uses last frame, but training on both helps)
            gt1_small = F.interpolate(gt1, size=(SEG_H, SEG_W), mode='bilinear')
            gt2_small = F.interpolate(gt2, size=(SEG_H, SEG_W), mode='bilinear')
            with torch.no_grad():
                gt_classes1 = segnet(gt1_small).argmax(dim=1)
                gt_classes2 = segnet(gt2_small).argmax(dim=1)

            fake_logits1 = segnet(fake1)
            fake_logits2 = segnet(fake2)
            loss_seg = (F.cross_entropy(fake_logits1, gt_classes1) +
                        F.cross_entropy(fake_logits2, gt_classes2)) / 2

            # PoseNet loss on the PAIR (this is what eval does)
            fake_pair = torch.stack([fake1, fake2], dim=1)
            gt_pair = torch.stack([gt1_small, gt2_small], dim=1)

            fake_pose_in = posenet.preprocess_input(fake_pair)
            with torch.no_grad():
                gt_pose_in = posenet.preprocess_input(gt_pair)
                gt_pose_out = posenet(gt_pose_in)
            fake_pose_out = posenet(fake_pose_in)

            loss_pose = sum(
                F.mse_loss(fake_pose_out[h.name][..., :h.out // 2],
                           gt_pose_out[h.name][..., :h.out // 2])
                for h in posenet.hydra.heads
            )

            # Pixel loss for stability
            loss_pixel = (F.l1_loss(fake1, gt1_small) + F.l1_loss(fake2, gt2_small)) / 2

            # Combined: heavy SegNet weight (100x in scoring) + PoseNet + pixel
            loss = 10.0 * loss_seg + 1.0 * loss_pose + 0.1 * loss_pixel

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_seg += loss_seg.item()
            train_pose += loss_pose.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)
        train_seg /= max(n_batches, 1)
        train_pose /= max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            generator.eval()
            val_seg_total = 0
            val_pose_total = 0
            n_val = 0
            with torch.no_grad():
                for mask1, mask2, gt1, gt2 in val_loader:
                    mask1 = mask1.to(DEVICE)
                    mask2 = mask2.to(DEVICE)
                    gt1 = gt1.to(DEVICE)
                    gt2 = gt2.to(DEVICE)

                    fake1, fake2 = generator(mask1, mask2)

                    # SegNet: argmax mismatch (actual eval metric)
                    gt2_small = F.interpolate(gt2, size=(SEG_H, SEG_W), mode='bilinear')
                    seg_mismatch = (segnet(gt2_small).argmax(1) != segnet(fake2).argmax(1)).float().mean()

                    # PoseNet: MSE on pose (actual eval metric)
                    gt_pair = torch.stack([
                        F.interpolate(gt1, size=(SEG_H, SEG_W), mode='bilinear'),
                        gt2_small
                    ], dim=1)
                    fake_pair = torch.stack([fake1, fake2], dim=1)

                    gt_pose_out = posenet(posenet.preprocess_input(gt_pair))
                    fake_pose_out = posenet(posenet.preprocess_input(fake_pair))
                    pose_mse = sum(
                        F.mse_loss(fake_pose_out[h.name][..., :h.out // 2],
                                   gt_pose_out[h.name][..., :h.out // 2])
                        for h in posenet.hydra.heads
                    ).item()

                    val_seg_total += seg_mismatch.item()
                    val_pose_total += pose_mse
                    n_val += 1

            val_seg = val_seg_total / max(n_val, 1)
            val_pose = val_pose_total / max(n_val, 1)
            val_score = 100 * val_seg + math.sqrt(10 * val_pose)

            marker = ''
            if val_score < best_val:
                best_val = val_score
                torch.save(generator.state_dict(), save_path)
                marker = '  <- saved'

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f} (seg={train_seg:.4f} pose={train_pose:.6f})  "
                  f"val: seg_mis={val_seg:.6f} pose_mse={val_pose:.6f} score={val_score:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.4f} (seg={train_seg:.4f} pose={train_pose:.6f})", flush=True)

    print(f"\n  Best val_score: {best_val:.4f}")
    print(f"  Saved: {save_path} ({os.path.getsize(save_path)/1024:.0f} KB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--base-ch', type=int, default=64)
    args = parser.parse_args()
    train(args)
