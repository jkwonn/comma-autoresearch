#!/usr/bin/env python
"""
Mask-to-frame generator for video compression challenge.
Approach: Compress segmentation masks, reconstruct frames with a neural generator
that optimizes directly for the evaluation metrics (SegNet argmax + PoseNet MSE).
"""
import os, sys, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5


class SPADENorm(nn.Module):
    """Spatially-Adaptive Normalization conditioned on segmentation mask."""
    def __init__(self, norm_nc, label_nc=NUM_CLASSES):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 64
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        self.beta = nn.Conv2d(nhidden, norm_nc, 3, padding=1)

    def forward(self, x, seg_map):
        # seg_map: (B, NUM_CLASSES, H, W) one-hot
        normalized = self.norm(x)
        seg_map = F.interpolate(seg_map, size=x.shape[2:], mode='nearest')
        shared = self.shared(seg_map)
        gamma = self.gamma(shared)
        beta = self.beta(shared)
        return normalized * (1 + gamma) + beta


class SPADEResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        self.norm1 = SPADENorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.norm2 = SPADENorm(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm_skip = SPADENorm(in_ch) if in_ch != out_ch else None

    def forward(self, x, seg):
        h = F.relu(self.norm1(x, seg))
        h = self.conv1(h)
        h = F.relu(self.norm2(h, seg))
        h = self.conv2(h)
        if self.norm_skip is not None:
            skip = self.skip(F.relu(self.norm_skip(x, seg)))
        else:
            skip = self.skip(x)
        return h + skip


class MaskToFrameGenerator(nn.Module):
    """
    Takes a one-hot segmentation mask and generates an RGB frame.
    Architecture: SPADE-based generator operating at low resolution,
    upsampled to target size.

    Working resolution: 128x96 (1/4 of 512x384)
    Output: 512x384x3 RGB (then upsampled to 1164x874 at eval time)
    """
    def __init__(self, z_dim=64, base_ch=128):
        super().__init__()
        # Initial constant input
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, base_ch * 12 * 16)  # 16x12 starting resolution

        # Progressive upsampling with SPADE normalization
        # 16x12 -> 32x24 -> 64x48 -> 128x96 -> 256x192 -> 512x384
        self.up1 = SPADEResBlock(base_ch, base_ch)       # 16x12
        self.up2 = SPADEResBlock(base_ch, base_ch)       # 32x24
        self.up3 = SPADEResBlock(base_ch, base_ch // 2)  # 64x48
        self.up4 = SPADEResBlock(base_ch // 2, base_ch // 4)  # 128x96
        self.up5 = SPADEResBlock(base_ch // 4, base_ch // 8)  # 256x192
        self.up6 = SPADEResBlock(base_ch // 8, base_ch // 8)  # 512x384

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_ch // 8, 3, 3, padding=1),
            nn.Sigmoid(),  # Output 0-1, multiply by 255 later
        )

    def forward(self, seg_mask, z=None):
        """
        seg_mask: (B, H, W) long tensor with class indices 0-4
        z: optional latent vector (B, z_dim)
        Returns: (B, 3, 384, 512) float tensor in [0, 255]
        """
        B = seg_mask.shape[0]

        # One-hot encode mask
        seg_onehot = F.one_hot(seg_mask.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()

        # Generate or use provided z
        if z is None:
            z = torch.zeros(B, self.z_dim, device=seg_mask.device)

        # Initial features from z
        x = self.fc(z).view(B, -1, 12, 16)

        # Progressive upsampling
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 32x24
        x = self.up1(x, seg_onehot)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 64x48
        x = self.up2(x, seg_onehot)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 128x96
        x = self.up3(x, seg_onehot)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 256x192
        x = self.up4(x, seg_onehot)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # 512x384
        x = self.up5(x, seg_onehot)
        x = self.up6(x, seg_onehot)

        rgb = self.to_rgb(x) * 255.0
        return rgb


def extract_masks(video_path, segnet, device):
    """Extract segmentation masks from a video using SegNet."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    masks = []

    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8 tensor
        # Resize to segnet input size
        x = rgb.permute(2, 0, 1).unsqueeze(0).float().to(device)
        x = F.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')

        with torch.no_grad():
            logits = segnet(x)
            mask = logits.argmax(dim=1).squeeze(0).cpu()  # (384, 512) long
        masks.append(mask)

    container.close()
    return torch.stack(masks)  # (N, 384, 512)


def encode_masks_to_video(masks, output_path):
    """Encode class masks (0-4) as grayscale video."""
    N, H, W = masks.shape
    container = av.open(output_path, mode='w')
    stream = container.add_stream('libx264', rate=20)
    stream.width = W
    stream.height = H
    stream.pix_fmt = 'gray'
    stream.options = {'crf': '0', 'preset': 'veryslow'}  # Lossless

    for i in range(N):
        # Map class 0-4 to grayscale 0, 63, 126, 189, 252
        gray = (masks[i].numpy() * 63).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(gray, format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


class MaskFrameDataset(Dataset):
    """Dataset of (mask_pair, gt_frame_pair) for training the generator."""
    def __init__(self, masks, gt_frames):
        # masks: (N, 384, 512) long
        # gt_frames: (N, H, W, 3) uint8
        assert len(masks) == len(gt_frames)
        self.masks = masks
        self.gt = gt_frames

    def __len__(self):
        return len(self.masks) - 1  # pairs

    def __getitem__(self, idx):
        mask1 = self.masks[idx]
        mask2 = self.masks[idx + 1]
        gt1 = self.gt[idx].permute(2, 0, 1).float()   # (3, H, W)
        gt2 = self.gt[idx + 1].permute(2, 0, 1).float()
        return mask1, mask2, gt1, gt2


def train_generator(args):
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(42)

    W, H = camera_size  # 1164, 874

    # Load SegNet for mask extraction
    print("Loading DistortionNet...", flush=True)
    distortion_net = DistortionNet().to(DEVICE).eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in distortion_net.parameters():
        p.requires_grad_(False)
    segnet = distortion_net.segnet
    posenet = distortion_net.posenet

    # Extract masks from video
    video_path = os.path.join(ROOT, 'videos/0.mkv')
    print(f"Extracting masks from {video_path}...", flush=True)
    masks = extract_masks(video_path, segnet, DEVICE)
    print(f"  {len(masks)} masks extracted", flush=True)

    # Load ground truth frames
    print(f"Loading GT frames...", flush=True)
    container = av.open(video_path)
    gt_frames = []
    for frame in container.decode(video=0):
        t = yuv420_to_rgb(frame)
        gt_frames.append(t)
    container.close()
    gt_frames_tensor = torch.stack(gt_frames)  # (N, H, W, 3)
    print(f"  {len(gt_frames)} frames loaded", flush=True)

    # Resize GT to segnet input size for training
    gt_small = F.interpolate(
        gt_frames_tensor.permute(0, 3, 1, 2).float(),
        size=(segnet_model_input_size[1], segnet_model_input_size[0]),
        mode='bilinear'
    )  # (N, 3, 384, 512)

    # Split
    split = 1000
    train_ds = MaskFrameDataset(masks[:split], gt_frames_tensor[:split])
    val_ds = MaskFrameDataset(masks[split:], gt_frames_tensor[split:])
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create generator
    generator = MaskToFrameGenerator(z_dim=64, base_ch=args.base_ch).to(DEVICE)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f"  Generator: {n_params:,} params", flush=True)

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_path = os.path.join(HERE, 'generator.pt')
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

            # Generate frames from masks
            fake1 = generator(mask1)  # (B, 3, 384, 512)
            fake2 = generator(mask2)

            # SegNet loss: argmax matching
            with torch.no_grad():
                gt1_small = F.interpolate(gt1, size=(384, 512), mode='bilinear')
                gt_logits = segnet(gt1_small)
                gt_classes = gt_logits.argmax(dim=1)  # (B, 384, 512)

            fake_logits = segnet(fake1)
            # Cross-entropy against GT argmax classes
            loss_seg = F.cross_entropy(fake_logits, gt_classes)

            # PoseNet loss: MSE on pose outputs
            # Stack as pairs: (B, 2, 3, H, W) -> preprocess
            fake_pair = torch.stack([fake1, fake2], dim=1)  # (B, 2, 3, 384, 512)
            gt_pair_small = torch.stack([
                F.interpolate(gt1, size=(384, 512), mode='bilinear'),
                F.interpolate(gt2, size=(384, 512), mode='bilinear')
            ], dim=1)

            # PoseNet expects (B, 2, C, H, W) after preprocess
            fake_pose_in = posenet.preprocess_input(fake_pair)
            with torch.no_grad():
                gt_pose_in = posenet.preprocess_input(gt_pair_small)
                gt_pose_out = posenet(gt_pose_in)
            fake_pose_out = posenet(fake_pose_in)

            loss_pose = sum(
                F.mse_loss(fake_pose_out[h.name][..., :h.out // 2],
                           gt_pose_out[h.name][..., :h.out // 2])
                for h in posenet.hydra.heads
            )

            # Also add pixel loss for stability
            loss_pixel = F.l1_loss(fake1, F.interpolate(gt1, size=(384, 512), mode='bilinear'))

            # Combined loss (weighted to match scoring formula)
            # score = 100*segnet + sqrt(10*posenet) + 25*rate
            # segnet has 100x weight, posenet has ~3x weight (sqrt), rate is fixed
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
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for mask1, mask2, gt1, gt2 in val_loader:
                    mask1 = mask1.to(DEVICE)
                    mask2 = mask2.to(DEVICE)
                    gt1 = gt1.to(DEVICE)
                    gt2 = gt2.to(DEVICE)

                    fake1 = generator(mask1)
                    fake2 = generator(mask2)

                    # Compute actual eval metrics
                    gt1_small = F.interpolate(gt1, size=(384, 512), mode='bilinear')
                    gt_logits = segnet(gt1_small)
                    fake_logits = segnet(fake1)
                    seg_mismatch = (gt_logits.argmax(1) != fake_logits.argmax(1)).float().mean()

                    fake_pair = torch.stack([fake1, fake2], dim=1)
                    gt_pair_small = torch.stack([
                        F.interpolate(gt1, size=(384, 512), mode='bilinear'),
                        F.interpolate(gt2, size=(384, 512), mode='bilinear')
                    ], dim=1)
                    fake_pose_in = posenet.preprocess_input(fake_pair)
                    gt_pose_in = posenet.preprocess_input(gt_pair_small)
                    gt_pose_out = posenet(gt_pose_in)
                    fake_pose_out = posenet(fake_pose_in)
                    pose_mse = sum(
                        F.mse_loss(fake_pose_out[h.name][..., :h.out // 2],
                                   gt_pose_out[h.name][..., :h.out // 2])
                        for h in posenet.hydra.heads
                    ).item()

                    val_loss += 100 * seg_mismatch.item() + math.sqrt(10 * pose_mse)
                    n_val += 1

            val_loss /= max(n_val, 1)
            marker = ''
            if val_loss < best_val:
                best_val = val_loss
                torch.save(generator.state_dict(), save_path)
                marker = '  <- saved'

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} (seg={train_seg:.4f} pose={train_pose:.6f})  "
                  f"val_score={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}", flush=True)
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} (seg={train_seg:.4f} pose={train_pose:.6f})", flush=True)

    print(f"\n  Best val_score: {best_val:.4f}")
    print(f"  Saved: {save_path}")

    # Save masks for compression
    mask_path = os.path.join(HERE, 'masks.mkv')
    print(f"  Saving masks to {mask_path}...")
    encode_masks_to_video(masks, mask_path)
    print(f"  Masks saved ({os.path.getsize(mask_path):,} bytes)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--base-ch', type=int, default=128)
    args = parser.parse_args()
    train_generator(args)
