#!/usr/bin/env python
"""
Approach 4: Learnable per-class colormap.

For each of 5 SegNet classes, learn the optimal RGB color.
Paint each frame as flat colors based on segmentation mask.
Result: trivially compressible (5 colors), perfect SegNet score.

Then optimize the colors to also minimize PoseNet distortion.
Archive = compressed mask video + 15 bytes (5 RGB values).
"""
import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import av, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
SEG_H, SEG_W = segnet_model_input_size[1], segnet_model_input_size[0]


def masks_to_rgb(masks, colors):
    """
    Convert class masks to RGB using learned colors.
    masks: (B, H, W) long
    colors: (5, 3) float [0-255]
    Returns: (B, 3, H, W) float [0-255]
    """
    B, H, W = masks.shape
    # Gather colors for each pixel
    flat_masks = masks.view(-1)  # (B*H*W,)
    flat_rgb = colors[flat_masks]  # (B*H*W, 3)
    return flat_rgb.view(B, H, W, 3).permute(0, 3, 1, 2)


def main():
    print(f"Device: {DEVICE}", flush=True)

    # Load eval models
    print("Loading DistortionNet...", flush=True)
    distortion_net = DistortionNet().to(DEVICE).eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in distortion_net.parameters():
        p.requires_grad_(False)
    segnet = distortion_net.segnet
    posenet = distortion_net.posenet

    # Load video and extract masks + GT
    video_path = os.path.join(ROOT, 'videos/0.mkv')
    print(f"Loading frames...", flush=True)
    container = av.open(video_path)
    gt_frames = []
    masks = []
    for frame in container.decode(video=0):
        t = yuv420_to_rgb(frame)
        gt_small = F.interpolate(t.permute(2, 0, 1).unsqueeze(0).float().to(DEVICE),
                                  size=(SEG_H, SEG_W), mode='bilinear')
        gt_frames.append(gt_small.squeeze(0))
        with torch.no_grad():
            mask = segnet(gt_small).argmax(dim=1).squeeze(0)
        masks.append(mask)
    container.close()
    gt_frames = torch.stack(gt_frames)  # (N, 3, H, W)
    masks = torch.stack(masks)  # (N, H, W)
    N = len(masks)
    print(f"  {N} frames, masks extracted", flush=True)

    # Initialize colors with mean RGB per class from GT
    print("\nInitializing per-class colors from GT means...", flush=True)
    colors = torch.zeros(NUM_CLASSES, 3, device=DEVICE)
    for c in range(NUM_CLASSES):
        class_mask = (masks == c).unsqueeze(1).expand_as(gt_frames)  # (N, 3, H, W)
        if class_mask.any():
            colors[c] = gt_frames[class_mask].view(3, -1).mean(dim=1) if class_mask.sum() > 0 else torch.tensor([128., 128., 128.], device=DEVICE)

    # Actually compute per-class means properly
    for c in range(NUM_CLASSES):
        mask_c = (masks == c)  # (N, H, W)
        count = mask_c.sum()
        if count > 0:
            # For each channel, get mean where mask matches
            for ch in range(3):
                colors[c, ch] = gt_frames[:, ch, :, :][mask_c].mean()

    print(f"  Initial colors: {colors.cpu().numpy().round().astype(int)}", flush=True)

    # Make colors optimizable
    colors = colors.requires_grad_(True)
    optimizer = torch.optim.Adam([colors], lr=2.0)

    # Optimize colors to minimize eval metrics
    print(f"\nOptimizing colors for {1000} steps...\n", flush=True)

    # Process in batches of consecutive pairs
    batch_size = 16

    for step in range(1000):
        optimizer.zero_grad()
        total_seg_loss = 0
        total_pose_loss = 0
        n_batches = 0

        # Sample random pairs
        indices = torch.randperm(N - 1)[:batch_size]

        for idx in indices:
            i = idx.item()
            mask1 = masks[i:i+1]   # (1, H, W)
            mask2 = masks[i+1:i+2]
            gt1 = gt_frames[i:i+1]  # (1, 3, H, W)
            gt2 = gt_frames[i+1:i+2]

            # Generate colored frames from masks
            fake1 = masks_to_rgb(mask1, colors.clamp(0, 255))
            fake2 = masks_to_rgb(mask2, colors.clamp(0, 255))

            # SegNet loss
            with torch.no_grad():
                gt_classes = segnet(gt2).argmax(dim=1)
            fake_logits = segnet(fake2)
            loss_seg = F.cross_entropy(fake_logits, gt_classes)

            # PoseNet loss
            fake_pair = torch.stack([fake1, fake2], dim=1)
            gt_pair = torch.stack([gt1, gt2], dim=1)
            fake_pose_in = posenet.preprocess_input(fake_pair)
            with torch.no_grad():
                gt_pose_out = posenet(posenet.preprocess_input(gt_pair))
            fake_pose_out = posenet(fake_pose_in)
            loss_pose = sum(
                F.mse_loss(fake_pose_out[h.name][..., :h.out // 2],
                           gt_pose_out[h.name][..., :h.out // 2])
                for h in posenet.hydra.heads
            )

            total_seg_loss += loss_seg
            total_pose_loss += loss_pose
            n_batches += 1

        loss = (10.0 * total_seg_loss + 1.0 * total_pose_loss) / max(n_batches, 1)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # Full eval on subset
            with torch.no_grad():
                seg_mis_total = 0
                pose_mse_total = 0
                n_eval = 0
                for i in range(0, min(200, N-1), 2):
                    m1, m2 = masks[i:i+1], masks[i+1:i+2]
                    g1, g2 = gt_frames[i:i+1], gt_frames[i+1:i+2]
                    f1 = masks_to_rgb(m1, colors.clamp(0, 255))
                    f2 = masks_to_rgb(m2, colors.clamp(0, 255))

                    seg_mis = (segnet(g2).argmax(1) != segnet(f2).argmax(1)).float().mean()
                    seg_mis_total += seg_mis.item()

                    fp = torch.stack([f1, f2], dim=1)
                    gp = torch.stack([g1, g2], dim=1)
                    gpo = posenet(posenet.preprocess_input(gp))
                    fpo = posenet(posenet.preprocess_input(fp))
                    pmse = sum(F.mse_loss(fpo[h.name][..., :h.out//2], gpo[h.name][..., :h.out//2]) for h in posenet.hydra.heads).item()
                    pose_mse_total += pmse
                    n_eval += 1

                avg_seg = seg_mis_total / max(n_eval, 1)
                avg_pose = pose_mse_total / max(n_eval, 1)
                score = 100 * avg_seg + math.sqrt(10 * avg_pose)

            print(f"  Step {step:4d}: colors={colors.data.clamp(0,255).cpu().numpy().round().astype(int).tolist()} "
                  f"seg_mis={avg_seg:.6f} pose_mse={avg_pose:.6f} score={score:.4f}", flush=True)

    final_colors = colors.data.clamp(0, 255).round().cpu()
    print(f"\n  Final colors: {final_colors.numpy().astype(int).tolist()}")
    torch.save(final_colors, os.path.join(HERE, 'colormap.pt'))
    print(f"  Saved colormap.pt")


if __name__ == '__main__':
    main()
