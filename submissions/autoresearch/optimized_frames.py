#!/usr/bin/env python
"""
Approach 3: Per-pixel optimized frames.

Instead of training a generator, directly optimize pixel values to minimize
the evaluation metrics. For each frame, run gradient descent on the pixels
to minimize SegNet argmax mismatch and PoseNet MSE.

Then compress the optimized frames with AV1 (they'll be smoother/simpler
than real video, so they compress better).

This is the "adversarial example" approach - find the simplest possible
images that score perfectly on SegNet and PoseNet.
"""
import os, sys, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import av, numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEG_H, SEG_W = segnet_model_input_size[1], segnet_model_input_size[0]


def optimize_frame_pair(frame1_gt, frame2_gt, segnet, posenet,
                         num_steps=500, lr=0.5, smooth_weight=0.01):
    """
    Optimize pixel values of a frame pair to minimize eval metrics.
    Start from GT frames and optimize to reduce SegNet/PoseNet distortion
    while making frames smoother (more compressible).

    frame1_gt, frame2_gt: (3, H, W) float tensors [0-255]
    Returns: optimized (3, SEG_H, SEG_W) tensors [0-255]
    """
    # Resize GT to working resolution
    gt1 = F.interpolate(frame1_gt.unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear')
    gt2 = F.interpolate(frame2_gt.unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear')

    # Get GT targets
    with torch.no_grad():
        gt_seg_logits = segnet(gt2)
        gt_classes = gt_seg_logits.argmax(dim=1)  # (1, H, W)

        gt_pair = torch.stack([gt1, gt2], dim=1)  # (1, 2, 3, H, W)
        gt_pose_in = posenet.preprocess_input(gt_pair)
        gt_pose_out = posenet(gt_pose_in)

    # Initialize optimizable frames from GT
    opt1 = gt1.clone().requires_grad_(True)
    opt2 = gt2.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([opt1, opt2], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Clamp to valid range
        f1 = opt1.clamp(0, 255)
        f2 = opt2.clamp(0, 255)

        # SegNet loss (on frame2, matching eval)
        seg_logits = segnet(f2)
        loss_seg = F.cross_entropy(seg_logits, gt_classes)

        # PoseNet loss (on pair)
        pair = torch.stack([f1, f2], dim=1)
        pose_in = posenet.preprocess_input(pair)
        pose_out = posenet(pose_in)
        loss_pose = sum(
            F.mse_loss(pose_out[h.name][..., :h.out // 2],
                       gt_pose_out[h.name][..., :h.out // 2])
            for h in posenet.hydra.heads
        )

        # Smoothness loss (total variation) - makes frames more compressible
        tv1 = torch.mean(torch.abs(f1[:, :, :, 1:] - f1[:, :, :, :-1])) + \
              torch.mean(torch.abs(f1[:, :, 1:, :] - f1[:, :, :-1, :]))
        tv2 = torch.mean(torch.abs(f2[:, :, :, 1:] - f2[:, :, :, :-1])) + \
              torch.mean(torch.abs(f2[:, :, 1:, :] - f2[:, :, :-1, :]))
        loss_smooth = (tv1 + tv2) / 2

        loss = 10.0 * loss_seg + 1.0 * loss_pose + smooth_weight * loss_smooth
        loss.backward()
        optimizer.step()

    return opt1.detach().clamp(0, 255).squeeze(0), opt2.detach().clamp(0, 255).squeeze(0)


def optimize_all_frames(args):
    print(f"Device: {DEVICE}", flush=True)

    # Load eval models
    print("Loading DistortionNet...", flush=True)
    distortion_net = DistortionNet().to(DEVICE).eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in distortion_net.parameters():
        p.requires_grad_(False)
    segnet = distortion_net.segnet
    posenet = distortion_net.posenet

    # Load video
    video_path = os.path.join(ROOT, 'videos/0.mkv')
    print(f"Loading frames from {video_path}...", flush=True)
    container = av.open(video_path)
    gt_frames = []
    for frame in container.decode(video=0):
        t = yuv420_to_rgb(frame)
        gt_frames.append(t.permute(2, 0, 1).float().to(DEVICE))
    container.close()
    print(f"  {len(gt_frames)} frames loaded", flush=True)

    # Optimize frame pairs
    optimized = []
    n_frames = len(gt_frames)

    print(f"\n  Optimizing {n_frames} frames (steps={args.steps}, lr={args.lr})...\n", flush=True)

    for i in range(0, n_frames - 1, 2):
        f1_opt, f2_opt = optimize_frame_pair(
            gt_frames[i], gt_frames[i + 1],
            segnet, posenet,
            num_steps=args.steps, lr=args.lr,
            smooth_weight=args.smooth_weight
        )
        optimized.extend([f1_opt, f2_opt])

        if (i // 2) % 10 == 0:
            # Quick eval on this pair
            with torch.no_grad():
                gt2_small = F.interpolate(gt_frames[i+1].unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear')
                seg_mis = (segnet(gt2_small).argmax(1) != segnet(f2_opt.unsqueeze(0)).argmax(1)).float().mean().item()

                gt_pair = torch.stack([
                    F.interpolate(gt_frames[i].unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear'),
                    gt2_small
                ], dim=1)
                opt_pair = torch.stack([f1_opt.unsqueeze(0), f2_opt.unsqueeze(0)], dim=1)
                gt_pose = posenet(posenet.preprocess_input(gt_pair))
                opt_pose = posenet(posenet.preprocess_input(opt_pair))
                pose_mse = sum(
                    F.mse_loss(opt_pose[h.name][..., :h.out // 2], gt_pose[h.name][..., :h.out // 2])
                    for h in posenet.hydra.heads
                ).item()

            print(f"  Pair {i//2}/{n_frames//2}: seg_mis={seg_mis:.6f} pose_mse={pose_mse:.6f}", flush=True)

    # Handle last frame if odd
    if n_frames % 2 == 1:
        optimized.append(F.interpolate(gt_frames[-1].unsqueeze(0), size=(SEG_H, SEG_W), mode='bilinear').squeeze(0))

    # Save optimized frames as video
    out_path = os.path.join(HERE, 'optimized_frames.mkv')
    print(f"\n  Saving to {out_path}...", flush=True)

    out_container = av.open(out_path, mode='w')
    out_stream = out_container.add_stream('ffv1', rate=20)
    out_stream.width = SEG_W
    out_stream.height = SEG_H
    out_stream.pix_fmt = 'yuv420p'

    for frame_tensor in optimized:
        rgb = frame_tensor.clamp(0, 255).permute(1, 2, 0).round().cpu().to(torch.uint8).numpy()
        video_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        for packet in out_stream.encode(video_frame):
            out_container.mux(packet)

    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()

    print(f"  Saved {len(optimized)} frames ({os.path.getsize(out_path):,} bytes)")
    print(f"  Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200, help='Optimization steps per pair')
    parser.add_argument('--lr', type=float, default=1.0, help='Pixel optimization learning rate')
    parser.add_argument('--smooth-weight', type=float, default=0.01, help='TV smoothness weight')
    args = parser.parse_args()
    optimize_all_frames(args)
