#!/usr/bin/env python3
"""Downscale + AV1 encode using PyAV (which bundles libsvtav1)."""
import argparse
import sys
from pathlib import Path

import av
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb


def encode(input_path, output_path, scale, crf, preset, film_grain, keyint):
    in_container = av.open(str(input_path))
    in_stream = in_container.streams.video[0]

    orig_w = in_stream.width
    orig_h = in_stream.height
    target_w = int(orig_w * scale / 2) * 2
    target_h = int(orig_h * scale / 2) * 2

    out_container = av.open(str(output_path), mode='w')
    out_stream = out_container.add_stream('libsvtav1', rate=20)
    out_stream.width = target_w
    out_stream.height = target_h
    out_stream.pix_fmt = 'yuv420p'

    # SVT-AV1 options
    out_stream.options = {
        'preset': str(preset),
        'crf': str(crf),
        'svtav1-params': f'film-grain={film_grain}:keyint={keyint}:scd=0',
    }

    n = 0
    for frame in in_container.decode(in_stream):
        rgb = yuv420_to_rgb(frame)
        h, w = rgb.shape[0], rgb.shape[1]

        if h != target_h or w != target_w:
            pil = Image.fromarray(rgb.numpy())
            pil = pil.resize((target_w, target_h), Image.LANCZOS)
            rgb_np = np.array(pil)
        else:
            rgb_np = rgb.numpy()

        out_frame = av.VideoFrame.from_ndarray(rgb_np, format='rgb24')
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)
        n += 1

    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    in_container.close()
    print(f"  Encoded {n} frames -> {output_path} ({target_w}x{target_h})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=0.45)
    parser.add_argument('--crf', type=int, default=33)
    parser.add_argument('--preset', type=int, default=0)
    parser.add_argument('--film-grain', type=int, default=22)
    parser.add_argument('--keyint', type=int, default=180)
    args = parser.parse_args()
    encode(args.input, args.output, args.scale, args.crf, args.preset, args.film_grain, args.keyint)


if __name__ == '__main__':
    main()
