#!/usr/bin/env python
import av, torch, numpy as np, os, io, bz2, struct
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from frame_utils import camera_size, yuv420_to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# 9-tap binomial unsharp kernel (Pascal row 8 / 65536)
_r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)
STRENGTH = 0.35


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

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


def dequantize_int8(data):
    buf = io.BytesIO(data)
    n_params = struct.unpack('<I', buf.read(4))[0]
    sd = {}
    for _ in range(n_params):
        name_len = struct.unpack('<I', buf.read(4))[0]
        name = buf.read(name_len).decode('utf-8')
        n_dims = struct.unpack('<I', buf.read(4))[0]
        shape = tuple(struct.unpack('<I', buf.read(4))[0] for _ in range(n_dims))
        scale = struct.unpack('<f', buf.read(4))[0]
        data_len = struct.unpack('<I', buf.read(4))[0]
        raw = buf.read(data_len)
        arr = np.frombuffer(raw, dtype=np.int8).reshape(shape)
        sd[name] = torch.from_numpy(arr.astype(np.float32)) * scale
    return sd


def load_ren(archive_dir):
    """Try to load REN model from archive directory."""
    model_path = os.path.join(archive_dir, 'ren_model.int8.bz2')
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        compressed = f.read()
    raw = bz2.decompress(compressed)
    sd = dequantize_int8(raw)
    model = REN(features=32)
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    return model


# REN model disabled until properly trained (epoch 1 model hurts quality)
_ren_model = None


def decode_and_resize_to_file(video_path: str, dst: str):
  target_w, target_h = camera_size
  container = av.open(video_path)
  stream = container.streams.video[0]
  n = 0
  with open(dst, 'wb') as f:
    for frame in container.decode(stream):
      t = yuv420_to_rgb(frame)
      H, W, _ = t.shape
      if H != target_h or W != target_w:
        pil = Image.fromarray(t.numpy())
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        if _ren_model is not None:
          with torch.no_grad():
            x = _ren_model(x)
        else:
          # Fallback: unsharp masking
          blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL, padding=0, groups=3)
          x = x + STRENGTH * (x - blur)

        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
      f.write(t.contiguous().numpy().tobytes())
      n += 1
  container.close()
  return n


if __name__ == "__main__":
  import sys
  src, dst = sys.argv[1], sys.argv[2]
  n = decode_and_resize_to_file(src, dst)
  print(f"saved {n} frames")
