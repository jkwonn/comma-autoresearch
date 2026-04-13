#!/usr/bin/env python
"""Package REN model weights as int8+bz2 for minimal archive size."""
import os, io, bz2, struct, zipfile, torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def quantize_int8(sd):
    buf = io.BytesIO()
    buf.write(struct.pack('<I', len(sd)))
    for name, tensor in sd.items():
        t = tensor.float().cpu()
        scale = t.abs().max().item() / 127.0 if t.abs().max().item() > 0 else 1.0
        quantized = (t / scale).round().clamp(-127, 127).to(torch.int8)
        data = quantized.numpy().tobytes()
        name_bytes = name.encode('utf-8')
        buf.write(struct.pack('<I', len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack('<I', len(t.shape)))
        for s in t.shape:
            buf.write(struct.pack('<I', s))
        buf.write(struct.pack('<f', scale))
        buf.write(struct.pack('<I', len(data)))
        buf.write(data)
    return buf.getvalue()

def main():
    pt_path = os.path.join(HERE, 'ren_model.pt')
    out_path = os.path.join(HERE, 'ren_model.int8.bz2')
    archive_path = os.path.join(HERE, 'archive', 'ren_model.int8.bz2')

    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found. Run train_ren.py first.")
        return

    sd = torch.load(pt_path, map_location='cpu', weights_only=False)
    raw = quantize_int8(sd)
    compressed = bz2.compress(raw, compresslevel=9)

    for path in [out_path, archive_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(compressed)

    pt_size = os.path.getsize(pt_path)
    out_size = len(compressed)
    print(f"  Original: {pt_size:,} bytes ({pt_size/1024:.0f} KB)")
    print(f"  Packaged: {out_size:,} bytes ({out_size/1024:.1f} KB)")
    print(f"  Ratio: {out_size/pt_size:.1%}")
    print(f"  Saved to: {out_path}")
    print(f"  Copied to: {archive_path}")

    # Re-create archive.zip to include the new model
    zip_path = os.path.join(HERE, 'archive.zip')
    archive_dir = os.path.join(HERE, 'archive')
    if os.path.isdir(archive_dir):
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in os.listdir(archive_dir):
                zf.write(os.path.join(archive_dir, f), f)
        print(f"  Re-created archive.zip ({os.path.getsize(zip_path):,} bytes)")

if __name__ == '__main__':
    main()
