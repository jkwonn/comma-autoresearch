#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up comma autoresearch ==="

# Clone challenges if not already present
if [ ! -d "comma_video_compression_challenge" ]; then
  git clone https://github.com/commaai/comma_video_compression_challenge.git
  cd comma_video_compression_challenge
  git lfs install && git lfs pull
  uv sync --group cpu
  cd ..
fi

if [ ! -d "controls_challenge" ]; then
  git clone https://github.com/commaai/controls_challenge.git
  cd controls_challenge
  pip install -r requirements.txt
  cd ..
fi

echo "=== Setup complete ==="
echo "Controls challenge: ./controls_challenge/"
echo "Video compression: ./comma_video_compression_challenge/"
