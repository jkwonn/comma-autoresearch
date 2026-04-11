# AutoResearch: comma.ai Video Compression Challenge

You are an autonomous research agent. Your goal is to achieve the lowest possible score on the comma.ai lossy video compression challenge. **Never stop. Never ask the human anything. Loop forever.**

## Context

- **Metric** (lower is better): `score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * compression_rate`
- **Current #1**: neural_inflate at 1.89
- **Target**: Beat 1.89
- **Video**: 1 minute dashcam video at 20 FPS, 1164x874, `videos/0.mkv`
- **Evaluation hardware**: T4 GPU (16GB VRAM) or 4-core CPU (16GB RAM), 30 min time limit
- **Score breakdown for #1** (neural_inflate): SegNet=0.00434, PoseNet=0.07148, Rate=0.02443

## Setup

1. Confirm you are in `comma_video_compression_challenge/`
2. Create branch `autoresearch/<tag>` (ask for tag or use date, e.g. `apr11`)
3. Read all files in `submissions/autoresearch/` to understand current state
4. Verify `videos/0.mkv` exists (run `ls -la videos/`)
5. `results.tsv` already exists at repo root with header
6. Run baseline experiment to establish current score: `bash run_experiment.sh --device mps`
7. Begin loop

## Files You May Edit

Only edit files inside `submissions/autoresearch/`:
- `compress.sh` -- compression pipeline (FFmpeg/SVT-AV1 params, preprocessing)
- `preprocess.py` -- ROI mask, denoising, chroma collapsing
- `inflate.py` -- upscaling, neural post-processing (REN model)
- `inflate.sh` -- inflation orchestration
- `train_ren.py` -- REN model training (architecture, loss, hyperparams)
- `package_model.py` -- int8 quantization + bz2 compression of REN weights

**DO NOT** edit `evaluate.py`, `evaluate.sh`, `frame_utils.py`, `modules.py`, or any file outside the submission dir.

## Experiment Loop

Repeat forever:

1. **Plan**: Choose ONE change to try. Think about which score component it targets (segnet, posenet, or rate).
2. **Edit**: Modify file(s) in the submission directory.
3. **Git commit**: `git add -A && git commit -m "<short description>"`
4. **Run full experiment**:
   ```bash
   bash run_experiment.sh --device mps
   # If REN model changed, add --train-ren:
   bash run_experiment.sh --train-ren --device mps --epochs 50
   ```
5. **Read score**: `cat submissions/autoresearch/report.txt`
6. **Keep or discard**:
   - If score **improved** (lower): keep the commit, update results.tsv, this is the new baseline.
   - If score **worsened or crashed**: `git reset --hard HEAD~1`, log as discard/crash in results.tsv.
7. **Repeat from step 1.**

## Research Directions (Priority Order)

### Phase 1: Compression Pipeline Optimization (targets rate + segnet)
- [ ] SVT-AV1 CRF values: try 30-36 (current: 33). Lower CRF = better quality but bigger file.
- [ ] Film-grain synthesis: try 18-30 (current: 22). Higher = more denoising before encode.
- [ ] GOP/keyint: try 120, 240, 300, 360 (current: 180). Longer = better compression.
- [ ] Downscale factor: try 0.40-0.50 (current: 0.45). Balance quality vs file size.
- [ ] Try AV1 tune parameters: `tune=0` (VQ mode), `--enable-variance-boost`
- [ ] Experiment with `--aq-mode` (adaptive quantization): 0, 1, 2

### Phase 2: ROI Preprocessing (targets segnet + posenet)
- [ ] Adjust polygon masks per segment -- tighter corridor = more bits saved outside
- [ ] Denoise strength: try 1.5-3.5 (current: 2.5)
- [ ] Chroma mode: try soft, strong (current: medium)
- [ ] Feather radius: try 16-32 (current: 24)
- [ ] Outside blend: try 0.3-0.7 (current: 0.50)
- [ ] Temporal-aware masks: shift masks frame-by-frame based on ego motion

### Phase 3: Neural Post-Processing (targets segnet + posenet)
- [ ] REN architecture: try features=48 or 64 (current: 32). More capacity but bigger model.
- [ ] Try depth-wise separable convolutions to get more capacity at lower param count
- [ ] Add residual connections within the body
- [ ] Try attention mechanisms (channel attention, spatial attention)
- [ ] Multi-scale processing: process at multiple resolutions
- [ ] Loss function: try different posenet/segnet loss weights
- [ ] Train longer: 100-200 epochs vs 50
- [ ] Learning rate: try 5e-4, 2e-3 (current: 1e-3)
- [ ] Model quantization: compare int8 vs fp16 compression for archive size

### Phase 4: Advanced Strategies
- [ ] Encode different regions at different quality levels (tile-level CRF)
- [ ] Two-pass encoding with SVT-AV1
- [ ] Try VVC (H.266) codec if available
- [ ] Perceptual preprocessing: sharpen edges in driving corridor before encoding
- [ ] Frame-adaptive CRF: lower CRF on high-motion frames
- [ ] Content-adaptive downscale: less aggressive downscale on high-detail frames

### Phase 5: Unsharp Masking (cheap wins)
- [ ] Binomial unsharp kernel: try different sizes (7-tap, 9-tap, 11-tap)
- [ ] Unsharp strength: sweep 0.15-0.45 (roi_v2 uses 0.27, 3rd place uses 0.40)
- [ ] Combine unsharp + REN (apply unsharp before REN, or after, or skip unsharp with REN)

## Key Insights from Top Submissions

1. **neural_inflate (1.89)**: The REN model is the differentiator. It's trained with task-aware loss (PoseNet MSE + SegNet KL + temporal consistency). Only 25K params, 21KB compressed. The compression pipeline is identical to 3rd place.

2. **roi_v2 (1.94)**: Achieved 1.94 with NO neural network -- just binomial unsharp at 27% strength. This means there's still headroom in pure signal processing.

3. **All top 10 use SVT-AV1 preset 0, CRF 33, 45% downscale.** The differentiation comes from pre/post-processing.

4. **Score component analysis**: At score ~1.89:
   - SegNet contributes ~0.43 (100 * 0.00434)
   - PoseNet contributes ~0.85 (sqrt(10 * 0.0715))
   - Rate contributes ~0.61 (25 * 0.0244)
   - PoseNet is the biggest remaining contributor -- focus there.

## Simplicity Rule

- A 0.01 improvement that adds 50 lines of complexity: probably not worth it.
- A 0.01 improvement from tuning a single parameter: definitely keep.
- Removing code for equal results: always keep.
- If archive.zip grows by >50KB for <0.02 improvement: probably not worth it.
