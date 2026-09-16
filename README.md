# Wan 2.2 Batch Pipeline — Phase 1 (hook-testing) — RTX 5090 32GB target

5–10s product/supplement demo clips on Vast.ai host #553680 (machine 148413, VERIFIED):
1x RTX 5090 32GB (109.1 TFLOPS, 1455.8 GB/s) + EPYC 7K62, 64GB RAM, NVMe.
Every clip gets a UID baked into its filename + a manifest row for the affiliate tracking system.

## Stack (5090-specific — do not downgrade)
- `torch==2.7.1` / `torchvision==0.22.1` from cu128 index (CUDA 12.8+). The 5090's
  sm_120 needs cu128 kernels — torch 2.4.1/cu121 WILL FAIL on this GPU.
- `diffusers==0.35.2` exact (v0.35.0 added Wan 2.2; 0.35.2 has the Wan
  scale_shift_factor fix). Don't float to 0.36+/0.40.x untested.
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (unified T2V+I2V, Apache-2.0).
  Full-VRAM on 5090 32GB (no tiling, no offload default).
  VAE tiling DISABLED: diffusers#12529 / Wan2.2#125 crash tiled decode
  ("tensor a (2) vs b (4)" at avg_shortcut).

## Files
- `config.toml` — profiles: `eco` (480p 5s, ~2-3min, ~$0.02), `standard`
  (720p 5s, ~4-6min, ~$0.04), `long8s`/`long10s` (480p only). Rate $0.40/hr.
- `prompts.example.csv` — hook × angle × SKU matrix template (6 rows).
- `batch.py` — unattended runner: CSV → clips + `manifest.csv` (UID, sha256, cost, UTM).
- `generate.py` — single-clip inference (bf16 full-VRAM, tiling/slicing OFF, offload opt-in via --offload).
- `utils.py` — UID filename convention `{sku}__{hook}__{angle}__{uid}.mp4`.
- `setup.sh` — Vast.ai 5090 bootstrap incl. stack verification (torch cu128, sm_120, pin).
- `Dockerfile` — optional baked image on `pytorch:2.7.1-cuda12.8-cudnn9-runtime`.
- `RUNBOOK.md` — full before/during/after procedure.

## Vast.ai run (RTX 5090, CUDA 12.8 / torch 2.7 template, disk 60GB+, /workspace volume)
```bash

Note: Replace `{YOUR_ORG}` with your GitHub repo and `{YOUR_BASE_URL}` with your affiliate shop URL (required for `manifest.csv` UTM columns).
git clone <your-repo> && cd wan22-pipeline
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
bash setup.sh --profile eco --csv prompts.example.csv   # verifies stack, dry-runs matrix
source .venv/bin/activate
python batch.py --csv prompts.example.csv --out outputs --profile eco --limit 2   # smoke (~5-8 min)
python batch.py --csv prompts.example.csv --out outputs --profile eco              # full matrix
python batch.py --csv prompts.example.csv --out outputs --profile standard --resume # resume/heroes
```

## Cost guide (host #553680, 5090 @ $0.40/hr, full-VRAM no-tiling baseline 180s)
| profile | size | est. min/clip | est. $/clip |
|---|---|---|---|
| eco | 480p 5s | ~1.3 | ~$0.008 |
| standard | 720p 5s | ~3.0 | ~$0.020 |
| long8s | 480p 8s | ~2.0 | ~$0.014 |

## Manifest → affiliate tracking
`outputs/manifest.csv` columns: `uid, filename, sku, hook_id, angle, seed, ... sha256, utm_campaign, utm_content, utm_url`.
Pass your own URL via `--base-url` to auto-fill `utm_url` per clip.
Filename UID == `utm_content` == manifest `uid`: one key joins video file ↔ ad post ↔ sale.

**Note:** Replace `{YOUR_BASE_URL}` with your affiliate shop URL and `{YOUR_CAMPAIGN}` with your campaign name before running. You'll need to provide `--base-url` at render time and optionally `--campaign` and `--utm-source` args in prompts.csv for full UTM attribution.
