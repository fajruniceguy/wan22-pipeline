# Wan 2.2 Batch Pipeline — Phase 1 (hook-testing) — RTX 5090 32GB target

5–10s product/supplement demo clips on Vast.ai host #553680 (machine 148413, VERIFIED):
1x RTX 5090 32GB (109.1 TFLOPS, 1455.8 GB/s) + EPYC 7K62, 64GB RAM, NVMe.
Every clip gets a UID baked into its filename + a manifest row for the affiliate tracking system.

## Stack (5090-specific — do not downgrade)
- `torch==2.7.1` / `torchvision==0.22.1` from cu128 index (CUDA 12.8+). The 5090's
  sm_120 needs cu128 kernels — torch 2.4.1/cu121 WILL FAIL on this GPU.
- `diffusers==0.35.2` exact (v0.35.0 added Wan 2.2; 0.35.2 has the Wan
  scale_shift_factor fix). Don't float to 0.36+/0.40.x untested.
- `ftfy` is a required runtime dependency of diffusers' Wan pipeline (used by the T5
  prompt cleaner). Not pulled in automatically — it is pinned in requirements.txt.
- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (unified T2V+I2V, Apache-2.0).
  Full-VRAM on 5090 32GB (no tiling, no offload default).
  VAE tiling DISABLED: diffusers#12529 / Wan2.2#125 crash tiled decode
  ("tensor a (2) vs b (4)" at avg_shortcut).

## Files
- `config.toml` — profiles: `eco` (480p 5s), `standard` (720p 5s, 1280x704),
  `long8s`/`long10s` (480p only). Rate $0.40/hr.
- `prompts.example.csv` — hook × angle × SKU matrix template (6 rows).
- `batch.py` — unattended runner: CSV → clips + `manifest.csv` (UID, sha256, cost, UTM).
  Selects `WanImageToVideoPipeline` when any row has an `image`, else `WanPipeline`.
- `generate.py` — single-clip inference (bf16 full-VRAM, tiling/slicing OFF, offload opt-in via --offload).
  Resizes any I2V input image to the profile's exact width×height before inference.
- `utils.py` — UID filename convention `{sku}__{hook}__{angle}__{uid}.mp4`.
- `setup.sh` — Vast.ai 5090 bootstrap incl. stack verification (torch cu128, sm_120, pin).
- `Dockerfile` — optional baked image on `pytorch:2.7.1-cuda12.8-cudnn9-runtime`.
- `RUNBOOK.md` — full before/during/after procedure.

## Vast.ai run (RTX 5090, CUDA 12.8 / torch 2.7 template, disk 60GB+, /workspace volume)

Note: Replace `{YOUR_ORG}` with your GitHub repo and `{YOUR_BASE_URL}` with your affiliate shop URL (required for `manifest.csv` UTM columns).

```bash
git clone <your-repo> && cd wan22-pipeline
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
bash setup.sh --profile eco --csv prompts.example.csv   # verifies stack, dry-runs matrix
source .venv/bin/activate
python batch.py --csv prompts.example.csv --out outputs --profile eco --limit 2   # smoke (~2.5 min)
python batch.py --csv prompts.example.csv --out outputs --profile eco              # full matrix
python batch.py --csv prompts.example.csv --out outputs --profile standard --resume # resume/heroes
```

## Cost guide — MEASURED on host #553680 (5090 @ $0.40/hr, full-VRAM, no tiling, offload OFF)
| profile | size | min/clip | $/clip | status |
|---|---|---|---|---|
| eco | 480p 5s (121f, 30 steps) | **1.18 (71s)** | **~$0.0079** | measured, 12+ consecutive runs |
| standard | 720p 5s (1280x704, 40 steps) | ~3.0 | ~$0.020 | estimated — T2V only, see below |
| long8s | 480p 8s | ~2.0 | ~$0.014 | estimated, untested |

eco timing is **71 seconds flat** with no meaningful variance across 12+ consecutive
inferences — treat it as a fixed constant for batch planning, not a range.
Batch of 8 eco clips ≈ 9.5 min ≈ $0.06.

## Known issues / gotchas
- **ftfy missing** → `NameError: name 'ftfy' is not defined` inside diffusers'
  `prompt_clean`. Install it; it's in requirements.txt now.
- **VAE tiling must stay disabled.** Enabling it triggers diffusers#12529 /
  Wan2.2#125: `RuntimeError: The size of tensor a (2) must match the size of
  tensor b (4)` at `avg_shortcut` during tiled decode.
- **I2V at 720p OOMs on 32GB** with offload disabled (image latents add VRAM on top
  of T2V). eco profile works for I2V. T2V at 720p is fine. Use `--offload` or a
  smaller profile for 720p I2V.
- **I2V input images must match the profile's exact dimensions.** The pipeline
  derives latent sequence length from the image; an aspect mismatch gives
  `size of tensor a (27280) must match tensor b (28520)`. generate.py now resizes
  every input image to width×height before inference.
- **Wan2.2 maxes out at 720P** — specifically `1280x704` / `704x1280`. Dimensions
  must be divisible by 32 (720 is not). There is no 1080p variant of this model.
- **Export bitrate**: `export_to_video()` defaults to quality=5 (~343 kbps, visibly
  over-compressed). Pinned to `quality=10`.
- **First run on a fresh instance** downloads ~12GB of weights (~7 min) and incurs
  Vast.ai **bandwidth charges billed separately** from the hourly GPU rate — the
  first hour can bill noticeably above $0.40. Point `HF_HOME` at a persistent
  volume to avoid repeating it.
- **Transient NVML errors** (`Failed to initialize NVML: Unknown Error`) on a fresh
  container usually clear on retry or a container reboot; they are not a code fault.

## Content notes (from test batches)
- Prompts that pin the subject ("product stays perfectly still") produce exactly that:
  a static subject with camera motion only. For anything livelier, the input image
  needs animatable content (liquid, steam, environment) and the prompt should describe
  in-scene motion, not suppress it.
- Hand/object interaction and faces are the weakest areas of the model. Expect a lower
  keep-rate on those variants and budget more generations per usable clip.

## Manifest → affiliate tracking
`outputs/manifest.csv` columns: `uid, filename, sku, hook_id, angle, seed, ... sha256, utm_campaign, utm_content, utm_url`.
Pass your own URL via `--base-url` to auto-fill `utm_url` per clip.
Filename UID == `utm_content` == manifest `uid`: one key joins video file ↔ ad post ↔ sale.

**Note:** Replace `{YOUR_BASE_URL}` with your affiliate shop URL and `{YOUR_CAMPAIGN}` with your campaign name before running. You'll need to provide `--base-url` at render time and optionally `--campaign` and `--utm-source` args in prompts.csv for full UTM attribution.
