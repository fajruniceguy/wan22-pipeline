# Wan 2.2 Pipeline — RTX 5090 Runbook (Phase 1)

Host: Vast.ai #553680 (machine 148413, VERIFIED) — 1x RTX 5090 32GB
(Blackwell sm_120, 109.1 TFLOPS, 1455.8 GB/s) on H12SSL-i, PCIe 4.0 x16
(measured 26.9 GB/s), AMD EPYC 7K62 48-Core (24 vCPU allocated), 64/258 GB RAM,
SAMSUNG MZQLB3T8HALS-0003 NVMe. $0.40/hr target.

## 0. pins — ANSWER
- `torch==2.7.1` + `torchvision==0.22.1` from cu128 index (CUDA 12.8+).
  5090 sm_120 has NO kernels in torch 2.4.1/cu121 — old pin WILL FAIL.
- `diffusers==0.35.2` exact: v0.35.0 added Wan 2.2; 0.35.2 has the Wan fix.
  Do NOT float to 0.36+/0.40.x untested (0.37 Modular refactor risk).

## 1. WHAT WAS BUILT (`wan22-pipeline/`)
- `requirements.txt` — torch==2.7.1, torchvision==0.22.1 (cu128),
  diffusers==0.35.2 exact, transformers/accelerate floors, pillow, imageio
  (+ffmpeg), tqdm, pandas, tomli for py3.10.
- `config.toml` — `eco` 832x480 121f/5s 30 steps (~$0.02); `standard`
  1280x720 121f 40 steps (~$0.04); `long8s` 193f / `long10s` 241f 480p-only.
  model=Wan-AI/Wan2.2-TI2V-5B-Diffusers, rate $0.40/hr.
- `utils.py` — `{sku}__{hook}__{angle}__{uid}.mp4` filenames; uid/date+rand;
  UTM builder; sha256.
- `generate.py` — bf16 full-VRAM (tiling/slicing OFF per diffusers#12529, offload opt-in); T2V or I2V
  if `image` set; seeded; mp4 via export_to_video. Raises so batch logs+continues.
- `batch.py` — CSV -> clips + manifest.csv (uid, file, sku/hook/angle, seed,
  dims, steps, profile, model, status, error, secs, cost, sha256, utm_*).
  Flags: --limit --dry-run --resume --manifest --base-url --model --offload.
  Pandas-optional, tomllib-optional fallbacks included.
- `prompts.example.csv` — 6-row hook x angle x SKU template.
- `setup.sh` — 5090 bootstrap: venv, stack verify (cu128 + sm_120 + pin), dry-run.
- `Dockerfile` — optional baked image (pytorch 2.7.1-cuda12.8-runtime + ffmpeg).
- `README.md` — quickstart. This file — full procedure.

## 2. BEFORE VAST.AI (local, no billing)
1. Push folder to private git. Vast pulls from git — never paste code.
2. Copy prompts.example.csv -> my_matrix.csv. Keep `prompt` column, one row per
   hook variant. Quote prompts containing commas. Leave size/steps empty to use
   profile. Set `seed` per row. Leave `image` empty for T2V.
3. FIRST RUN always profile `eco`. Move to `standard` only after eco smoke
   passes. `long10s` last (highest OOM risk).
4. Budget @ $0.40/hr (host #553680 full-VRAM, no tiling, 2026-09-09): eco ~1.3min (~$0.008),
   standard ~3.0min (~$0.020). 6 clips eco ~8min ~$0.05; standard ~18min ~$0.12.
   50 clips eco ~65min ~$0.43. Set Vast.ai spend alarm.
5. Accounts ready: Vast.ai + credits, SSH key added, HuggingFace account (public Apache-2.0 model). You'll provide --base-url at render time.
7. Use the same campaign/utm params at render time for consistent UTM attribution. This should be a brief summary to help align the surrounding steps more clearly. You'll provide your values at render time.


## 3. DURING VAST.AI (metered — be deliberate)
1. Create instance: RTX 5090 32GB, CUDA 12.8 image
   pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime (or Vast PyTorch 2.7 template),
   disk 60GB+ (12GB weights + 20GB torch + outputs). Prefer ON-DEMAND for
   first run (interruptible can die mid-render). Attach volume at /workspace.
2. Connect (SSH/Jupyter terminal): `nvidia-smi` must show RTX 5090 32GB.
   Driver 575+ required for Blackwell. `git clone <repo> && cd wan22-pipeline`.
3. Persist weights: `export HF_HOME=/workspace/hf-cache
   HF_HUB_ENABLE_HF_TRANSFER=1` — else every session re-downloads 12GB.
4. Setup: `bash setup.sh --profile eco --csv prompts.example.csv`
   WATCH: setup verifies torch cu128 + sm_120 + diffusers 0.35.2, aborts
   otherwise. If it aborts, STOP — wrong template or venv.
5. Smoke (~5-8min): `source .venv/bin/activate && python batch.py --csv
   prompts.example.csv --out outputs --profile eco --limit 2`
   WATCH: first clip downloads ~12GB once (slow, cached after). Monitor VRAM
   in second shell (`nvidia-smi -l 5`). Host #553680 full-VRAM default
   (tiling OFF per diffusers#12529, offload OFF): expect peak <20GB.
   Only if OOM, retry with `--offload` fallback. OOM = stay on eco, never scale up yet.
6. Inspect: `ls -lh outputs/*.mp4`, play one. Black/garbled = check
   negative_prompt + steps before scaling.
7. Full matrix: `nohup python batch.py --csv my_matrix.csv --out outputs
   --profile eco --threads 24 --base-url https://shop.example.com/p/sku > run.log 2>&1 &`
   (`--threads 24` = alokasi EPYC 7K62 host ini; default config sudah 24.)
   Use nohup/tmux so SSH drops don't kill it. Heroes after eco passes: same
8. If interrupted: re-run SAME command + `--resume` (skips ok UIDs).
   Never delete manifest.csv mid-run — it is resume state.

## 4. AFTER VAST.AI (stop billing first)
1. Verify: check manifest ok vs error counts, `ls outputs/*.mp4`, spot-play.
2. Download FIRST: scp/rsync outputs + manifest + run.log local. Confirm
   sizes/sha256 vs manifest. Backup manifest (joins filename UID =
   utm_content = sale attribution).
3. DESTROY instance in Vast.ai panel immediately. Confirm DESTROYED + billing
   stopped. Post-destroy download is impossible.
4. Review winners by hook/angle; filename UID maps clip -> exact prompt row.
5. Optional: build Dockerfile once, push to GHCR, launch next pod from it
   (~1min starts, no pip wait).

## 5. WATCH CAREFULLY (where money/sessions die)
- BILLING: idle 5090 still bills $0.40/hr. Destroy when done. Set spend limit.
- DISK: <40GB = weight download fails. Use 60GB+.
- DRIVER: Blackwell needs 575+. `nvidia-smi` showing N/A = wrong host image.
- VRAM: eco before standard/long. 32GB fits 5B full-VRAM (tiling OFF per
  diffusers#12529, offload OFF default) — expect peak <20GB. Only use
  `--offload` if OOM. Crashed minutes still bill.
- VERSION DRIFT: torch + diffusers exact-pinned. If newer transformers/
  accelerate breaks 0.35.2, pin those from run.log evidence.
- CSV: unquoted commas split columns silently — dry-run catches it free.
- RESUME: needs same manifest; new UIDs = re-renders.
- I2V image paths must exist in-pod; missing = per-row error, batch continues.
- UTM: pass --base-url at render; backfilling later breaks uid linkage.