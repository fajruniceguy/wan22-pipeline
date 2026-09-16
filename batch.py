"""Unattended batch runner: CSV hook x angle x SKU matrix -> tagged clips + manifest.

Usage:
    python batch.py --csv prompts.example.csv --out outputs --profile eco --limit 2 --dry-run
    python batch.py --csv my_matrix.csv --out outputs --profile standard
    python batch.py --csv my_matrix.csv --out outputs --profile standard --resume

Each row -> one clip named: {sku}__{hook}__{angle}__{uid}.mp4
Manifest rows carry UID, params, sha256, cost estimate, UTM skeleton.
Failures are logged (status=error) and the batch continues.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # py310 fallback
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore  # last-resort mini-parser below handles our simple config

from utils import build_utm, make_filename, make_uid, sanitize, sha256_of

BASE = Path(__file__).resolve().parent
DEFAULT_NEGATIVE = "blurry, low quality, distorted, watermark, text overlay"

MANIFEST_FIELDS = ["uid", "filename", "sku", "hook_id", "angle", "prompt", "seed",
                   "width", "height", "num_frames", "steps", "fps", "profile", "model",
                   "status", "error", "gen_sec", "est_cost_usd", "sha256", "utm_campaign",
                   "utm_content", "utm_url", "created_utc"]


def load_config(path: Path) -> dict:
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    # Minimal fallback parser: handles [section] / [a.b], str/int/float values, # comments.
    cfg: dict = {}
    section: dict | None = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                node = cfg
                for part in line[1:-1].strip().split("."):
                    node = node.setdefault(part.strip(), {})
                section = node
                continue
            if "=" in line and section is not None:
                k, v = (s.strip() for s in line.split("=", 1))
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                elif v.lower() in ("true", "false"):
                    # TOML booleans are lowercase; fallback parser must not leave
                    # them as truthy strings ("false" -> True via bool() bug).
                    v = (v.lower() == "true")
                else:
                    try:
                        v = int(v)  # type: ignore
                    except ValueError:
                        try:
                            v = float(v)  # type: ignore
                        except ValueError:
                            pass
                section[k] = v
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Wan 2.2 unattended batch clip generator (Phase 1)")
    p.add_argument("--csv", required=True, help="Prompt matrix CSV file")
    p.add_argument("--out", default="outputs", help="Output directory for clips + manifest")
    p.add_argument("--profile", default="eco", choices=["eco", "standard", "long8s", "long10s"])
    p.add_argument("--model", default=None, help="Override HF model id")
    p.add_argument("--limit", type=int, default=0, help="Only first N rows (0 = all). For smoke tests.")
    p.add_argument("--dry-run", action="store_true", help="Validate CSV + plan filenames, no model load")
    p.add_argument("--offload", action="store_true",
                   help="Enable model CPU offload (OOM fallback only; default is full-VRAM on 5090 32GB)")
    p.add_argument("--no-offload", action="store_true",
                   help="Deprecated no-op (full-VRAM is now default; tiling bug diffusers#12529)")
    p.add_argument("--threads", type=int, default=0, help="Torch CPU threads (0 = config tuning.torch_threads, 24 for EPYC 7K62)")
    p.add_argument("--resume", action="store_true", help="Skip UIDs already ok in manifest")
    p.add_argument("--manifest", default=None, help="Manifest path (default: <out>/manifest.csv)")
    p.add_argument("--base-url", default="", help="Product URL to stamp UTM links onto")
    return p.parse_args(argv)


def done_uids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    done: set[str] = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("uid"):
                done.add(row["uid"])
    return done


def estimate_cost(gen_sec: float, hourly_rate: float) -> float:
    return round(gen_sec / 3600 * hourly_rate, 4)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(BASE / "config.toml")
    prof = cfg["profiles"][args.profile]
    defaults = cfg.get("defaults", {})
    model_id = args.model or defaults.get("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    hourly_rate = float(defaults.get("hourly_rate_usd", 0.40))
    default_neg = defaults.get("negative_prompt", DEFAULT_NEGATIVE)
    # Tuning host #553680: EPYC 7K62 (24 vCPU alokasi), PCIe 4.0 x16.
    tuning = cfg.get("tuning", {})
    torch_threads = int(args.threads or tuning.get("torch_threads", 24))
    ffmpeg_threads = int(tuning.get("ffmpeg_threads", 8))
    # Full-VRAM default on 5090 32GB (tiling removed: diffusers#12529 crash).
    # --offload enables CPU offload as OOM fallback; --no-offload is a
    # deprecated no-op kept for old run commands. Config offload_* keys are
    # legacy fallbacks (default False) if neither flag is given.
    if args.offload:
        offload = True
    elif args.no_offload:
        print("[batch] WARNING: --no-offload is deprecated (full-VRAM is default); ignoring")
        offload = False
    elif args.profile.startswith("long"):
        offload = bool(tuning.get("offload_long", False))
    else:
        offload = bool(tuning.get(f"offload_{args.profile}", False))
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[batch] ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.csv"
    need_header = not manifest_path.exists()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path).fillna("")
        if "prompt" not in df.columns:
            print("[batch] ERROR: CSV needs a 'prompt' column", file=sys.stderr)
            return 2
        if args.limit and args.limit > 0:
            df = df.head(args.limit)
        rows = df.fillna("").to_dict("records")
    except ImportError:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "prompt" not in rows[0]:
            print("[batch] ERROR: CSV needs a 'prompt' column", file=sys.stderr)
            return 2
        if args.limit and args.limit > 0:
            rows = rows[:args.limit]
        for r in rows:
            for k in list(r.keys()):
                if r[k] is None:
                    r[k] = ""
    if rows and "uid" not in rows[0]:
        for r in rows:
            r["uid"] = make_uid()
    else:
        for r in rows:
            if not str(r.get("uid", "")).strip():
                r["uid"] = make_uid()
    if args.resume:
        done = done_uids(manifest_path)
        rows = [r for r in rows if str(r.get("uid")) not in done]
        print(f"[batch] resume: skipping {len(done)} done, {len(rows)} left")
    if not rows:
        print("[batch] nothing to do.")
        return 0
    print(f"[batch] model={model_id} profile={args.profile} "
          f"{prof['width']}x{prof['height']} f={prof['num_frames']} "
          f"steps={prof['num_inference_steps']} clips={len(rows)} out={out_dir} "
          f"offload={'ON-fallback' if offload else 'OFF-fullVRAM'}")
    if args.dry_run:
        for r in rows:
            w_ = int(r.get("width") or prof["width"])
            h_ = int(r.get("height") or prof["height"])
            nf = int(r.get("num_frames") or prof["num_frames"])
            st = int(r.get("steps") or prof["num_inference_steps"])
            # Host #553680: 5090 (109.1 TFLOPS, 1455.8 GB/s), EPYC 7K62, PCIe4.
            # Baseline dry-run: ~3.0 min utk 121f@720p/40 steps full-VRAM
            # (no tiling, no offload — tiling removed per diffusers#12529).
            # --offload fallback adds ~15% (PCIe4 block-swap penalty).
            base = 180.0  # detik, full-VRAM 5090 (lebih cepat dari 210 offload)
            est = base * (nf / 121) * (w_ * h_ / (1280 * 720)) ** 0.7 * (st / 40)
            if offload:
                est *= 1.15
            fname = make_filename(str(r.get("sku", "")), str(r.get("hook_id", "")),
                                  str(r.get("angle", "")), str(r.get("uid")))
            print(f"  DRY  {fname}  ~{est/60:.1f}min  ~${estimate_cost(est, hourly_rate):.3f}")
        print(f"[batch] dry-run OK: {len(rows)} clips planned.")
        return 0
    from generate import generate_clip, load_pipeline
    print("[batch] loading model (first run downloads ~12GB, cached after)...")
    print(f"[batch] host-tuning: threads={torch_threads} ffmpeg={ffmpeg_threads} offload={offload}")
    pipe = load_pipeline(model_id, offload=offload, torch_threads=torch_threads)
    ok = fail = 0
    total_sec = 0.0
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        wr = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if need_header:
            wr.writeheader()
        for i, r in enumerate(rows, 1):
            uid = str(r.get("uid"))
            sku = str(r.get("sku", ""))
            hook = str(r.get("hook_id", ""))
            angle = str(r.get("angle", ""))
            fname = make_filename(sku, hook, angle, uid)
            out_path = out_dir / fname
            width = int(r.get("width") or prof["width"])
            height = int(r.get("height") or prof["height"])
            nfr = int(r.get("num_frames") or prof["num_frames"])
            steps = int(r.get("steps") or prof["num_inference_steps"])
            seed = int(r.get("seed") or 0)
            neg = str(r.get("negative_prompt") or default_neg)
            campaign = str(r.get("utm_campaign") or f"{sanitize(sku)}-test")
            image = str(r.get("image") or "").strip() or None
            created = datetime.now(timezone.utc).isoformat()
            print(f"[batch] ({i}/{len(rows)}) {fname} ...", flush=True)
            base = {"uid": uid, "filename": fname, "sku": sku, "hook_id": hook,
                    "angle": angle, "prompt": str(r["prompt"])[:500], "seed": seed,
                    "width": width, "height": height, "num_frames": nfr,
                    "steps": steps, "fps": prof.get("fps", 24), "profile": args.profile,
                    "model": model_id, "utm_campaign": campaign, "utm_content": uid,
                    "created_utc": created}
            try:
                meta = generate_clip(pipe, str(r["prompt"]), out_path, neg, image,
                                     seed, width, height, nfr, steps,
                                     float(prof.get("guidance_scale", 5.0)),
                                     int(prof.get("fps", 24)))
                gen_sec = float(meta["gen_sec"])
                total_sec += gen_sec
                ok += 1
                wr.writerow({**base, "status": "ok", "error": "", "gen_sec": gen_sec,
                             "est_cost_usd": estimate_cost(gen_sec, hourly_rate),
                             "sha256": sha256_of(out_path),
                             "utm_url": build_utm(args.base_url, campaign, uid)})
                mf.flush()
                print(f"[batch]   done in {gen_sec:.0f}s -> {out_path.name}")
            except Exception as e:
                fail += 1
                import traceback
                traceback.print_exc()
                wr.writerow({**base, "status": "error",
                             "error": f"{type(e).__name__}: {e}"[:500],
                             "gen_sec": 0, "est_cost_usd": 0, "sha256": "", "utm_url": ""})
                mf.flush()
                try:
                    import gc
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    print(f"[batch] COMPLETE ok={ok} fail={fail} render={total_sec/60:.1f}min "
          f"~${estimate_cost(total_sec, hourly_rate):.2f} manifest={manifest_path}")
    return 1 if fail and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())


