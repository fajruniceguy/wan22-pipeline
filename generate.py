"""Single-clip Wan 2.2 inference. Imported by batch.py; also usable standalone.

Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (unified T2V + I2V, Apache-2.0).
Target: RTX 5090 32GB (Blackwell sm_120) — needs torch cu128 / CUDA 12.8+.
The 5B fits 32GB for 720p@24fps in full-VRAM mode (no tiling, no offload).
VAE tiling is DISABLED: diffusers#12529 / Wan2.2#125 crash tiled decode
  ("size of tensor a (2) must match tensor b (4)" at avg_shortcut).
Requires: torch==2.7.1 (cu128) — torch 2.4.1/cu121 has no sm_120 kernels.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image


def load_pipeline(model_id: str, offload: bool = False, use_tf32: bool = True,
                  torch_threads: int = 24, i2v: bool = False):
    """Lazy-import diffusers so --dry-run / CSV validation works without GPU deps."""
    import os
    from diffusers import WanPipeline, WanImageToVideoPipeline

    # Host #553680 (EPYC 7K62, 24 vCPU alokasi): batasi thread CPU agar tidak
    # oversubscribe. EPYC server single-thread lemah — 24 thread pas alokasi,
    # bukan 48/96 fisik (mencegah context-switch thrash saat VAE decode + ffmpeg).
    try:
        torch_threads = max(1, int(torch_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(min(4, torch_threads))
    except Exception:
        pass

    if use_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipeline_cls = WanImageToVideoPipeline if i2v else WanPipeline
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype)
    # VAE tiling stays OFF: AutoencoderKLWan.tiled_decode() crashes on Wan2.2
    # (diffusers#12529 / Wan-Video/Wan2.2#125 — "tensor a (2) vs b (4)" at
    # avg_shortcut because _decode() drops first_chunk). TI2V-5B full-VAE
    # decode of 121f 720p (~2 GB frame buffer + few GB activations) fits
    # comfortably in 5090 32GB, so tiling buys nothing but the crash.
    try:
        pipe.vae.disable_tiling()
    except Exception:
        pass
    # Attention slicing also stays OFF: it trades ~10% speed for VRAM we
    # don't need (5B bf16 ~10 GB weights + <8 GB transient < 32GB).
    # Host #553680: full-VRAM is default now (offload=False). 5B bf16 weights
    # ~10 GB + transient <8 GB = <20 GB peak, well under 5090 32GB, and PCIe
    # 4.0 x16 @26.9GB/s makes block-swap ~2x slower than PCIe5 anyway. Keep
    # --offload only as OOM fallback (long10s edge case). PyTorch 2.7 SDP
    # default already picks the fastest sm_120 kernel; don't force manually.
    # NVMe SAMSUNG PM9A3 lokal >1GB/s: HF cache default cukup; volume /workspace
    # hanya utk reuse antar pod (lihat RUNBOOK 3.3).
    if offload and torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to("cuda")
    elif torch.cuda.is_available():
        pipe.to("cuda")
    return pipe


def generate_clip(
    pipe,
    prompt: str,
    out_path: str | Path,
    negative_prompt: str = "",
    image_path: str | Path | None = None,
    seed: int = 0,
    width: int = 832,
    height: int = 480,
    num_frames: int = 121,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    fps: int = 24,
) -> dict:
    """Run one clip. Returns timing/meta dict. Raises on failure (caller logs + continues)."""
    from diffusers.utils import export_to_video

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    kwargs: dict = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    if image_path:
        img = Image.open(image_path).convert("RGB")
        kwargs["image"] = img

    # Drop Nones (some pipeline revisions reject negative_prompt=None explicitly).
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    t0 = time.time()
    out = pipe(**kwargs)
    gen_sec = time.time() - t0

    # Diffusers video pipelines return .frames[0]; older snippets show .images — handle both.
    # NOTE: never use `or` chaining here (frames or images or videos): .frames may be
    # a numpy/torch array and `bool(array)` raises "truth value of an array is
    # ambiguous". Use explicit `is not None` checks instead.
    frames_attr = getattr(out, "frames", None)
    if frames_attr is None:
        frames_attr = getattr(out, "images", None)
    if frames_attr is None:
        frames_attr = getattr(out, "videos", None)
    if frames_attr is None and isinstance(out, dict):
        for _key in ("frames", "images", "videos"):
            if out.get(_key) is not None:
                frames_attr = out.get(_key)
                break
    if frames_attr is None:
        raise RuntimeError(f"Unexpected pipeline output type: {type(out)}")

    # Normalise batch-dim container -> single video (sequence of frames).
    # Typical: frames_attr = [[PIL, PIL, ...]] (batch of 1). Also handle
    # bare [PIL, ...], (B,T,H,W,C)/(T,H,W,C) numpy/torch arrays.
    video = frames_attr
    try:
        import numpy as _np  # local import: generate.py must import without GPU deps
        _has_np = True
    except Exception:
        _np = None  # type: ignore
        _has_np = False
    if isinstance(frames_attr, (list, tuple)):
        if len(frames_attr) == 0:
            raise RuntimeError(f"Empty pipeline output container: {type(out)}")
        first = frames_attr[0]
        # Batch-of-videos (outer batch dim) -> unwrap one level.
        # Heuristic: outer[0] is itself a frame-sequence (list/tuple) OR a
        # 4D video array (T,H,W,C), while a single frame is PIL / 3D array.
        _first_is_seq = isinstance(first, (list, tuple))
        _first_is_video_nd = False
        if _has_np and isinstance(first, _np.ndarray) and first.ndim == 4:
            _first_is_video_nd = True
        if isinstance(first, torch.Tensor) and first.dim() == 4:
            _first_is_video_nd = True
        if _first_is_seq or _first_is_video_nd:
            video = first
        else:
            video = frames_attr  # already a flat frame sequence
    elif isinstance(frames_attr, torch.Tensor):
        if frames_attr.dim() == 5:  # (B,T,H,W,C) -> first video
            video = frames_attr[0]
        elif frames_attr.dim() == 4:  # (T,H,W,C) already a video
            video = frames_attr
        else:
            raise RuntimeError(f"Unexpected video tensor shape: {tuple(frames_attr.shape)}")
    elif _has_np and isinstance(frames_attr, _np.ndarray):
        if frames_attr.ndim == 5:  # (B,T,H,W,C)
            video = frames_attr[0]
        elif frames_attr.ndim == 4:  # (T,H,W,C)
            video = frames_attr
        else:
            raise RuntimeError(f"Unexpected video array shape: {frames_attr.shape}")
    # Validate non-empty BEFORE export (len() is safe for list/tuple/ndarray/
    # tensor — unlike truthiness). Convert torch video -> numpy for export.
    try:
        _nframes = len(video)  # type: ignore[arg-type]
    except Exception:
        raise RuntimeError(f"Unindexable pipeline output container: {type(frames_attr)}")
    if _nframes == 0:
        raise RuntimeError(f"Empty video output: {type(out)}")
    if isinstance(video, torch.Tensor):
        _t = video.detach().cpu()
        if _t.is_floating_point():
            _t = (_t.clamp(0, 1) * 255).to(torch.uint8)
        video = _t.numpy()
    export_to_video(video, str(out_path), fps=int(fps), quality=10)
    return {"gen_sec": round(gen_sec, 1), "num_frames_out": len(video)}
