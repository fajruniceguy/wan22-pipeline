"""Shared helpers: UID filenames, manifest rows, cost estimates."""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

_SAFE = re.compile(r"[^a-z0-9-]+")


def sanitize(s: str | None, default: str = "na") -> str:
    if not s:
        return default
    s = str(s).strip().lower().replace("_", "-").replace(" ", "-")
    s = _SAFE.sub("-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s or default


def make_uid(sku: str = "", hook_id: str = "", angle: str = "") -> str:
    """Short, unique, human-sortable ID. Example: 20260906-a3f9c1"""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    rand = uuid.uuid4().hex[:6]
    return f"{date}-{rand}"


def make_filename(sku: str, hook_id: str, angle: str, uid: str | None = None) -> str:
    """Bake the tracking UID directly into the filename.

    Format: {sku}__{hook}__{angle}__{uid}.mp4
    Double-underscore separators survive single-dash sanitization and are easy to split.
    """
    uid = uid or make_uid()
    parts = [sanitize(sku), sanitize(hook_id), sanitize(angle), sanitize(uid, default=uid)]
    return "__".join(parts) + ".mp4"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_utm(base_url: str, campaign: str, content_uid: str, source: str = "tiktok",
              medium: str = "affiliate") -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    sep = "&" if "?" in base_url else "?"
    return (f"{base_url}{sep}utm_source={source}&utm_medium={medium}"
            f"&utm_campaign={sanitize(campaign)}&utm_content={sanitize(content_uid)}")
