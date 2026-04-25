"""
Backend cache + static-asset paths.

Single source of truth for *where* hackathon artefacts live on disk:

  backend/static/uploads/{garment_id}.{ext}        — original uploaded photos
  backend/static/tts/cinematic/{name}.wav          — pre-rendered cinematic clips
  backend/static/tts/garment/{garment_id}.wav      — live-rendered roast TTS
  backend/static/video/{session_id}.mp4            — per-session upcycle MP4
  backend/static/video/upcycle_hero.mp4            — generic cached fallback
  backend/static/fallbacks/*.json                  — frontend offline fallbacks

The router layer reads / writes these via the helpers here, so the URL paths
stay consistent across endpoints and the LOVABLE_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# `services/cache.py` lives 4 levels under the FrankenFit repo root.
# parents[0]=services, parents[1]=app, parents[2]=backend, parents[3]=FrankenFit
REPO_ROOT = Path(__file__).resolve().parents[3]

STATIC_DIR = REPO_ROOT / "backend" / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
TTS_DIR = STATIC_DIR / "tts"
TTS_CINEMATIC_DIR = TTS_DIR / "cinematic"
TTS_GARMENT_DIR = TTS_DIR / "garment"
VIDEO_DIR = STATIC_DIR / "video"
UPCYCLE_DIR = STATIC_DIR / "upcycle"
FALLBACKS_DIR = STATIC_DIR / "fallbacks"

# Pre-rendered cinematic clip source (lives in func_test/out/lines/).
PRERENDERED_LINES_DIR = REPO_ROOT / "func_test" / "out" / "lines"

# Pioneer live-swipes JSONL (Day 1 dataset for the overnight fine-tune).
LIVE_SWIPES_JSONL = REPO_ROOT / "func_test" / "out" / "live_swipes.jsonl"

GENERIC_VIDEO_FALLBACK = VIDEO_DIR / "upcycle_hero.mp4"
GENERIC_UPCYCLE_IMAGE_FALLBACK = UPCYCLE_DIR / "upcycle_hero.jpg"


def ensure_static_dirs() -> None:
    """Create every static subdir on backend startup. Idempotent."""
    for d in (UPLOADS_DIR, TTS_DIR, TTS_CINEMATIC_DIR, TTS_GARMENT_DIR,
              VIDEO_DIR, UPCYCLE_DIR, FALLBACKS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def sync_cinematic_clips() -> dict[str, str]:
    """Copy pre-rendered cinematic clips into static/tts/cinematic/.

    Pre-renders live in func_test/out/lines/ (committed). On startup we copy
    them into the static dir so the URL path is `/static/tts/cinematic/...`.
    Returns {clip_name: 'copied' | 'present' | 'missing'} for diagnostics.
    """
    status: dict[str, str] = {}
    for src_name in ("cold_open.wav", "upcycle_reveal.wav", "rejected_upcycle.wav", "resale_cheer.wav"):
        src = PRERENDERED_LINES_DIR / src_name
        dst = TTS_CINEMATIC_DIR / src_name
        if dst.exists():
            status[src_name] = "present"
            continue
        if src.exists():
            shutil.copy(src, dst)
            status[src_name] = "copied"
        else:
            status[src_name] = "missing"
    return status


# ---------------------------------------------------------------------------
# URL builders — the frontend calls these via the response payload.
# ---------------------------------------------------------------------------

def upload_url(garment_id: str, ext: str) -> str:
    safe_ext = ext.lstrip(".").lower() or "jpg"
    return f"/static/uploads/{garment_id}.{safe_ext}"


def upload_path(garment_id: str, ext: str) -> Path:
    safe_ext = ext.lstrip(".").lower() or "jpg"
    return UPLOADS_DIR / f"{garment_id}.{safe_ext}"


def garment_tts_url(garment_id: str) -> str:
    return f"/static/tts/garment/{garment_id}.wav"


def garment_tts_path(garment_id: str) -> Path:
    return TTS_GARMENT_DIR / f"{garment_id}.wav"


def session_video_url(session_id: str) -> str:
    return f"/static/video/{session_id}.mp4"


def session_video_path(session_id: str) -> Path:
    return VIDEO_DIR / f"{session_id}.mp4"


def generic_video_url() -> str:
    return "/static/video/upcycle_hero.mp4"


def promote_to_generic_fallback(session_video: Path) -> bool:
    """If the generic fallback MP4 doesn't exist yet, copy this session's MP4
    into place so subsequent runs have a safety net.

    Returns True iff a copy was performed.
    """
    if not session_video.exists():
        return False
    if GENERIC_VIDEO_FALLBACK.exists():
        return False
    GENERIC_VIDEO_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(session_video, GENERIC_VIDEO_FALLBACK)
    return True


def has_generic_video_fallback() -> bool:
    return GENERIC_VIDEO_FALLBACK.exists()


# ---------------------------------------------------------------------------
# Upcycle hero still — fal CDN URLs expire, so we mirror locally.
# ---------------------------------------------------------------------------

def upcycle_image_url(garment_id: str) -> str:
    return f"/static/upcycle/{garment_id}.jpg"


def upcycle_image_path(garment_id: str) -> Path:
    return UPCYCLE_DIR / f"{garment_id}.jpg"


def generic_upcycle_image_url() -> str:
    return "/static/upcycle/upcycle_hero.jpg"


def has_generic_upcycle_image_fallback() -> bool:
    return GENERIC_UPCYCLE_IMAGE_FALLBACK.exists()


def promote_upcycle_image_to_generic(garment_image: Path) -> bool:
    """Mirror the per-garment upcycle still to upcycle_hero.jpg on first run.

    Same first-wins semantics as ``promote_to_generic_fallback`` for video:
    once a canonical fallback is in place, subsequent runs don't clobber it.
    """
    if not garment_image.exists():
        return False
    if GENERIC_UPCYCLE_IMAGE_FALLBACK.exists():
        return False
    GENERIC_UPCYCLE_IMAGE_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(garment_image, GENERIC_UPCYCLE_IMAGE_FALLBACK)
    return True
