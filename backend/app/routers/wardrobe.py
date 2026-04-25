"""
Wardrobe routes.

POST /v1/wardrobe/analyze  — image(s) → garment metadata (Gemini vision)
                              + per-garment roast TTS rendered async into cache
POST /v1/wardrobe/swipe    — record keep / toss + Pioneer JSONL taste signal
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from backend.app.config import get_settings
from backend.app.models import (
    AnalyzeResponse,
    GarmentDescription,
    SwipeDirection,
    SwipeRequest,
    SwipeResponse,
)
from backend.app.services import cache, gemini, pioneer
from backend.app.session import get_or_create, record_swipe

logger = logging.getLogger(__name__)
router = APIRouter()


_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/avif": "avif",
    "image/heic": "heic",
    "image/heif": "heif",
}

_SUPPORTED_EXTS = {"jpg", "jpeg", "png", "webp", "avif", "heic", "heif"}


def _ext_from(upload: UploadFile) -> str:
    if upload.filename and "." in upload.filename:
        ext = upload.filename.rsplit(".", 1)[-1].lower()
        if ext in _SUPPORTED_EXTS:
            return "jpg" if ext == "jpeg" else ext
    return _MIME_TO_EXT.get((upload.content_type or "").lower(), "jpg")


async def _analyze_one(
    image_bytes: bytes,
    *,
    mime_type: str,
    api_key: str,
    model: str,
    fallback_models: list[str] | None = None,
) -> GarmentDescription:
    return await gemini.analyze_garment(
        image_bytes,
        mime_type=mime_type,
        api_key=api_key,
        model=model,
        fallback_models=fallback_models,
    )


async def _render_garment_tts_task(
    *,
    garment_id: str,
    text: str,
    api_key: str,
    voice_id: str,
    model: str,
    fallback_models: list[str] | None = None,
) -> None:
    """Background coroutine: render the roast TTS into the cache file.

    Failures are logged and swallowed — the frontend silently 404s when the
    audio is missing, so the demo never blocks on TTS.
    """
    if not text or not text.strip():
        return
    dest = cache.garment_tts_path(garment_id)
    if dest.exists():
        return
    try:
        await gemini.synthesize_tts(
            text,
            api_key=api_key,
            voice_id=voice_id,
            model=model,
            fallback_models=fallback_models,
            dest=dest,
        )
        logger.info("Rendered roast TTS for garment_id=%s -> %s", garment_id, dest)
    except Exception as exc:  # noqa: BLE001 — don't let demo crash on TTS
        logger.warning("Roast TTS failed for garment_id=%s: %s", garment_id, exc)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(..., description="One or more garment photos."),
    session_id: str | None = Form(default=None),
) -> AnalyzeResponse:
    """Accept garment photo(s), call Gemini Vision in parallel, return structured
    GarmentDescription list, and queue background tasks to render the roast TTS
    for each garment.

    Demo flow (BATTLE_PLAN beat 2): user uploads 6-8 photos → one call returns
    the swipe deck data. Photos saved to /static/uploads/{garment_id}.{ext}.
    """
    if not images:
        raise HTTPException(status_code=422, detail="At least one image is required.")

    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY missing on backend.")

    session = get_or_create(session_id)

    # Load upload bytes and persist to disk first; we need the bytes for Gemini
    # AND a stable on-disk path for the upcycle step later.
    pending: list[tuple[bytes, str, str, Path]] = []  # (bytes, mime, ext, dest_path)
    for img in images:
        data = await img.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"Empty upload: {img.filename!r}")
        ext = _ext_from(img)
        mime = img.content_type or f"image/{ 'jpeg' if ext == 'jpg' else ext }"
        # Reserve a garment_id slot so the path matches the analyse output.
        # We can't know the id yet — Gemini analyze() generates one — so we
        # save under a temp name and rename below. Cleaner: save AFTER analyze.
        pending.append((data, mime, ext, Path()))

    # Run all Gemini Vision calls in parallel.
    analyse_tasks = [
        _analyze_one(
            data,
            mime_type=mime,
            api_key=settings.gemini_api_key,
            model=settings.gemini_vision_model,
            fallback_models=settings.gemini_vision_fallback_models,
        )
        for data, mime, _, _ in pending
    ]
    try:
        garments: list[GarmentDescription] = await asyncio.gather(*analyse_tasks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini Vision batch failed")
        raise HTTPException(status_code=502, detail=f"Gemini Vision error: {exc}") from exc

    # Persist photos under the assigned garment_ids and update the response.
    for garment, (data, _, ext, _) in zip(garments, pending, strict=True):
        dest = cache.upload_path(garment.garment_id, ext)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        garment.image_url = cache.upload_url(garment.garment_id, ext)
        garment.tts_url = cache.garment_tts_url(garment.garment_id)
        session.garments[garment.garment_id] = garment.model_dump()
        session.garment_local_paths[garment.garment_id] = str(dest)

        # Queue the per-garment roast TTS render. The frontend silently 404s
        # until the WAV file appears, so this fires-and-forgets without blocking.
        if garment.roast_line:
            background_tasks.add_task(
                _render_garment_tts_task,
                garment_id=garment.garment_id,
                text=garment.roast_line,
                api_key=settings.gemini_api_key,
                voice_id=settings.gemini_tts_voice,
                model=settings.gemini_tts_model,
                fallback_models=settings.gemini_tts_fallback_models,
            )

    return AnalyzeResponse(session_id=session.session_id, garments=garments)


@router.post("/swipe", response_model=SwipeResponse)
async def swipe(body: SwipeRequest) -> SwipeResponse:
    """Record a keep/toss swipe and append the taste signal to the live JSONL.

    The JSONL file (func_test/out/live_swipes.jsonl) feeds the Pioneer
    overnight LoRA fine-tune (BATTLE_PLAN issue 5-G).
    """
    meta = body.garment_meta.model_dump() if body.garment_meta else {}
    session = record_swipe(
        session_id=body.session_id,
        garment_id=body.garment_id,
        direction=body.direction.value,
        meta=meta,
    )

    taste_signal = False
    text = ""
    stored = session.garments.get(body.garment_id, {})
    text = (
        stored.get("description")
        or stored.get("title")
        or (meta.get("description") if isinstance(meta, dict) else "")
        or (meta.get("title") if isinstance(meta, dict) else "")
        or ""
    )

    if text.strip():
        try:
            pioneer.append_live_swipe(
                garment_text=text,
                label=body.direction.value,
                jsonl_path=cache.LIVE_SWIPES_JSONL,
            )
            taste_signal = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to append live swipe: %s", exc)

    return SwipeResponse(
        session_id=session.session_id,
        garment_id=body.garment_id,
        direction=body.direction,
        keepers_count=len(session.keepers),
        franken_bin_count=len(session.franken_bin),
        taste_signal_appended=taste_signal,
    )
