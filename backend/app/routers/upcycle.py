"""
Upcycle routes.

POST /v1/upcycle/generate  — tossed garments → fal FLUX.2 multi-reference upcycle
POST /v1/upcycle/animate   — hero still → fal image-to-video MP4
                              (live render with cached fallback on timeout)

Cache strategy (from BATTLE_PLAN.md §6 + the Saturday hackathon decision):
  - First successful animate run downloads the MP4 to /static/video/{session_id}.mp4
    AND, if no generic exists yet, copies it to /static/video/upcycle_hero.mp4.
  - Subsequent runs serve the per-session MP4 immediately if present.
  - Live render is wrapped in asyncio.wait_for(timeout=N). On timeout the
    cached generic is served instead — never crash the demo.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.app.config import get_settings
from backend.app.models import (
    AnimateRequest,
    AnimateResponse,
    GarmentDescription,
    UpcycleRequest,
    UpcycleResponse,
)
from backend.app.services import cache, fal, gemini
from backend.app.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=UpcycleResponse)
async def generate_upcycle(body: UpcycleRequest) -> UpcycleResponse:
    """Combine tossed garments into a single upcycled-fashion editorial still
    via fal FLUX.2 [flex]. Registers the upcycle as a synthetic garment in the
    session so /v1/listings/draft can target it directly afterwards.
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    missing = [gid for gid in body.garment_ids if gid not in session.garments]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Garment IDs not found in session: {missing}",
        )

    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY missing on backend.")
    if not settings.fal_key:
        raise HTTPException(status_code=503, detail="FAL_KEY missing on backend.")

    paths: list[Path] = []
    for gid in body.garment_ids:
        local = session.garment_local_paths.get(gid)
        if not local:
            raise HTTPException(
                status_code=409,
                detail=f"No local image stored for garment {gid!r}. Re-run /analyze.",
            )
        p = Path(local)
        if not p.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"Stored image for garment {gid!r} is missing on disk: {p}",
            )
        paths.append(p)

    try:
        prompt = await gemini.generate_upcycle_prompt(
            paths,
            api_key=settings.gemini_api_key,
            style_hint=body.style_prompt,
            model=settings.gemini_vision_model,
            fallback_models=settings.gemini_vision_fallback_models,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini upcycle prompt failed")
        raise HTTPException(status_code=502, detail=f"Upcycle prompt error: {exc}") from exc

    try:
        result = await fal.upcycle_garments(
            paths,
            prompt,
            session_id=session.session_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("fal FLUX.2 flex edit failed")
        raise HTTPException(status_code=502, detail=f"FLUX.2 flex edit error: {exc}") from exc

    remote_image_url = result["image_url"]
    revised_prompt = result.get("revised_prompt") or prompt
    seed = result.get("seed")

    upcycle_id = str(uuid.uuid4())

    # fal.media URLs are short-lived. Mirror the JPG into our static dir so
    # the demo never depends on the CDN being warm 90 minutes from now.
    # We always serve the local URL; the remote URL is kept in raw{} for debug.
    served_image_url = remote_image_url
    local_path = cache.upcycle_image_path(upcycle_id)
    try:
        await fal.download_to(remote_image_url, local_path)
        served_image_url = cache.upcycle_image_url(upcycle_id)
        # Promote first successful run to generic fallback (no-op if exists).
        if cache.promote_upcycle_image_to_generic(local_path):
            logger.info("Promoted upcycle hero %s to generic %s",
                        local_path, cache.GENERIC_UPCYCLE_IMAGE_FALLBACK)
    except Exception as exc:  # noqa: BLE001 — never fail the demo on a mirror miss
        logger.warning("Upcycle hero mirror failed (%s); serving fal URL %s",
                       exc, remote_image_url)

    upcycle_garment = GarmentDescription(
        garment_id=upcycle_id,
        title="Franken-Fit upcycled creation",
        description=(
            f"One-of-a-kind upcycled garment combining {len(body.garment_ids)} "
            "tossed pieces into a single editorial silhouette. Crafted by Franken-Fit."
        ),
        category="Outerwear",
        style="Avant-garde",
        brand="Franken-Fit",
        color="Multicolor",
        size="One size",
        material="Mixed",
        department="Unisex Adult",
        condition="New with tags",
        condition_id="1000",
        suggested_price=89.0,
        currency="USD",
        roast_line="",
        image_url=served_image_url,
        tts_url="",
        raw={"upcycle_prompt": prompt, "revised_prompt": revised_prompt, "seed": seed,
             "fal_image_url": remote_image_url,
             "source_garment_ids": list(body.garment_ids)},
    )
    session.garments[upcycle_id] = upcycle_garment.model_dump()
    session.upcycle_image_url = served_image_url
    session.upcycle_garment_id = upcycle_id

    return UpcycleResponse(
        session_id=session.session_id,
        garment_id=upcycle_id,
        image_url=served_image_url,
        revised_prompt=revised_prompt,
        seed=seed,
        model="fal-ai/flux-2-flex/edit",
    )


@router.post("/animate", response_model=AnimateResponse)
async def animate_upcycle(body: AnimateRequest) -> AnimateResponse:
    """Animate the upcycled hero still with a fal.ai image-to-video model.

    Live render with cached fallback:
      1. If /static/video/{session_id}.mp4 already exists → serve it immediately.
      2. Else attempt live render with timeout=body.timeout_seconds.
      3. On success: download MP4 → cache; promote to generic if no generic exists.
      4. On timeout / failure: serve /static/video/upcycle_hero.mp4 if present;
         else 503.
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    settings = get_settings()
    if not settings.fal_key:
        raise HTTPException(status_code=503, detail="FAL_KEY missing on backend.")

    image_url = body.image_url or session.upcycle_image_url or ""
    if not image_url:
        raise HTTPException(
            status_code=422,
            detail="image_url missing and session has no upcycle_image_url. Call /v1/upcycle/generate first.",
        )

    cache_path = cache.session_video_path(session.session_id)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        session.upcycle_video_url = cache.session_video_url(session.session_id)
        return AnimateResponse(
            session_id=session.session_id,
            video_url=session.upcycle_video_url,
            model=body.model,
            duration_s=body.duration,
            cached=True,
        )

    # Live render with bounded wait.
    try:
        result = await asyncio.wait_for(
            fal.image_to_video(
                image_url,
                session_id=session.session_id,
                model=body.model,
                duration=body.duration,
                aspect_ratio=body.aspect_ratio,
            ),
            timeout=max(5.0, float(body.timeout_seconds)),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "I2V live render exceeded %.1fs — serving cached generic fallback.",
            body.timeout_seconds,
        )
        if cache.has_generic_video_fallback():
            url = cache.generic_video_url()
            session.upcycle_video_url = url
            return AnimateResponse(
                session_id=session.session_id,
                video_url=url,
                model=body.model,
                duration_s=body.duration,
                cached=True,
            )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Live image-to-video render timed out after {body.timeout_seconds:.0f}s "
                "and no cached fallback exists. Run the generate flow once to prime the cache."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("fal image-to-video failed")
        if cache.has_generic_video_fallback():
            url = cache.generic_video_url()
            session.upcycle_video_url = url
            return AnimateResponse(
                session_id=session.session_id,
                video_url=url,
                model=body.model,
                duration_s=body.duration,
                cached=True,
            )
        raise HTTPException(status_code=502, detail=f"I2V error: {exc}") from exc

    video_url_remote = result["video_url"]

    # Download the remote MP4 to the cache path so subsequent loads are instant
    # AND we have a stable local fallback if fal CDN times out next time.
    try:
        await fal.download_to(video_url_remote, cache_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to download MP4 to cache (%s); falling back to remote URL.", exc)
        session.upcycle_video_url = video_url_remote
        return AnimateResponse(
            session_id=session.session_id,
            video_url=video_url_remote,
            model=result.get("model", body.model),
            duration_s=int(result.get("duration_s", body.duration)),
            cached=False,
        )

    if cache.promote_to_generic_fallback(cache_path):
        logger.info("Promoted %s to generic fallback %s",
                    cache_path, cache.GENERIC_VIDEO_FALLBACK)

    served_url = cache.session_video_url(session.session_id)
    session.upcycle_video_url = served_url
    return AnimateResponse(
        session_id=session.session_id,
        video_url=served_url,
        model=result.get("model", body.model),
        duration_s=int(result.get("duration_s", body.duration)),
        cached=False,
    )
