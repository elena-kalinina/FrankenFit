"""
Upcycle routes.

POST /v1/upcycle/generate  — tossed garments → fal FLUX.2 upcycled hero still
POST /v1/upcycle/animate   — hero still → fal image-to-video MP4

DEMO NOTE: Pre-render both artefacts at T-1 day (see BATTLE_PLAN.md §4).
Do NOT trigger live fal rendering on stage — render latency is the wow-beat death zone.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models import (
    AnimateRequest,
    AnimateResponse,
    GarmentDescription,
    UpcycleRequest,
    UpcycleResponse,
)
from backend.app.session import get_session

router = APIRouter()


@router.post("/generate", response_model=UpcycleResponse)
async def generate_upcycle(body: UpcycleRequest) -> UpcycleResponse:
    """
    Combine tossed garments into a single upcycled-fashion editorial still via fal FLUX.2.

    Hackathon day implementation steps:
      1. Retrieve garments from session by garment_ids.
      2. Call services.gemini.generate_upcycle_prompt(garments, api_key=..., style_hint=body.style_prompt)
      3. Call services.fal.upcycle_garments(garments, prompt, session_id=...)
      4. Store image_url in session.upcycle_image_url.
      5. Return UpcycleResponse.
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

    garments = [GarmentDescription(**session.garments[gid]) for gid in body.garment_ids]  # noqa: F841

    # --- stub ---
    stub_url = "https://placehold.co/768x1024?text=upcycle+stub"
    session.upcycle_image_url = stub_url

    return UpcycleResponse(
        session_id=body.session_id,
        image_url=stub_url,
        revised_prompt="Stub — implement: services.gemini.generate_upcycle_prompt + services.fal.upcycle_garments",
        seed=None,
    )


@router.post("/animate", response_model=AnimateResponse)
async def animate_upcycle(body: AnimateRequest) -> AnimateResponse:
    """
    Animate the upcycled hero still with a fal.ai image-to-video model.

    Hackathon day implementation steps:
      1. Call services.fal.image_to_video(body.image_url, model=body.model, ...)
      2. Store video_url in session.upcycle_video_url.
      3. Return AnimateResponse.

    Pre-render this at T-1 day — DO NOT run live on stage.
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    # --- stub ---
    stub_url = "https://placehold.co/768x1024?text=video+stub"
    session.upcycle_video_url = stub_url

    return AnimateResponse(
        session_id=body.session_id,
        video_url=stub_url,
        model=body.model,
        duration_s=body.duration,
    )
