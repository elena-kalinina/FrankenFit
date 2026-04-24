"""
Wardrobe routes.

POST /v1/wardrobe/analyze  — image(s) → garment metadata (Gemini vision)
POST /v1/wardrobe/swipe    — record keep / toss + Pioneer taste signal
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.models import (
    AnalyzeResponse,
    GarmentDescription,
    SwipeDirection,
    SwipeRequest,
    SwipeResponse,
)
from backend.app.session import get_or_create, record_swipe

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    images: list[UploadFile] = File(..., description="One or more garment photos."),
    session_id: str | None = Form(default=None),
) -> AnalyzeResponse:
    """
    Accept garment photo(s), call Gemini Vision, return structured GarmentDescription list.

    Hackathon day implementation steps:
      1. Read each UploadFile: bytes = await img.read()
      2. Call services.gemini.analyze_garment(bytes, mime_type=img.content_type, api_key=...)
      3. Store garments in session.
      4. Return AnalyzeResponse.
    """
    session = get_or_create(session_id)

    # --- stub: return one placeholder garment per uploaded file ---
    garments: list[GarmentDescription] = []
    for img in images:
        garments.append(
            GarmentDescription(
                garment_id=str(uuid.uuid4()),
                title=f"Stub garment — {img.filename or 'unknown'}",
                description="Implement: wire Gemini Vision in wardrobe.analyze (see implementation_notes.md).",
                category="Tops",
                style="Pullover",
            )
        )
        # Store stub in session so swipe can reference the garment_id.
        session.garments[garments[-1].garment_id] = garments[-1].model_dump()

    return AnalyzeResponse(session_id=session.session_id, garments=garments)


@router.post("/swipe", response_model=SwipeResponse)
async def swipe(body: SwipeRequest) -> SwipeResponse:
    """
    Record a keep (like) or toss (dislike) swipe.

    Hackathon day extras to wire in:
      - Append to Pioneer JSONL dataset when direction=dislike (taste signal).
      - Trigger upcycle automatically when franken_bin reaches 3 garments.
    """
    meta = body.garment_meta.model_dump() if body.garment_meta else {}
    session = record_swipe(
        session_id=body.session_id,
        garment_id=body.garment_id,
        direction=body.direction.value,
        meta=meta,
    )

    taste_signal = False
    if body.direction == SwipeDirection.DISLIKE:
        # TODO: append to func_test/out/live_swipes.jsonl for Pioneer dataset
        taste_signal = False  # flip to True once implemented

    return SwipeResponse(
        session_id=session.session_id,
        garment_id=body.garment_id,
        direction=body.direction,
        keepers_count=len(session.keepers),
        franken_bin_count=len(session.franken_bin),
        taste_signal_appended=taste_signal,
    )
