"""
Preference classification route — Pioneer beat (beat 8 on stage).

POST /v1/preferences/classify — garment text → love/meh/hate (Qwen baseline +
                                  fine-tuned GLiNER side-by-side).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.config import get_settings
from backend.app.models import ClassifyRequest, ClassifyResponse
from backend.app.services import pioneer
from backend.app.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_preference(body: ClassifyRequest) -> ClassifyResponse:
    """Run garment description text through Pioneer Qwen baseline AND the
    fine-tuned GLiNER 2 LoRA side-by-side. Returns trained label/confidence
    plus baseline label/confidence so the frontend can render the
    "10× smaller, same answer" efficiency story.

    Required env vars:
      - PIONEER_API_KEY
      - PIONEER_TRAINED_MODEL_ID  (training_job_id from the Day-1 fine-tune)
      - PIONEER_QWEN_MODEL        (full Qwen model id used as the baseline)
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    settings = get_settings()
    if not settings.pioneer_api_key:
        raise HTTPException(status_code=503, detail="PIONEER_API_KEY missing on backend.")

    text = body.text.strip() if body.text else ""
    if not text:
        garment = session.garments.get(body.garment_id) or {}
        text = (garment.get("description") or garment.get("title") or "").strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="No text to classify. Provide body.text or call /analyze first.",
        )

    trained_id = settings.pioneer_trained_model_id
    baseline_id = settings.pioneer_qwen_model
    if not trained_id and not baseline_id:
        raise HTTPException(
            status_code=503,
            detail="Pioneer model IDs missing. Set PIONEER_TRAINED_MODEL_ID and/or PIONEER_QWEN_MODEL.",
        )

    try:
        return await pioneer.classify_preference_sidebyside(
            text,
            garment_id=body.garment_id,
            api_key=settings.pioneer_api_key,
            trained_model_id=trained_id,
            baseline_model_id=baseline_id,
            api_base=settings.pioneer_api_base,
            per_call_timeout=settings.pioneer_per_call_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pioneer side-by-side failed")
        raise HTTPException(status_code=502, detail=f"Pioneer classify error: {exc}") from exc
