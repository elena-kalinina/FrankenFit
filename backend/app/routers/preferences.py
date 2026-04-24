"""
Preference classification route — Pioneer beat (beat 8 on stage).

POST /v1/preferences/classify — garment text → love/meh/hate (baseline vs trained)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models import ClassifyRequest, ClassifyResponse
from backend.app.session import get_session

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_preference(body: ClassifyRequest) -> ClassifyResponse:
    """
    Run garment description text through Pioneer GLiNER 2 preference classifier.

    When body.use_trained_model=True, calls the LoRA-fine-tuned model (training_job_id
    from PIONEER_TRAINED_MODEL_ID env var) alongside the baseline for side-by-side display.

    Hackathon day implementation steps:
      1. Retrieve garment description from session (or use body.text directly).
      2. Call services.pioneer.classify_preference_sidebyside(
             text, trained_model_id=..., baseline_model_id=..., api_key=...
         )
      3. Return ClassifyResponse.

    Pre-conditions:
      - Pioneer fine-tune completed (func_test/test_pioneer_finetune_loop.py --phase poll)
      - PIONEER_TRAINED_MODEL_ID set to the training_job_id from phase_train output
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    # --- stub ---
    return ClassifyResponse(
        garment_id=body.garment_id,
        label="meh",
        confidence=None,
        model_id="stub-baseline",
        baseline_label="meh",
        baseline_confidence=None,
    )
