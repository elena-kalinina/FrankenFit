"""
Pioneer (Fastino Labs) service stubs.

API base: https://api.pioneer.ai  — auth header: X-API-Key
See func_test/test_pioneer_finetune_loop.py for the full Day-1→Day-2 flow.

These endpoint stubs cover the runtime inference call only (beat 8 on stage).
The fine-tune workflow runs offline via func_test/, not through the live API.
"""

from __future__ import annotations

from backend.app.models import ClassifyResponse


async def classify_preference(
    text: str,
    *,
    api_key: str,
    model_id: str,
    api_base: str = "https://api.pioneer.ai",
) -> dict[str, object]:
    """
    POST /inference — classify a garment description as love / meh / hate.

    Returns the raw inference JSON from Pioneer.
    Wrap result into ClassifyResponse in the router.
    """
    raise NotImplementedError("implement: Pioneer /inference → preference label")


async def classify_preference_sidebyside(
    text: str,
    *,
    api_key: str,
    trained_model_id: str,
    baseline_model_id: str,
    api_base: str = "https://api.pioneer.ai",
) -> ClassifyResponse:
    """
    Run the same text through both baseline and trained models and return a
    side-by-side ClassifyResponse for the demo's beat 8 ("the model learned you").
    """
    raise NotImplementedError("implement: Pioneer side-by-side inference")


async def get_training_job_status(
    job_id: str,
    *,
    api_key: str,
    api_base: str = "https://api.pioneer.ai",
) -> dict[str, object]:
    """
    GET /felix/training-jobs/{job_id} — poll fine-tune status.
    Returns the raw job JSON (status: queued | running | completed | failed).
    """
    raise NotImplementedError("implement: Pioneer training job status poll")
