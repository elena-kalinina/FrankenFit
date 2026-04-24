"""
Pioneer (Fastino Labs) service stubs.

API base: https://api.pioneer.ai  — auth header: X-API-Key
See func_test/test_pioneer_finetune_loop.py for the full Day-1→Day-2 flow.

Demo narrative (beat 8 on stage):
  Day 1 (live):   Qwen model (PIONEER_QWEN_MODEL) handles love/meh/hate
                  inference — strong zero-shot predictions, impressive on stage.
  Overnight:      LoRA fine-tune on fastino/gliner2-base-v1 trains from
                  the user's swipe JSONL. Proven: deployed in ~145 s.
  Day 2 side-by-side: Qwen (big, generic) vs fine-tuned GLiNER (task-
                  specific, 10× smaller). Pitch: same task, fraction of the
                  cost — "models that trained themselves on YOUR taste."

Q&A guard-rails:
  - GLiNER handles the classifier layer only; Gemini stays on copy generation.
  - Side-by-side is an efficiency win (size/cost), not accuracy win.
    Both models should agree on the label — choose garment probes that
    both call correctly.
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

    On hackathon Day 1, pass model_id = settings.pioneer_qwen_model (Qwen).
    On hackathon Day 2, pass model_id = settings.pioneer_trained_model_id (fine-tuned GLiNER).

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
    Run the same text through Qwen (baseline) and fine-tuned GLiNER (trained)
    and return a side-by-side ClassifyResponse for demo beat 8.

    baseline_model_id = settings.pioneer_qwen_model  (big, generic)
    trained_model_id  = settings.pioneer_trained_model_id  (compact, task-specific)

    Implementation hint:
      1. Call classify_preference(text, model_id=baseline_model_id, ...)
      2. Call classify_preference(text, model_id=trained_model_id, ...)
      3. Merge into ClassifyResponse.
      Both calls can be awaited concurrently with asyncio.gather().
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
    Returns the raw job JSON (status: queued | running | complete | deployed | failed).
    Pre-hackathon test job: 941f616d-4c09-43eb-9155-80a623efde83 — deployed in 145 s.
    """
    raise NotImplementedError("implement: Pioneer training job status poll")
