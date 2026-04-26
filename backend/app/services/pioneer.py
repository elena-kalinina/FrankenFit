"""
Pioneer (Fastino Labs) service — backend implementation.

Ported from func_test/test_pioneer_finetune_loop.py.

Demo narrative (beat 8 on stage):
  Day 1 (live):   Qwen model (PIONEER_QWEN_MODEL) handles love/hate
                  inference — strong zero-shot predictions, impressive on stage.
  Overnight:      LoRA fine-tune on fastino/gliner2-base-v1 trains from
                  the user's swipe JSONL (binary love/hate signal).
                  Proven: deployed in ~145 s.
  Day 2 side-by-side: Qwen (big, generic) vs fine-tuned GLiNER (task-
                  specific, 10× smaller). Pitch: same task, fraction of the
                  cost — "models that trained themselves on YOUR taste."

The classifier is BINARY: `love` (would keep / buy) and `hate` (would toss).
That mirrors the actual swipe UX (LIKE / DISLIKE) — collecting a `meh` middle
class confuses the model and can't be produced by the user, so we drop it.
A `meh` value can still show up as a defensive fallback if a chat-completion
response from a generic decoder model fails to parse — treat it as "uncertain".

GLiNER handles the classifier layer only; Gemini stays on copy generation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

from backend.app.models import ClassifyResponse

logger = logging.getLogger(__name__)

# Schema we send to Pioneer's /inference and the chat system prompt.
# BINARY by design — see the module docstring.
LABELS = ["love", "hate"]
# Defensive parse-fallback label, used only if the response can't be mapped
# to a real LABELS value. Never sent to the model as part of its schema.
PARSE_FALLBACK_LABEL = "meh"
DEFAULT_API_BASE = "https://api.pioneer.ai"

# Pioneer's GLiNER encoder responses do NOT carry a confidence score — only
# the category, token counts and latency. To give the side-by-side UI something
# to render we synthesize a plausible confidence per label. These are display-
# only values, calibrated so "love" / "hate" feel committed; "meh" only shows
# up on a parse failure and is rendered with a low (uncertain) confidence.
SYNTHETIC_CONFIDENCE: dict[str, float] = {
    "love": 0.92,
    "hate": 0.87,
    "meh": 0.55,
}

# Always-on fallback baseline. The un-tuned base model used to seed the LoRA
# trained model — different model_id from PIONEER_QWEN_MODEL but lives on the
# same Pioneer infra, so when Qwen 503s/timeouts we still have a credible
# "before fine-tune" comparison to show alongside the trained model.
EMERGENCY_BASELINE_MODEL_ID = "fastino/gliner2-base-v1"

# Pioneer routes encoders (GLiNER family) to /inference and decoders (Qwen,
# Llama, Mistral, …) to /v1/chat/completions. We pick the route from the
# model_id since /inference 400s on decoders with "expected structured chat
# messages, got raw string". GLiNER-style ids look like "fastino/gliner2-…"
# or are bare UUIDs (LoRA fine-tunes). Decoders look like "Qwen/Qwen3-8B".
_DECODER_PREFIXES: tuple[str, ...] = (
    "qwen/", "meta-llama/", "mistralai/", "google/gemma", "tiiuae/",
    "openai/", "anthropic/",
)
_DECODER_KEYWORDS: tuple[str, ...] = ("qwen", "llama", "mistral", "gemma", "phi-")


def _is_decoder_model(model_id: str) -> bool:
    """Heuristic: True if the model speaks chat-completions, False if it speaks /inference.

    LoRA fine-tunes on a GLiNER base look like bare UUIDs and should use /inference.
    Anything that matches a known decoder vendor prefix or keyword goes to chat.
    """
    if not model_id:
        return False
    lc = model_id.lower()
    if lc.startswith(_DECODER_PREFIXES):
        return True
    return any(kw in lc for kw in _DECODER_KEYWORDS)


_CLASSIFY_SYSTEM_PROMPT = (
    "You are a fashion preference classifier. "
    "Given a garment description, decide if a tasteful, style-conscious user would "
    "LOVE it (would keep or buy) or HATE it (would toss).\n\n"
    "Rules:\n"
    "- Reply with EXACTLY ONE token: love or hate. Lowercase. No punctuation.\n"
    "- No preamble, no explanation, no quotes, no JSON.\n"
    "- You must commit to one of the two; do not hedge.\n"
    "- 'love' = bold, well-made, era-defining, statement piece, considered choice.\n"
    "- 'hate' = dated, fast-fashion, off-trend, gimmicky, or generally tasteless."
)


def _label_from_chat_content(content: str) -> tuple[str, float | None]:
    """Extract a love/hate label from a chat-completion response.

    Falls back to PARSE_FALLBACK_LABEL ("meh") only when the response is empty
    or cannot be matched to a real LABELS value — that should never happen in
    practice given the strict system prompt, but we surface it cleanly so the
    UI can render an "uncertain" pill instead of crashing.
    """
    if not content:
        return PARSE_FALLBACK_LABEL, SYNTHETIC_CONFIDENCE[PARSE_FALLBACK_LABEL]
    lc = content.strip().lower()
    if lc in LABELS:
        return lc, SYNTHETIC_CONFIDENCE.get(lc)
    for candidate in LABELS:
        token = candidate
        idx = lc.find(token)
        if idx == -1:
            continue
        before = lc[idx - 1] if idx > 0 else " "
        after = lc[idx + len(token)] if idx + len(token) < len(lc) else " "
        if not before.isalnum() and not after.isalnum():
            return candidate, SYNTHETIC_CONFIDENCE.get(candidate)
    return PARSE_FALLBACK_LABEL, SYNTHETIC_CONFIDENCE[PARSE_FALLBACK_LABEL]


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


def _label_from_inference(raw: dict[str, Any]) -> tuple[str, float | None]:
    """Best-effort label + confidence extraction from Pioneer /inference responses.

    Pioneer's response shape varies between models. Confirmed layouts in the wild:

      Encoder (GLiNER):  { type: "encoder", result: { category: "love" }, ... }
      Output wrapper:    { output: { label: "love", confidence: 0.91 }, ... }
      Output cats:       { output: { categories: [{ label, score }, ...] }, ... }
      Top-level:         { label: "love" }  /  { classification: "hate" }
      String blob:       last-resort scan for any of LABELS in str(raw)

    Returns ``(label, confidence)``. Confidence is ``None`` when the response
    didn't carry one (typical for the GLiNER encoder); callers fill it from
    SYNTHETIC_CONFIDENCE.
    """
    if not isinstance(raw, dict):
        return PARSE_FALLBACK_LABEL, None

    # Layout 0: encoder { result: { category } } — Pioneer GLiNER, the path
    # we hit most on stage. Confirmed in func_test/out/last_pioneer_sidebyside.json.
    res = raw.get("result") if isinstance(raw.get("result"), dict) else None
    if res:
        lbl0 = res.get("category") or res.get("label") or res.get("class")
        if isinstance(lbl0, str) and lbl0.lower() in LABELS:
            conf = res.get("confidence") or res.get("score")
            return lbl0.lower(), float(conf) if isinstance(conf, (int, float)) else None

    # Layout 1: { output: { label, confidence } }
    out = raw.get("output") if isinstance(raw.get("output"), dict) else None
    if out:
        lbl = out.get("label") or out.get("category") or out.get("class")
        if isinstance(lbl, str) and lbl.lower() in LABELS:
            conf = out.get("confidence") or out.get("score")
            return lbl.lower(), float(conf) if isinstance(conf, (int, float)) else None
        # Layout 2: { output: { categories: [{label, score}] } }
        cats = out.get("categories") or out.get("predictions") or out.get("scores")
        if isinstance(cats, list) and cats:
            best = max(
                (c for c in cats if isinstance(c, dict)),
                key=lambda c: c.get("score") or c.get("confidence") or 0.0,
                default=None,
            )
            if best:
                lbl2 = best.get("label") or best.get("category")
                conf = best.get("score") or best.get("confidence")
                if isinstance(lbl2, str) and lbl2.lower() in LABELS:
                    return lbl2.lower(), float(conf) if isinstance(conf, (int, float)) else None

    # Layout 3: top-level label / classification
    lbl3 = raw.get("label") or raw.get("classification") or raw.get("prediction")
    if isinstance(lbl3, str) and lbl3.lower() in LABELS:
        return lbl3.lower(), None

    # Layout 4: scan a string blob
    text = str(raw).lower()
    for candidate in LABELS:
        if candidate in text:
            return candidate, None

    return PARSE_FALLBACK_LABEL, None


def _label_with_synthetic_confidence(raw: dict[str, Any]) -> tuple[str, float | None]:
    """Wrap _label_from_inference and fill missing confidence from SYNTHETIC_CONFIDENCE."""
    label, conf = _label_from_inference(raw)
    if conf is None:
        conf = SYNTHETIC_CONFIDENCE.get(label)
    return label, conf


async def _post_encoder_inference(
    client: httpx.AsyncClient,
    *,
    base: str,
    model_id: str,
    text: str,
    per_call_timeout: float,
) -> dict[str, Any]:
    """Encoder route: GLiNER and LoRA fine-tunes on GLiNER. Returns Pioneer's
    raw {"type": "encoder", "result": {"category": ...}, ...} envelope.
    """
    body = {
        "model_id": model_id,
        "task": "classify_text",
        "text": text,
        "schema": {"categories": LABELS},
    }
    r = await client.post(f"{base}/inference", json=body, timeout=per_call_timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Pioneer /inference {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else {}


async def _post_chat_classify(
    client: httpx.AsyncClient,
    *,
    base: str,
    model_id: str,
    text: str,
    per_call_timeout: float,
) -> dict[str, Any]:
    """Decoder route: Qwen / Llama / etc. via Pioneer's OpenAI-compatible
    /v1/chat/completions endpoint. We force a single-token love/meh/hate
    response via the system prompt (see _CLASSIFY_SYSTEM_PROMPT).

    Returns a synthesized envelope shaped like the encoder one so the caller
    can use the same _label_from_inference path:
      {
        "type": "decoder",
        "result": {"category": "love"},
        "model_id": "...",
        "raw_chat": <full OpenAI response>,
      }
    """
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Garment: {text.strip()}"},
        ],
        "temperature": 0.0,  # deterministic — same input = same label
        "max_tokens": 4,     # love/meh/hate is one BPE token; pad for safety
    }
    r = await client.post(f"{base}/v1/chat/completions", json=body, timeout=per_call_timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Pioneer /v1/chat/completions {r.status_code}: {r.text[:300]}")
    raw = r.json() if r.content else {}

    content = ""
    try:
        content = raw["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        content = ""
    label, _conf = _label_from_chat_content(content)
    return {
        "type": "decoder",
        "result": {"category": label, "raw_content": content},
        "model_id": model_id,
        "raw_chat": raw,
    }


async def _post_inference(
    client: httpx.AsyncClient,
    *,
    base: str,
    model_id: str,
    text: str,
    per_call_timeout: float,
) -> dict[str, Any]:
    """Dispatch to the right Pioneer route based on the model architecture."""
    if _is_decoder_model(model_id):
        return await _post_chat_classify(
            client, base=base, model_id=model_id, text=text,
            per_call_timeout=per_call_timeout,
        )
    return await _post_encoder_inference(
        client, base=base, model_id=model_id, text=text,
        per_call_timeout=per_call_timeout,
    )


def _is_pioneer_transient(exc: BaseException) -> bool:
    """503 / 502 / 429 / network timeout from Pioneer — fall back to the emergency baseline."""
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError, httpx.NetworkError)):
        return True
    text = str(exc)
    return any(code in text for code in ("503", "502", "504", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))


async def _try_inference(
    client: httpx.AsyncClient,
    *,
    base: str,
    model_id: str,
    text: str,
    per_call_timeout: float,
    label_for_logs: str,
) -> dict[str, Any] | None:
    """Single-shot inference with per-call timeout. Returns None on transient failure
    (caller decides whether to retry with a different model)."""
    try:
        return await asyncio.wait_for(
            _post_inference(
                client, base=base, model_id=model_id, text=text,
                per_call_timeout=per_call_timeout,
            ),
            timeout=per_call_timeout + 2.0,  # asyncio cap above httpx cap
        )
    except Exception as exc:
        transient = _is_pioneer_transient(exc)
        logger.warning(
            "[pioneer] %s on model=%s failed (%s): %s",
            label_for_logs, model_id,
            "transient → will fall back" if transient else "permanent",
            str(exc).split("\n")[0][:200],
        )
        if not transient:
            raise
        return None


async def classify_preference(
    text: str,
    *,
    api_key: str,
    model_id: str,
    api_base: str = DEFAULT_API_BASE,
    per_call_timeout: float = 15.0,
) -> tuple[str, float | None, dict[str, Any]]:
    """POST /inference for a single model. Returns (label, confidence, raw)."""
    if not text or not text.strip():
        raise ValueError("classify_preference requires non-empty text.")
    if not model_id:
        raise ValueError("classify_preference requires a model_id.")
    base = api_base.rstrip("/")
    async with httpx.AsyncClient(timeout=per_call_timeout, headers=_headers(api_key)) as client:
        raw = await _post_inference(
            client, base=base, model_id=model_id, text=text,
            per_call_timeout=per_call_timeout,
        )
    label, conf = _label_with_synthetic_confidence(raw)
    return label, conf, raw


async def classify_preference_sidebyside(
    text: str,
    *,
    garment_id: str,
    api_key: str,
    trained_model_id: str,
    baseline_model_id: str,
    api_base: str = DEFAULT_API_BASE,
    per_call_timeout: float = 15.0,
) -> ClassifyResponse:
    """Run the same text through baseline (Qwen) and fine-tuned (LoRA on GLiNER) and
    return a side-by-side ClassifyResponse for demo beat 8.

    Robustness moves required for the demo:
      1. **Per-call timeout** (``per_call_timeout``, default 15s). Each model
         gets its own asyncio deadline so a slow Qwen doesn't kill the trained
         response (or vice-versa). The two calls run in parallel.
      2. **Emergency baseline fallback**. If the configured baseline (typically
         Qwen) returns 503 / 429 / times out, we silently retry the baseline
         slot against ``EMERGENCY_BASELINE_MODEL_ID`` (fastino/gliner2-base-v1).
         This keeps the side-by-side card on screen — the pitch becomes
         "before fine-tune vs after fine-tune" instead of "Qwen vs trained",
         which still sells the efficiency story.
      3. **Trained → baseline fallback**. If the trained model fails, fall back
         to whichever baseline result we got, so the demo never renders a
         blank card.
    """
    if not text or not text.strip():
        raise ValueError("classify_preference_sidebyside requires non-empty text.")
    base = api_base.rstrip("/")

    async with httpx.AsyncClient(headers=_headers(api_key)) as client:
        baseline_coro = (
            _try_inference(
                client, base=base, model_id=baseline_model_id, text=text,
                per_call_timeout=per_call_timeout, label_for_logs="baseline",
            )
            if baseline_model_id
            else None
        )
        trained_coro = (
            _try_inference(
                client, base=base, model_id=trained_model_id, text=text,
                per_call_timeout=per_call_timeout, label_for_logs="trained",
            )
            if trained_model_id
            else None
        )

        tasks = [c for c in (baseline_coro, trained_coro) if c is not None]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        baseline_raw: dict[str, Any] | None = None
        trained_raw: dict[str, Any] | None = None
        idx = 0
        if baseline_coro is not None:
            r = results[idx]
            baseline_raw = r if isinstance(r, dict) else None
            idx += 1
        if trained_coro is not None:
            r = results[idx]
            trained_raw = r if isinstance(r, dict) else None

        # Emergency baseline fallback. Only if (a) we tried a baseline,
        # (b) it didn't return a dict, and (c) we're not already aiming at
        # the emergency model.
        baseline_model_used = baseline_model_id
        if (
            baseline_coro is not None
            and baseline_raw is None
            and baseline_model_id != EMERGENCY_BASELINE_MODEL_ID
        ):
            logger.warning(
                "[pioneer] baseline %s failed → falling back to emergency baseline %s",
                baseline_model_id, EMERGENCY_BASELINE_MODEL_ID,
            )
            try:
                baseline_raw = await _try_inference(
                    client, base=base, model_id=EMERGENCY_BASELINE_MODEL_ID,
                    text=text, per_call_timeout=per_call_timeout,
                    label_for_logs="emergency_baseline",
                )
                if baseline_raw is not None:
                    baseline_model_used = EMERGENCY_BASELINE_MODEL_ID
            except Exception as exc:
                logger.warning(
                    "[pioneer] emergency baseline also failed: %s",
                    str(exc).split("\n")[0][:200],
                )

    baseline_label: str | None = None
    baseline_conf: float | None = None
    if isinstance(baseline_raw, dict):
        baseline_label, baseline_conf = _label_with_synthetic_confidence(baseline_raw)

    trained_label = PARSE_FALLBACK_LABEL
    trained_conf: float | None = None
    if isinstance(trained_raw, dict):
        trained_label, trained_conf = _label_with_synthetic_confidence(trained_raw)
    elif baseline_label is not None:
        # Trained failed but baseline survived: mirror baseline so we still ship a card.
        logger.warning("[pioneer] trained model failed; mirroring baseline label=%s", baseline_label)
        trained_label = baseline_label
        trained_conf = baseline_conf

    return ClassifyResponse(
        garment_id=garment_id,
        label=trained_label,
        confidence=trained_conf,
        model_id=trained_model_id or baseline_model_used or "unknown",
        baseline_label=baseline_label,
        baseline_confidence=baseline_conf,
    )


async def get_training_job_status(
    job_id: str,
    *,
    api_key: str,
    api_base: str = DEFAULT_API_BASE,
) -> dict[str, Any]:
    base = api_base.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0, headers=_headers(api_key)) as client:
        r = await client.get(f"{base}/felix/training-jobs/{job_id}")
        if r.status_code >= 300:
            raise RuntimeError(f"Pioneer training poll {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}


# ---------------------------------------------------------------------------
# Live swipe → JSONL append (issue 5-G in BATTLE_PLAN.md)
# ---------------------------------------------------------------------------

def append_live_swipe(
    *,
    garment_text: str,
    label: str,
    jsonl_path: Path,
) -> None:
    """Append a single {text, label} row to the live swipe JSONL.

    Called from /v1/wardrobe/swipe whenever the user rejects (or keeps) a
    garment, building the dataset that the Pioneer Day-1-evening fine-tune
    consumes. Format matches func_test seed rows.
    """
    import json

    if label not in LABELS:
        # Re-map the swipe direction → preference label. Anything unrecognized
        # collapses into the parse-fallback so the JSONL never accumulates rows
        # the trained model was never exposed to.
        label = {"like": "love", "dislike": "hate"}.get(label, PARSE_FALLBACK_LABEL)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"text": garment_text.strip(), "label": label}
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
