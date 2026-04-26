"""
Gemini service — backend implementations.

Uses the supported `google-genai` SDK throughout (the legacy `google-generativeai`
package is deprecated). All functions are awaitable; the underlying SDK calls
are sync, so we run them via `asyncio.to_thread(...)` to avoid blocking the
FastAPI event loop.

Reference (proven) scripts ported here:
  - func_test/passed/test_gemini_garment_description.py  →  analyze_garment
  - func_test/passed/test_gemini_tts_sassy.py            →  synthesize_tts
  - func_test/passed/test_marketplace_listing_draft.py   →  generate_listing_copy
  - func_test/test_fal_upcycle_from_garments.py          →  generate_upcycle_prompt

Models:
  Vision / text:  gemini-3.1-flash-lite-preview  (override via GEMINI_VISION_MODEL)
  TTS:            gemini-3.1-flash-tts-preview   (override via GEMINI_TTS_MODEL)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from backend.app.models import (
    GarmentDescription,
    ListingDraft,
    MarketplaceCopy,
    PriceBand,
)

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the first complete JSON object from text.

    Gemini occasionally appends commentary or a second object after the main
    JSON block even when response_mime_type='application/json' is set.  This
    helper finds the first '{...}' span and parses only that.
    """
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    # Try direct parse first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Walk forward to find balanced first object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in Gemini response: {text[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in Gemini response: {text[:200]!r}")

DEFAULT_VISION_MODEL = "gemini-3-flash-preview"
# Tried left-to-right when the primary 503s / 429s. Last entry is the emergency
# GA model with the highest free-tier daily quota — guarantees the demo never
# dies on Gemini availability even if every preview model is saturated.
DEFAULT_VISION_FALLBACK_MODELS: list[str] = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
]
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
# TTS preview models live on tiny per-day free-tier quotas (~10 req/day each).
# We rotate through them so when one is exhausted the next picks up. Order is:
# fastest → most-likely-healthy. Pro-tts last (slower, but separate quota pool).
DEFAULT_TTS_FALLBACK_MODELS: list[str] = [
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-pro-preview-tts",
]
DEFAULT_TTS_VOICE = "Puck"

# Status codes / status names that warrant a retry (transient overload).
# google-genai raises ServerError(status_code, response_json, response).
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_STATUS_NAMES = {
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
}


def _is_retryable_error(exc: BaseException) -> bool:
    """True if exc is a transient Gemini error worth retrying / falling back."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
        return True
    status = (getattr(exc, "status", "") or "").upper()
    if status in _RETRYABLE_STATUS_NAMES:
        return True
    msg = str(exc).upper()
    return any(name in msg for name in _RETRYABLE_STATUS_NAMES) or any(
        f" {c} " in f" {msg} " for c in (" 503", " 429", " 500", " 502", " 504")
    )


def _try_model(
    client,
    model: str,
    *,
    attempts: int,
    schedule: list[float],
    label: str,
    role: str,  # "primary" | "fallback"
    **call_kwargs,
):
    """Try a single model up to ``attempts`` times, sleeping between transient retries.

    Returns the SDK response on success. On non-retryable error: raises immediately.
    On exhausting all attempts with retryable errors: returns None (caller moves on
    to the next model in the chain).
    """
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            resp = client.models.generate_content(model=model, **call_kwargs)
            if attempt > 0 or role == "fallback":
                logger.warning(
                    "[gemini] %s served by %s model=%s (attempt %d)",
                    label, role.upper(), model, attempt + 1,
                )
            return resp, None
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc):
                # Permanent error — re-raise immediately, don't waste fallbacks.
                raise
            if attempt == attempts - 1:
                logger.warning(
                    "[gemini] %s exhausted %s=%s after %d attempts: %s",
                    label, role, model, attempts, str(exc).split("\n")[0][:160],
                )
                break
            sleep_for = schedule[min(attempt, len(schedule) - 1)]
            logger.warning(
                "[gemini] %s transient on %s=%s (attempt %d/%d): %s — retrying in %.1fs",
                label, role, model, attempt + 1, attempts,
                str(exc).split("\n")[0][:160], sleep_for,
            )
            time.sleep(sleep_for)
    return None, last_exc


def _generate_with_retry(
    client_factory,
    *,
    primary_model: str,
    fallback_models: list[str] | None = None,
    label: str,
    primary_attempts: int = 3,
    fallback_attempts: int = 2,
    backoff_seconds: tuple[float, ...] = (1.0, 3.0, 6.0),
    **call_kwargs,
):
    """Call ``client.models.generate_content`` with retries + automatic model chain.

    Tries primary_model first (with up to ``primary_attempts`` retries on transient
    503/429/UNAVAILABLE/RESOURCE_EXHAUSTED). On exhaustion, walks down
    ``fallback_models`` left-to-right (with up to ``fallback_attempts`` retries each).

    Permanent errors (4xx other than 429, schema errors, etc.) are raised immediately
    so we don't waste fallbacks on a broken request. Returns the SDK response object.
    """
    client = client_factory()
    schedule = list(backoff_seconds) + [backoff_seconds[-1]] * 8  # generous tail
    last_exc: BaseException | None = None

    chain: list[tuple[str, str, int]] = [(primary_model, "primary", primary_attempts)]
    seen = {primary_model}
    for fb in fallback_models or []:
        if fb and fb not in seen:
            chain.append((fb, "fallback", fallback_attempts))
            seen.add(fb)

    for model, role, attempts in chain:
        resp, exc = _try_model(
            client, model,
            attempts=attempts, schedule=schedule, label=label, role=role,
            **call_kwargs,
        )
        if resp is not None:
            return resp
        last_exc = exc

    assert last_exc is not None
    raise last_exc

# Director's Notes for the Franken-Fit narrator: witty about clothes/eras only,
# never cruel about the wearer's body. Same preset as func_test STYLE_SASSY.
ROAST_DIRECTOR_NOTES = (
    "Style: Enthusiastic and highly sassy. Tone: Playful and slightly sarcastic. "
    "Pacing: Energetic with sharp enunciation. "
    "You narrate for a sustainable-fashion wardrobe app: witty about clothes and eras only, "
    "never cruel and never about the wearer's body."
)

# Schema we want Gemini Vision to fill from a single garment photo. We ask for
# every field of GarmentDescription up-front so the analyze endpoint is a single
# round-trip — including the comedic roast_line, which the per-garment TTS task
# will read aloud.
_GARMENT_PROMPT = """\
You are a wardrobe analyst for a sustainable second-hand fashion app called Franken-Fit.

Look at the single main garment in the photo and return a JSON object with EXACTLY these keys:

{
  "title": "short marketable listing title, max 80 chars, no emoji, no quotes",
  "description": "1-paragraph listing-ready description, 50-90 words, second-hand resale tone",
  "category": "top-level category, one of: Tops | Bottoms | Outerwear | Dresses | Footwear | Accessories",
  "style": "specific style descriptor, e.g. Pullover, Blazer, Maxi Dress, Sneakers, Trucker Jacket",
  "brand": "brand visible on the garment, or 'Unbranded' if none",
  "color": "dominant color, single word or two-word descriptor (e.g. 'Charcoal', 'Off-white')",
  "size": "size label visible on the garment, or best guess from fit cues (S | M | L | XL | numeric)",
  "material": "fabric/material best guess, e.g. Wool, Polyester, Denim, Cotton",
  "department": "intended department, one of: Women | Men | Unisex Adult | Boys | Girls",
  "condition": "human-readable condition label, one of: New with tags | Like new | Used | Used - good | Used - fair",
  "condition_id": "eBay ConditionID matching condition: 1000=New with tags, 1500=New other, 2750=Like new, 3000=Used, 4000=Very good, 5000=Good, 6000=Acceptable. Return the numeric string.",
  "suggested_price": <number, USD, conservative quick-sale resale price for second-hand>,
  "currency": "USD",
  "roast_line": "see ROAST_LINE RULES below",
  "stylist_suggestion": "see STYLIST_SUGGESTION RULES below",
  "vibe_tags": ["array of 3-8 short style tags, e.g. minimal, y2k, normcore, streetwear"],
  "era_guess": "approximate era, e.g. '1990s', 'Y2K', 'contemporary'",
  "pattern": "pattern descriptor, e.g. 'solid', 'striped', 'floral', 'graphic-print'"
}

================================================================
ROAST_LINE — RULES (this is what plays out loud on the swipe card)
================================================================

Format: ONE breath. Single sentence. **Max 90 characters total** (including audio tags).
You are a grumpy fashion stylist who has seen everything — witty, exhausted,
slightly mean about CLOTHES AND ERAS ONLY, never about the wearer's body or face.

Required structure (exactly one of these):
  A) [audio_tag] One-liner with a punchline.
  B) [audio_tag] Setup. [audio_tag] Punchline.        ← only if total stays ≤ 90 chars

Audio tags you may use inline (Gemini TTS reads them as expressions, not words):
  [sighs] [gasps] [whispered] [laughs] [scoffs] [scandalized] [horrified]
  [sarcastic] [amusement] [annoyance] [bored] [resigned] [delighted]
  [mock_excitement] [smug] [deadpan]

HARD RULES — break any of these and the line is unusable:
  1. ≤ 90 chars TOTAL (count tags). Punchy beats clever.
  2. AT LEAST ONE inline audio tag at the very start.
  3. Punch lands in the LAST 5 words. No "but..." pivots dragging the joke out.
  4. Anchor to ONE specific signal: the era, the brand, the fabric, the silhouette,
     OR the trope it represents — never multiple.
  5. NEVER use these tired tropes (overused already):
     - "It's giving [X]"
     - "[Item] called, it wants [Y] back"
     - "fast-fashion crime scene"
     - "Pinterest board"
  6. NEVER reference the wearer's body, age, gender, weight, or appearance.
  7. NEVER swear. NEVER mention real human names.

GOOD examples (study the rhythm — these are the bar):
  - "[gasps] A 2014 Topshop blazer? [horrified] We buried this look for a reason."
  - "[laughs] *Aggressively* Sambas. [bored] Pick a personality."
  - "[sighs] Asymmetric hem in 2026. [resigned] Brave choice."
  - "[whispered] This says 'I peaked in college.' [scoffs] The hem confirms it."
  - "[scandalized] Cargo pockets AND a drawstring? [deadpan] No."
  - "[smug] Wool blend. [amusement] In Lisbon. [sighs] Optimistic."
  - "[deadpan] Polyester pretending to be silk. [scoffs] We see you."

BAD examples (do NOT do these — too long, multi-pivot, or cliché):
  - "These pants say I have a gym membership, but the wide leg says I only go there to flex"
    (too long, two clauses joined by 'but')
  - "It's giving 2015 minimalist core energy with a side of regret"
    (banned trope "It's giving")
  - "Oh wonderful, another fast-fashion crime scene"
    (banned trope "fast-fashion crime scene")

================================================================
STYLIST_SUGGESTION — RULES (the comedic turn when the user LIKES)
================================================================

This is the SAME stylist who roasted the garment, now grudgingly helping.
Same dry voice, but constructive. Frontend hides this until LIKE is hit, then
slides it up at the bottom of the screen for ~3 seconds. That contrast —
roast on swipe-in, useful tip on like — IS the joke.

Format: ONE sentence. **50–110 chars total**. Plain text — NO audio tags
(this isn't read aloud, it's read on screen).

What it MUST do:
  1. Name 1-2 SPECIFIC items to pair with — "wide-leg jeans", "white tank",
     "loafers", "thin gold chain" — never vague ("denim", "accessories").
  2. OR name 1 specific styling ACTION — "half-tuck", "cuff the sleeves once",
     "layer over a graphic tee", "leave unbuttoned".
  3. End with a half-sentence justification in the stylist's voice — dry,
     specific, no fluff. The justification is what stops it sounding generic.

What it MUST NOT do:
  - "Easy to dress up or down" — meaningless.
  - "Pairs with anything" — meaningless.
  - "Great with denim" — too vague.
  - Repeat the roast line's vibe — switch to constructive, not snarky.
  - Use audio tags. (Roast has them; this doesn't.)
  - Mention the wearer's body / age / gender.

GOOD examples (the bar — note the structure: pair + action, then a dry kicker):
  - "Tuck into wide black trousers and skip the belt — let the silhouette do the talking."
  - "Layer over a white tank with cuffed sleeves — kills the corporate aftertaste."
  - "Anchor with high-rise denim and a thin gold chain — let the sleeves stay loud."
  - "Pair with chunky loafers and ankle socks — the only way this proportion works."
  - "Half-tuck a black tee and add silver hoops — quiet luxury, loud earrings."
  - "Crop a hoodie at the waist and white sneakers — owning the gym-rat brief."
  - "Throw on stone-washed jeans and a leather belt — undercuts the formality just enough."

BAD examples (do NOT do these):
  - "Great with anything in your closet, super versatile." (vague + meaningless)
  - "Try wearing it with cool shoes and a fun bag." (no specifics)
  - "[sighs] Pair with denim." (audio tag — wrong, not for TTS)
  - "Pair with the trousers you swiped in the prior step." (assumes context)

Output ONLY the JSON object, no markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Garment analysis
# ---------------------------------------------------------------------------

def _mime_for_image_bytes(mime_hint: str | None) -> str:
    if mime_hint and mime_hint.startswith("image/"):
        return mime_hint
    return "image/jpeg"


def _analyze_garment_sync(
    image_bytes: bytes,
    mime_type: str,
    *,
    api_key: str,
    model: str,
    fallback_models: list[str] | None,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.7,
    )
    resp = _generate_with_retry(
        lambda: genai.Client(api_key=api_key),
        primary_model=model,
        fallback_models=fallback_models,
        label="analyze_garment",
        contents=[_GARMENT_PROMPT, image_part],
        config=config,
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty response for garment analysis.")
    return _extract_json(text)


async def analyze_garment(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    *,
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    fallback_models: list[str] | None = None,
) -> GarmentDescription:
    """Send image to Gemini Vision and parse the structured garment JSON response.

    Returns a GarmentDescription with a fresh `garment_id` (uuid4). Extra fields
    Gemini supplies (vibe_tags, era_guess, pattern, …) are stashed in `raw`.
    """
    mime = _mime_for_image_bytes(mime_type)
    chain = fallback_models if fallback_models is not None else DEFAULT_VISION_FALLBACK_MODELS
    raw = await asyncio.to_thread(
        _analyze_garment_sync,
        image_bytes,
        mime,
        api_key=api_key,
        model=model,
        fallback_models=chain,
    )

    # Coerce types defensively — Gemini occasionally returns numbers as strings.
    def _f(key: str, default: float) -> float:
        v = raw.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _s(key: str, default: str) -> str:
        v = raw.get(key, default)
        return str(v) if v is not None else default

    garment = GarmentDescription(
        garment_id=str(uuid.uuid4()),
        title=_s("title", "Untitled garment")[:80],
        description=_s("description", ""),
        category=_s("category", "Tops"),
        style=_s("style", "Pullover"),
        brand=_s("brand", "Unbranded"),
        color=_s("color", "Multicolor"),
        size=_s("size", "M"),
        material=_s("material", "Unknown"),
        department=_s("department", "Women"),
        condition=_s("condition", "Used"),
        condition_id=_s("condition_id", "3000"),
        suggested_price=round(_f("suggested_price", 19.99), 2),
        currency=_s("currency", "USD"),
        roast_line=_s("roast_line", ""),
        stylist_suggestion=_s("stylist_suggestion", "")[:140],
        raw=raw,
    )
    return garment


# ---------------------------------------------------------------------------
# Listing copy generation (Gemini text)
# ---------------------------------------------------------------------------

_LISTING_PROMPT = """\
You write second-hand resale listings. Given garment facts and an optional price band,
produce a single JSON object with this exact shape:

{
  "title": "max 80 chars, hook-forward, no emoji",
  "description": "2-3 short paragraphs, honest and charming, no hype",
  "hashtags": ["6-10 lowercase hashtags without the leading # symbol"],
  "marketplace_copies": [
    {
      "platform": "ebay",
      "title": "max 80 chars, eBay tone (product-forward, search-keyword friendly)",
      "description": "2-3 short paragraphs",
      "hashtags": []
    },
    {
      "platform": "vinted",
      "title": "max 80 chars, Vinted tone (honest charming second-hand)",
      "description": "2-3 short paragraphs",
      "hashtags": ["6-10 short lowercase tags, no #"]
    },
    {
      "platform": "depop",
      "title": "max 80 chars, Depop tone (punchy, vintage/aesthetic vocabulary, no misrepresentation)",
      "description": "1 short paragraph; first line hooks the style vibe",
      "hashtags": ["10-20 lowercase Depop-style tags, no #"]
    }
  ]
}

Garment facts:
{garment_json}

Price band (optional, may be null):
{price_band_json}

Output ONLY the JSON object, no markdown fences.
"""


def _generate_listing_copy_sync(
    *,
    garment_json: str,
    price_band_json: str,
    api_key: str,
    model: str,
    fallback_models: list[str] | None,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    prompt = _LISTING_PROMPT.replace("{garment_json}", garment_json).replace(
        "{price_band_json}", price_band_json
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.5,
    )
    resp = _generate_with_retry(
        lambda: genai.Client(api_key=api_key),
        primary_model=model,
        fallback_models=fallback_models,
        label="listing_copy",
        contents=prompt,
        config=config,
    )
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty response for listing copy.")
    return _extract_json(text)


async def generate_listing_copy(
    garment: GarmentDescription,
    price_band: PriceBand | None,
    *,
    api_key: str,
    marketplace: str = "ebay",
    model: str = DEFAULT_VISION_MODEL,
    fallback_models: list[str] | None = None,
) -> ListingDraft:
    """Generate marketplace-ready listing copy from garment facts + price comps.

    Returns a fully populated ListingDraft, including marketplace_copies for
    eBay / Vinted / Depop. eBay item specifics are always derived from the
    garment via build_ebay_item_specifics().
    """
    suggested_price = price_band.suggested if price_band else garment.suggested_price
    currency = (price_band.currency if price_band else garment.currency) or "USD"

    garment_payload = {
        "title": garment.title,
        "category": garment.category,
        "style": garment.style,
        "brand": garment.brand,
        "color": garment.color,
        "size": garment.size,
        "material": garment.material,
        "department": garment.department,
        "condition": garment.condition,
        "suggested_price": suggested_price,
        "currency": currency,
        "extra": garment.raw,
    }
    band_payload = price_band.model_dump() if price_band else None

    chain = fallback_models if fallback_models is not None else DEFAULT_VISION_FALLBACK_MODELS
    raw = await asyncio.to_thread(
        _generate_listing_copy_sync,
        garment_json=json.dumps(garment_payload, ensure_ascii=False),
        price_band_json=json.dumps(band_payload, ensure_ascii=False),
        api_key=api_key,
        model=model,
        fallback_models=chain,
    )

    title = str(raw.get("title") or garment.title)[:80]
    description = str(raw.get("description") or garment.description)
    hashtags = [str(t).lstrip("#") for t in (raw.get("hashtags") or []) if t]

    copies: list[MarketplaceCopy] = []
    for entry in raw.get("marketplace_copies") or []:
        if not isinstance(entry, dict):
            continue
        copies.append(
            MarketplaceCopy(
                platform=str(entry.get("platform") or "").lower() or marketplace,
                title=str(entry.get("title") or title)[:80],
                description=str(entry.get("description") or description),
                hashtags=[str(t).lstrip("#") for t in (entry.get("hashtags") or []) if t],
            )
        )

    return ListingDraft(
        garment_id=garment.garment_id,
        title=title,
        description=description,
        suggested_price=round(float(suggested_price), 2),
        currency=currency,
        hashtags=hashtags,
        marketplace_copies=copies,
        ebay_item_specifics=build_ebay_item_specifics(garment),
    )


# ---------------------------------------------------------------------------
# TTS — Gemini 3.1 Flash TTS preview with Director's Notes + inline tags
# ---------------------------------------------------------------------------

def _write_wav_pcm16(path: Path, pcm_bytes: bytes, sample_rate_hz: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_bytes)


def _synthesize_tts_sync(
    *,
    director: str,
    spoken: str,
    api_key: str,
    voice_id: str,
    model: str,
    fallback_models: list[str] | None,
) -> tuple[bytes, int, str]:
    """Returns (pcm_bytes, sample_rate_hz, mime_string).

    Walks ``model`` (primary) and then any ``fallback_models`` left-to-right.
    TTS preview models live on tiny per-day free-tier quotas — when one is
    exhausted the chain rolls forward instead of failing the whole render.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    merged_prompt = (
        f"{director.strip()}\n\n"
        "Say exactly the following line. Treat bracketed cues like [sarcastic], "
        "[sighs], [laughs] as audio expressions, not words to read aloud:\n"
        f"\"{spoken.strip()}\""
    )
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_id)
            )
        ),
    )
    response = _generate_with_retry(
        lambda: client,
        primary_model=model,
        fallback_models=fallback_models,
        label="tts",
        # Lighter retries per model than for vision: TTS quotas reset hourly,
        # so spending 6+ seconds backing off on one model is wasteful when the
        # next model in the chain is probably healthy right now.
        primary_attempts=2,
        fallback_attempts=2,
        backoff_seconds=(1.0, 2.0),
        contents=merged_prompt,
        config=config,
    )
    cand = response.candidates[0] if response.candidates else None
    if not cand or not cand.content or not cand.content.parts:
        raise RuntimeError("Gemini TTS returned no audio parts.")

    pcm: bytes | None = None
    mime = "audio/L16;codec=pcm;rate=24000"
    for part in cand.content.parts:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        raw = getattr(inline, "data", None)
        if raw is None:
            continue
        mime = getattr(inline, "mime_type", None) or mime
        pcm = base64.b64decode(raw) if isinstance(raw, str) else raw
        break

    if not pcm:
        raise RuntimeError("Gemini TTS response did not contain audio bytes.")

    sample_rate = 24000
    mime_lc = (mime or "").lower()
    if "rate=" in mime_lc:
        try:
            rate_part = mime_lc.split("rate=")[1].split(";")[0].split(",")[0].strip()
            sample_rate = int(rate_part)
        except ValueError:
            pass
    return pcm, sample_rate, mime


async def synthesize_tts(
    text: str,
    *,
    api_key: str,
    voice_id: str = DEFAULT_TTS_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    fallback_models: list[str] | None = None,
    director_notes: str = ROAST_DIRECTOR_NOTES,
    dest: Path | None = None,
) -> bytes:
    """Call Gemini TTS with Director's Notes + inline audio tags in *text*.

    Returns the raw PCM bytes (s16 LE @ 24000 Hz). If *dest* is given, also
    writes a WAV file at that path. The caller is responsible for caching the
    output if needed.
    """
    if not text or not text.strip():
        raise ValueError("synthesize_tts called with empty text.")

    chain = fallback_models if fallback_models is not None else DEFAULT_TTS_FALLBACK_MODELS
    pcm, sample_rate, _ = await asyncio.to_thread(
        _synthesize_tts_sync,
        director=director_notes,
        spoken=text,
        api_key=api_key,
        voice_id=voice_id,
        model=model,
        fallback_models=chain,
    )

    if dest is not None:
        _write_wav_pcm16(Path(dest), pcm, sample_rate_hz=sample_rate)

    return pcm


# ---------------------------------------------------------------------------
# Upcycle prompt — Gemini reads the tossed-garment images and writes a
# single FLUX.2-ready style prompt.
# ---------------------------------------------------------------------------

_UPCYCLE_PROMPT_INSTRUCTION = (
    "You are designing a Franken-Fit upcycle: combine the garments shown into ONE "
    "cohesive avant-garde runway piece. Write a single detailed image-generation prompt "
    "for a photorealistic editorial product shot (neutral studio background, dramatic light). "
    "No text in the image. Max 120 words. Output plain text only, no quotes."
)


def _generate_upcycle_prompt_sync(
    image_paths: list[Path],
    *,
    api_key: str,
    model: str,
    fallback_models: list[str] | None,
    style_hint: str,
) -> str:
    from google import genai
    from google.genai import types

    parts: list[Any] = []
    instruction = _UPCYCLE_PROMPT_INSTRUCTION
    if style_hint:
        instruction = f"{instruction}\n\nAdditional style direction from the user: {style_hint.strip()}"
    parts.append(instruction)
    for p in image_paths:
        suffix = p.suffix.lower()
        mime = (
            "image/png" if suffix == ".png"
            else "image/webp" if suffix == ".webp"
            else "image/jpeg"
        )
        parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
    resp = _generate_with_retry(
        lambda: genai.Client(api_key=api_key),
        primary_model=model,
        fallback_models=fallback_models,
        label="upcycle_prompt",
        contents=parts,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return (getattr(resp, "text", "") or "").strip()


async def generate_upcycle_prompt(
    image_paths: list[Path],
    *,
    api_key: str,
    style_hint: str = "",
    model: str = DEFAULT_VISION_MODEL,
    fallback_models: list[str] | None = None,
) -> str:
    """Ask Gemini to write a fal FLUX.2 style prompt that merges the tossed
    garments into a single upcycled-fashion editorial shot.

    Accepts a list of local image paths (the tossed garments). Reads bytes
    inline so we don't need to upload to fal storage just for the prompt step.
    """
    if not image_paths:
        raise ValueError("generate_upcycle_prompt requires at least one image path.")
    chain = fallback_models if fallback_models is not None else DEFAULT_VISION_FALLBACK_MODELS
    prompt = await asyncio.to_thread(
        _generate_upcycle_prompt_sync,
        image_paths,
        api_key=api_key,
        model=model,
        fallback_models=chain,
        style_hint=style_hint,
    )
    if not prompt:
        raise RuntimeError("Gemini returned an empty upcycle prompt.")
    return prompt


# ---------------------------------------------------------------------------
# eBay item specifics — pure data mapping (no API call)
# ---------------------------------------------------------------------------

def build_ebay_item_specifics(garment: GarmentDescription) -> dict[str, str]:
    """Map GarmentDescription fields to the eBay ItemSpecifics dict.

    The sandbox's apparel categories require at least Brand, Department,
    Style, Size, Color — missing any of these returns error code 21919303.
    """
    return {
        "Brand": garment.brand or "Unbranded",
        "Department": garment.department or "Women",
        "Style": garment.style or "Pullover",
        "Size": garment.size or "M",
        "Size Type": "Regular",
        "Color": garment.color or "Multicolor",
        "Type": garment.category or "Sweater",
        "Material": garment.material or "Knit",
        "Country/Region of Manufacture": "Unknown",
    }
