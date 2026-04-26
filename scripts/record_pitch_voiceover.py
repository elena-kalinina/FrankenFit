#!/usr/bin/env python3
"""
FrankenFit hackathon pitch — Gradium TTS voiceover recorder.

Records four narration clips for the demo video.  Edit the CLIPS dict below
and re-run any time to regenerate individual clips or all of them.

Usage (from repo root with venv active):
    python scripts/record_pitch_voiceover.py            # render all clips
    python scripts/record_pitch_voiceover.py --clip intro
    python scripts/record_pitch_voiceover.py --clip swipe
    python scripts/record_pitch_voiceover.py --clip pioneer
    python scripts/record_pitch_voiceover.py --list-voices   # show available voices

Output: scripts/out/voiceover_<clip>.wav
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=True)

# macOS Python 3.12 doesn't use system certs — point aiohttp at certifi's bundle.
import certifi, os as _os
_os.environ.setdefault("SSL_CERT_FILE", certifi.where())
_os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ---------------------------------------------------------------------------
# EDITABLE: voice + clips
# ---------------------------------------------------------------------------

# Leave as None to use Gradium's default voice, or set to a voice_id string.
# Run --list-voices to see what's available on your account.
VOICE_ID: str | None = None

# Clips — edit text freely, re-run to regenerate.
# Keys are used as filenames: voiceover_<key>.wav
CLIPS: dict[str, str] = {

    "intro": (
        "Show of hands: who here would voluntarily sort their closet this weekend? "
        "[beat] Right. "
        "The average person wears twenty percent of their wardrobe. "
        "The other eighty is four hundred euros of dormant inventory per closet — "
        "two-point-five billion garments to landfill every year in the E.U. "
        "We asked: why? The answer isn't taste. It's that sorting is boring. "
        "So we built an M.L. pipeline that makes sorting faster than Tinder. "
        "This is fourteen A.I. integrations across four modalities. "
        "Gemini three-point-one for vision and T.T.S., Tavily for real-time market data, "
        "fal-dot-ai Kling for image-to-video, "
        "and we trained a LoRA on GLiNER two via Pioneer overnight. "
        "The domain happens to be wardrobe auditing — "
        "but the mechanism is a general-purpose preference fine-tune loop. "
        "Here's how it works."
    ),

    "swipe": (
        "You upload a photo of your wardrobe. "
        "Gemini vision analyzes each garment — brand, condition, resale potential — "
        "and writes a one-line roast just for you. "
        "Then you swipe. Love it, hate it. Faster than Tinder, we promised. "
        "Every swipe is a labeled training signal. "
        "Right now, a zero-shot Qwen language model is classifying your taste in real time. "
        "But tonight, those swipes become a fine-tuning dataset. "
        "By morning, the model has learned your preferences "
        "from your actual wardrobe choices — not from generic fashion data. "
        "Fashion is the highest-dimensional, least-data-governed preference domain in consumer tech. "
        "If a model can learn your taste from fifty swipes on clothes, it can learn anything. "
        "Same pipeline runs on your Magic: The Gathering collection, "
        "your mechanical keyboards, your wine cellar. "
        "We picked clothing because it has a four-hundred-billion-euro resale market "
        "and a landfill problem that LEGO doesn't."
    ),

    "pioneer": (
        "This is the Style D.N.A. screen. "
        "On the left: the baseline Qwen model — strong zero-shot, good on stage. "
        "On the right: your fine-tuned GLiNER, trained overnight on your own swipes. "
        "Smaller, faster, and measurably smarter about you specifically. "
        "Training loss: five-point-one percent. Validation loss: near zero. "
        "The model learned your taste. "
        "We started with a seven-billion-parameter model. "
        "By morning, your preferences live in a task-specific model ten times smaller. "
        "That's the loop. Day one: swipe. Day two: your A.I. knows you."
    ),

}

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_OUT_DIR = _REPO_ROOT / "scripts" / "out"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def list_voices(api_key: str) -> None:
    from gradium import GradiumClient
    import gradium.voices as gv
    client = GradiumClient(api_key=api_key)
    result = await gv.get(client, include_catalog=True)
    voices = result if isinstance(result, list) else result.get("voices", [result])
    if not voices:
        print("No voices found.")
        return
    for v in voices:
        uid = v.get("uid") or v.get("voice_id") or v.get("id") or "?"
        name = v.get("name") or v.get("voice_name") or "?"
        desc = v.get("description") or ""
        print(f"  {uid}  {name}  {desc}")


async def render_clip(
    api_key: str,
    key: str,
    text: str,
    voice_id: str | None,
    out_dir: Path,
) -> Path:
    from gradium import GradiumClient
    from gradium.speech import tts, TTSSetup

    setup: TTSSetup = {"output_format": "wav"}
    if voice_id:
        setup["voice_id"] = voice_id

    print(f"  rendering '{key}' ({len(text)} chars)…", end=" ", flush=True)
    client = GradiumClient(api_key=api_key)
    result = await tts(client, setup, text)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"voiceover_{key}.wav"
    out_path.write_bytes(result.raw_data)
    duration = len(result.raw_data) / (24000 * 2) if result.raw_data else 0
    print(f"→ {out_path.name}  ({duration:.1f}s)")
    return out_path


async def main() -> int:
    parser = argparse.ArgumentParser(description="FrankenFit pitch voiceover recorder.")
    parser.add_argument("--clip", choices=list(CLIPS.keys()),
                        help="Render only this clip (default: all).")
    parser.add_argument("--voice-id", default=VOICE_ID,
                        help="Gradium voice_id to use.")
    parser.add_argument("--list-voices", action="store_true",
                        help="Print available voices and exit.")
    args = parser.parse_args()

    api_key = os.environ.get("GRADIUM_API_KEY")
    if not api_key:
        print("FAIL: GRADIUM_API_KEY not set in .env", file=sys.stderr)
        return 1

    if args.list_voices:
        await list_voices(api_key)
        return 0

    voice_id = args.voice_id

    clips_to_render = (
        {args.clip: CLIPS[args.clip]} if args.clip else CLIPS
    )

    print(f"\nRendering {len(clips_to_render)} clip(s) → {_OUT_DIR}/\n")
    for key, text in clips_to_render.items():
        await render_clip(api_key, key, text, voice_id, _OUT_DIR)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
