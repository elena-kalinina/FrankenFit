"""
Gemini service stubs.

Implement with google-genai (not deprecated google-generativeai).
See func_test/passed/ for working examples of each call.

Models to target:
  - Vision / text:  gemini-2.5-flash-preview-04-17  (or gemini-3.1-flash-lite-preview)
  - TTS:            gemini-3.1-flash-tts-preview
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.models import GarmentDescription, ListingDraft, PriceBand


async def analyze_garment(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    *,
    api_key: str,
    model: str = "gemini-2.5-flash-preview-04-17",
) -> GarmentDescription:
    """
    Send image to Gemini Vision and parse the structured garment JSON response.

    Prompt shape (implement on hackathon day):
      - Ask for JSON with keys: title, description, category, style, brand,
        color, size, material, department, condition, suggested_price, currency,
        roast_line.
      - Return a GarmentDescription with a fresh garment_id (uuid4).
    """
    raise NotImplementedError("implement: Gemini vision → garment JSON")


async def generate_listing_copy(
    garment: GarmentDescription,
    price_band: PriceBand,
    *,
    api_key: str,
    marketplace: str = "ebay",
    model: str = "gemini-2.5-flash-preview-04-17",
) -> ListingDraft:
    """
    Generate marketplace-ready listing copy from garment facts + price comps.

    Prompt shape (implement on hackathon day):
      - Include garment metadata + price_band.suggested in the prompt.
      - Ask for title (≤80 chars), description paragraph, 5–8 hashtags,
        and a marketplace_copy block for each of: ebay, vinted, depop.
      - Return a fully populated ListingDraft.
    """
    raise NotImplementedError("implement: Gemini text → listing copy")


async def synthesize_tts(
    text: str,
    *,
    api_key: str,
    voice_id: str = "Aoede",
    model: str = "gemini-3.1-flash-tts-preview",
    dest: Path | None = None,
) -> bytes:
    """
    Call Gemini TTS with Director's Notes / emotion tags embedded in *text*.

    Emotion tag examples (inline in text):
      [sarcastic] Oh wonderful.
      [sighs] Fine, it stays.
      [laughs] This is not a look.

    Returns raw PCM/WAV bytes. If *dest* is given, also writes to disk.
    See func_test/passed/test_gemini_tts_sassy.py for a working reference.
    """
    raise NotImplementedError("implement: Gemini TTS → audio bytes")


async def generate_upcycle_prompt(
    garments: list[GarmentDescription],
    *,
    api_key: str,
    style_hint: str = "",
    model: str = "gemini-2.5-flash-preview-04-17",
) -> str:
    """
    Ask Gemini to write a fal FLUX.2 style prompt that merges the tossed
    garments into a single upcycled-fashion editorial shot.

    Returns a single prompt string ready for fal_service.upcycle_garments().
    """
    raise NotImplementedError("implement: Gemini → fal upcycle prompt")


def build_ebay_item_specifics(garment: GarmentDescription) -> dict[str, Any]:
    """
    Map GarmentDescription fields to the eBay ItemSpecifics dict expected by
    func_test/test_ebay_sandbox_listing.py's _build_item_specifics_xml helper.

    This is pure data mapping — no API call — so it's safe to implement now.
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
