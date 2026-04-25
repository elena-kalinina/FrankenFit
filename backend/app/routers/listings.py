"""
Listing routes.

POST /v1/listings/draft    — garment + Tavily comps → listing draft (Gemini copy)
POST /v1/listings/publish  — push draft to eBay Sandbox (dry-run by default)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.config import get_settings
from backend.app.models import (
    GarmentDescription,
    ListingDraft,
    ListingDraftRequest,
    ListingDraftResponse,
    PriceBand,
    PublishRequest,
    PublishResponse,
)
from backend.app.services import ebay, gemini, tavily
from backend.app.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/draft", response_model=ListingDraftResponse)
async def draft_listing(body: ListingDraftRequest) -> ListingDraftResponse:
    """Generate a marketplace-ready listing draft from garment metadata + live
    Tavily price comps, with Gemini-generated copy per marketplace.
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    raw_garment = session.garments.get(body.garment_id)
    if not raw_garment:
        raise HTTPException(
            status_code=404,
            detail=f"Garment {body.garment_id!r} not found in session {body.session_id!r}.",
        )

    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY missing on backend.")

    garment = GarmentDescription(**raw_garment)

    price_band: PriceBand | None = None
    if body.run_tavily and settings.tavily_api_key:
        try:
            price_band = await tavily.fetch_resale_comps(
                garment,
                api_key=settings.tavily_api_key,
                marketplace="any",
                region="eu",
                currency=garment.currency,
            )
        except Exception as exc:  # noqa: BLE001 — Tavily failures shouldn't block the listing
            logger.warning("Tavily price comps failed: %s", exc)
            price_band = None

    # Fallback price band — keeps the demo flowing if Tavily returns nothing.
    if price_band is None:
        price_band = PriceBand(
            min=round(garment.suggested_price * 0.6, 2),
            median=round(garment.suggested_price * 0.9, 2),
            suggested=round(garment.suggested_price, 2),
            max=round(garment.suggested_price * 1.4, 2),
            currency=garment.currency,
            sources=[],
        )

    try:
        draft: ListingDraft = await gemini.generate_listing_copy(
            garment,
            price_band,
            api_key=settings.gemini_api_key,
            marketplace=body.marketplace,
            model=settings.gemini_vision_model,
            fallback_models=settings.gemini_vision_fallback_models,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini listing copy failed")
        raise HTTPException(status_code=502, detail=f"Gemini listing copy error: {exc}") from exc

    session.listing_draft = draft.model_dump()
    session.price_band = price_band.model_dump()

    return ListingDraftResponse(draft=draft, price_band=price_band)


@router.post("/publish", response_model=PublishResponse)
async def publish_listing(body: PublishRequest) -> PublishResponse:
    """Push the session's listing draft to eBay Sandbox via Trading API."""
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    if not session.listing_draft:
        raise HTTPException(
            status_code=422,
            detail="No listing draft in session. Call POST /v1/listings/draft first.",
        )

    settings = get_settings()
    draft = ListingDraft(**session.listing_draft)

    raw_garment = session.garments.get(body.garment_id) or session.garments.get(draft.garment_id)
    condition_id = (raw_garment or {}).get("condition_id", "3000") if raw_garment else "3000"

    try:
        result = await ebay.publish_listing(
            draft,
            app_id=settings.ebay_app_id,
            dev_id=settings.ebay_dev_id,
            cert_id=settings.ebay_cert_id,
            user_token=settings.ebay_user_token,
            site_id=settings.ebay_site_id,
            category_id=settings.ebay_category_id,
            condition_id=str(condition_id),
            dry_run=body.dry_run,
            sandbox=settings.ebay_env == "sandbox",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("eBay publish failed")
        raise HTTPException(status_code=502, detail=f"eBay publish error: {exc}") from exc

    if result.sandbox_url:
        session.upcycle_video_url = session.upcycle_video_url  # noop, kept for symmetry
    return result
