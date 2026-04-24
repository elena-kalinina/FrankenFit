"""
Listing routes.

POST /v1/listings/draft    — garment + Tavily comps → listing draft (Gemini copy)
POST /v1/listings/publish  — push draft to eBay Sandbox (dry-run or live)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models import (
    GarmentDescription,
    ListingDraft,
    ListingDraftRequest,
    ListingDraftResponse,
    MarketplaceCopy,
    PriceBand,
    PublishRequest,
    PublishResponse,
)
from backend.app.session import get_session

router = APIRouter()


@router.post("/draft", response_model=ListingDraftResponse)
async def draft_listing(body: ListingDraftRequest) -> ListingDraftResponse:
    """
    Generate a ready-to-post listing draft from garment metadata + Tavily price comps.

    Hackathon day implementation steps:
      1. Retrieve garment from session by garment_id.
      2. If body.run_tavily: call services.tavily.fetch_resale_comps(garment, api_key=...)
      3. Call services.gemini.generate_listing_copy(garment, price_band, api_key=..., marketplace=...)
      4. Populate draft.ebay_item_specifics via services.gemini.build_ebay_item_specifics(garment).
      5. Store draft in session; return ListingDraftResponse.
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

    garment = GarmentDescription(**raw_garment)

    # --- stub price band ---
    price_band = PriceBand(
        min=garment.suggested_price * 0.6,
        median=garment.suggested_price * 0.9,
        suggested=garment.suggested_price,
        max=garment.suggested_price * 1.4,
        currency=garment.currency,
        sources=[],
    )

    # --- stub listing draft ---
    draft = ListingDraft(
        garment_id=garment.garment_id,
        title=garment.title,
        description=garment.description,
        suggested_price=price_band.suggested,
        currency=garment.currency,
        hashtags=["#secondhand", "#sustainable", "#frankenfit"],
        marketplace_copies=[
            MarketplaceCopy(
                platform=body.marketplace,
                title=garment.title,
                description="Implement: wire Gemini copy generation (see implementation_notes.md).",
                hashtags=["#secondhand"],
            )
        ],
        ebay_item_specifics={
            "Brand": garment.brand,
            "Department": garment.department,
            "Style": garment.style,
            "Size": garment.size,
            "Color": garment.color,
        },
    )

    session.listing_draft = draft.model_dump()
    session.price_band = price_band.model_dump()

    return ListingDraftResponse(draft=draft, price_band=price_band)


@router.post("/publish", response_model=PublishResponse)
async def publish_listing(body: PublishRequest) -> PublishResponse:
    """
    Push the session's listing draft to eBay Sandbox.

    Hackathon day implementation steps:
      1. Retrieve draft from session.
      2. Call services.ebay.publish_listing(draft, ..., dry_run=body.dry_run).
      3. If successful, store sandbox_url in session.
      4. Return PublishResponse.

    Pre-demo checklist (from BATTLE_PLAN.md):
      - Run func_test/test_ebay_sandbox_listing.py --publish first to confirm token is valid.
      - category_id must be a LEAF — use --suggest-category if 87 comes back.
    """
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id!r} not found.")

    if not session.listing_draft:
        raise HTTPException(
            status_code=422,
            detail="No listing draft in session. Call POST /v1/listings/draft first.",
        )

    # --- stub response ---
    return PublishResponse(
        ack="stub",
        item_id=None,
        errors=[{"severity": "Info", "code": "0", "short": "Stub — implement services.ebay.publish_listing."}],
        sandbox_url=None,
    )
