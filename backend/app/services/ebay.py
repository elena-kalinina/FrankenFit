"""
eBay Sandbox Trading API service stub.

Full working implementation lives in func_test/test_ebay_sandbox_listing.py.
This module exposes the same logic as a callable service so the backend router
can trigger it without shelling out to the script.

Auth: Auth'n'Auth token (EBAY_USER_TOKEN) — NOT OAuth2.
"""

from __future__ import annotations

from backend.app.models import ListingDraft, PublishResponse


async def publish_listing(
    draft: ListingDraft,
    *,
    app_id: str,
    dev_id: str,
    cert_id: str,
    user_token: str,
    site_id: str = "0",
    dry_run: bool = True,
    sandbox: bool = True,
) -> PublishResponse:
    """
    Call eBay Trading API to verify (dry_run=True) or create (dry_run=False) a listing.

    Reuse the XML builders from func_test/test_ebay_sandbox_listing.py:
      - _build_item_xml(call_name, token, ..., item_specifics)
      - _build_item_specifics_xml(specifics)

    On hackathon day:
      1. Import (or inline-copy) those helpers here.
      2. Fire the httpx POST with the correct X-EBAY-API-* headers.
      3. Parse Ack + ItemID + errors.
      4. Return PublishResponse.

    Pre-demo checklist:
      - dry_run=True passes in func_test ✓ (already verified)
      - Regenerate EBAY_USER_TOKEN if > 18 months old
    """
    raise NotImplementedError("implement: eBay Trading API publish")
