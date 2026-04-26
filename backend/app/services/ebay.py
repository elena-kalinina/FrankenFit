"""
eBay Sandbox Trading API service — backend implementation.

Ported from func_test/test_ebay_sandbox_listing.py. Uses the Trading API XML
schema with Auth'n'Auth user token (NOT OAuth2). Defaults to the sandbox
endpoint — refuses to publish to production unless EBAY_ENV says so AND the
caller passes dry_run=False explicitly.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from backend.app.models import ListingDraft, PublishResponse

SANDBOX_ENDPOINT = "https://api.sandbox.ebay.com/ws/api.dll"
PRODUCTION_ENDPOINT = "https://api.ebay.com/ws/api.dll"
COMPATIBILITY_LEVEL = "1193"


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_item_specifics_xml(specifics: dict[str, str | list[str]] | None) -> str:
    if not specifics:
        return ""
    rows: list[str] = []
    for name, raw in specifics.items():
        if raw is None:
            continue
        values: list[str]
        if isinstance(raw, (list, tuple)):
            values = [str(v).strip() for v in raw if str(v).strip()]
        else:
            s = str(raw).strip()
            values = [s] if s else []
        if not values:
            continue
        value_xml = "\n      ".join(f"<Value>{_xml_escape(v)}</Value>" for v in values)
        rows.append(
            "    <NameValueList>\n"
            f"      <Name>{_xml_escape(name)}</Name>\n"
            f"      {value_xml}\n"
            "    </NameValueList>"
        )
    if not rows:
        return ""
    return "  <ItemSpecifics>\n" + "\n".join(rows) + "\n  </ItemSpecifics>\n"


def _build_item_xml(
    *,
    call_name: str,
    token: str,
    title: str,
    description: str,
    price: float,
    currency: str,
    country: str,
    postal_code: str,
    category_id: str,
    condition_id: str,
    quantity: int,
    item_specifics: dict[str, str | list[str]] | None,
    picture_url: str | None = None,
) -> str:
    specifics_xml = _build_item_specifics_xml(item_specifics)
    picture_xml = (
        f"    <PictureDetails><PictureURL>{_xml_escape(picture_url)}</PictureURL></PictureDetails>\n"
        if picture_url else ""
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<{call_name}Request xmlns="urn:ebay:apis:eBLBaseComponents">
  <RequesterCredentials>
    <eBayAuthToken>{_xml_escape(token)}</eBayAuthToken>
  </RequesterCredentials>
  <ErrorLanguage>en_US</ErrorLanguage>
  <WarningLevel>High</WarningLevel>
  <Item>
    <Title>{_xml_escape(title)[:80]}</Title>
    <Description>{_xml_escape(description)}</Description>
    <PrimaryCategory><CategoryID>{_xml_escape(category_id)}</CategoryID></PrimaryCategory>
    <StartPrice currencyID="{_xml_escape(currency)}">{price:.2f}</StartPrice>
    <CategoryMappingAllowed>true</CategoryMappingAllowed>
    <Country>{_xml_escape(country)}</Country>
    <Currency>{_xml_escape(currency)}</Currency>
    <ConditionID>{_xml_escape(condition_id)}</ConditionID>
    <DispatchTimeMax>3</DispatchTimeMax>
    <ListingDuration>GTC</ListingDuration>
    <ListingType>FixedPriceItem</ListingType>
    <PaymentMethods>CreditCard</PaymentMethods>
    <PostalCode>{_xml_escape(postal_code)}</PostalCode>
    <Quantity>{int(quantity)}</Quantity>
{picture_xml}    <ReturnPolicy>
      <ReturnsAcceptedOption>ReturnsAccepted</ReturnsAcceptedOption>
      <RefundOption>MoneyBack</RefundOption>
      <ReturnsWithinOption>Days_30</ReturnsWithinOption>
      <ShippingCostPaidByOption>Buyer</ShippingCostPaidByOption>
    </ReturnPolicy>
    <ShippingDetails>
      <ShippingType>Flat</ShippingType>
      <ShippingServiceOptions>
        <ShippingServicePriority>1</ShippingServicePriority>
        <ShippingService>USPSMedia</ShippingService>
        <ShippingServiceCost currencyID="{_xml_escape(currency)}">4.99</ShippingServiceCost>
      </ShippingServiceOptions>
    </ShippingDetails>
    <Site>US</Site>
{specifics_xml}  </Item>
</{call_name}Request>
"""


_TAG_RX = re.compile(r"<([A-Za-z:]+)[^>]*>(.*?)</\1>", re.DOTALL)


def _quick_tag(xml: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_errors(xml: str) -> list[dict[str, str]]:
    errs: list[dict[str, str]] = []
    for m in re.finditer(r"<Errors>(.*?)</Errors>", xml, re.DOTALL):
        block = m.group(1)
        errs.append(
            {
                "severity": _quick_tag(block, "SeverityCode") or "",
                "code": _quick_tag(block, "ErrorCode") or "",
                "short": _quick_tag(block, "ShortMessage") or "",
                "long": _quick_tag(block, "LongMessage") or "",
            }
        )
    return errs


def _sandbox_listing_url(item_id: str | None) -> str | None:
    if not item_id:
        return None
    return f"https://www.sandbox.ebay.com/itm/{item_id}"


async def publish_listing(
    draft: ListingDraft,
    *,
    app_id: str,
    dev_id: str,
    cert_id: str,
    user_token: str,
    site_id: str = "0",
    category_id: str = "15724",
    country: str = "US",
    postal_code: str = "95125",
    condition_id: str = "1000",
    dry_run: bool = True,
    sandbox: bool = True,
) -> PublishResponse:
    """Call the eBay Trading API to verify (dry_run=True) or create (dry_run=False).

    The default condition_id can be overridden by the caller; the listing
    draft's currency is force-coerced to USD when site=0 (US) — see the note
    in func_test/test_ebay_sandbox_listing.py for why.
    """
    if not all([app_id, dev_id, cert_id, user_token]):
        return PublishResponse(
            ack="Failure",
            errors=[
                {
                    "severity": "Error",
                    "code": "missing_credentials",
                    "short": "EBAY_APP_ID, EBAY_DEV_ID, EBAY_CERT_ID, EBAY_USER_TOKEN must all be set.",
                }
            ],
        )

    endpoint = SANDBOX_ENDPOINT if sandbox else PRODUCTION_ENDPOINT
    if not sandbox and not dry_run:
        return PublishResponse(
            ack="Failure",
            errors=[
                {
                    "severity": "Error",
                    "code": "production_publish_blocked",
                    "short": "Refusing to publish to eBay production from the hackathon backend.",
                }
            ],
        )

    call_name = "AddFixedPriceItem" if not dry_run else "VerifyAddFixedPriceItem"

    # Currency / site coercion — see func_test/test_ebay_sandbox_listing.py.
    currency = (draft.currency or "USD").upper()
    if str(site_id) == "0" and currency != "USD":
        currency = "USD"

    xml_body = _build_item_xml(
        call_name=call_name,
        token=user_token,
        title=draft.title,
        description=draft.description,
        price=float(draft.suggested_price),
        currency=currency,
        country=country,
        postal_code=postal_code,
        category_id=category_id,
        condition_id=condition_id,
        quantity=1,
        item_specifics=draft.ebay_item_specifics or None,
        picture_url=draft.image_url or None,
    )

    headers = {
        "X-EBAY-API-COMPATIBILITY-LEVEL": COMPATIBILITY_LEVEL,
        "X-EBAY-API-CALL-NAME": call_name,
        "X-EBAY-API-SITEID": str(site_id),
        "X-EBAY-API-APP-NAME": app_id,
        "X-EBAY-API-DEV-NAME": dev_id,
        "X-EBAY-API-CERT-NAME": cert_id,
        "Content-Type": "text/xml",
    }

    async with httpx.AsyncClient(
        timeout=30.0, http2=False, headers={"User-Agent": "FrankenFit-hackathon/0.1"}
    ) as client:
        # eBay sandbox edge occasionally drops connections (h11 RemoteProtocolError).
        # One quick retry handles it; see func_test/test_ebay_sandbox_listing.py.
        last_err: Exception | None = None
        body = ""
        status = 0
        for attempt in range(3):
            try:
                r = await client.post(endpoint, headers=headers, content=xml_body)
                body = r.text
                status = r.status_code
                last_err = None
                break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
                last_err = e
                continue
        if last_err and not body:
            return PublishResponse(
                ack="Failure",
                errors=[
                    {
                        "severity": "Error",
                        "code": "transport",
                        "short": f"{type(last_err).__name__}: {last_err}",
                    }
                ],
            )

    ack = _quick_tag(body, "Ack") or "Failure"
    item_id = _quick_tag(body, "ItemID")
    errors = _extract_errors(body)

    # Drop pure-Warning entries when the listing succeeded — they are eBay
    # deprecation notices (e.g. collectibles grading aspects) that don't
    # affect the listing and would confuse the frontend into showing an error.
    success = ack in ("Success", "Warning") and bool(item_id)
    visible_errors = (
        [e for e in errors if e.get("severity", "").lower() != "warning"]
        if success else errors
    )

    return PublishResponse(
        ack=ack,
        item_id=item_id if success else None,
        errors=[{k: str(v) for k, v in e.items()} for e in visible_errors],
        sandbox_url=_sandbox_listing_url(item_id) if (sandbox and success) else None,
    )
