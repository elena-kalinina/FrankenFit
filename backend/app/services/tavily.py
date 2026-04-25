"""
Tavily service — second-hand price comps for a garment.

Pipeline (ported from func_test/passed/test_tavily_price_comps.py):
  1. Build a targeted query from the garment metadata.
  2. POST https://api.tavily.com/search  (advanced).
  3. Extract numeric price candidates from result snippets (€ / £ / $).
  4. Aggregate to a PriceBand (min, median, suggested=p75-ish, max).
"""

from __future__ import annotations

import asyncio
import re
import statistics
from typing import Any

import httpx

from backend.app.models import GarmentDescription, PriceBand

PRICE_RE = re.compile(
    r"(?P<cur>[€£$])\s?(?P<amount>\d{1,4}(?:[.,]\d{1,2})?)|"
    r"(?P<amount2>\d{1,4}(?:[.,]\d{1,2})?)\s?(?P<cur2>EUR|GBP|USD)",
    re.IGNORECASE,
)
CURRENCY_NORMAL = {
    "€": "EUR", "£": "GBP", "$": "USD",
    "eur": "EUR", "gbp": "GBP", "usd": "USD",
}

MARKETPLACE_HINTS = {
    "vinted": "site:vinted.com OR site:vinted.fr OR site:vinted.de OR site:vinted.co.uk",
    "depop": "site:depop.com",
    "ebay": "site:ebay.com OR site:ebay.co.uk OR site:ebay.de",
    "any": "",
}


def _query_for(garment: GarmentDescription, marketplace: str = "any", region: str = "eu") -> str:
    bits = [
        garment.brand,
        garment.category,
        garment.style,
        garment.color,
        garment.material,
        "second hand resale price",
        "used preloved",
    ]
    core = " ".join(str(x) for x in bits if x and str(x).lower() != "unbranded")
    region_hint = {"eu": "EU Europe", "uk": "UK", "us": "US", "any": ""}.get(region, "")
    mp_hint = MARKETPLACE_HINTS.get(marketplace, "")
    return " ".join(x for x in [core, region_hint, mp_hint] if x)


def _extract_prices(snippet: str) -> list[dict[str, Any]]:
    prices: list[dict[str, Any]] = []
    for m in PRICE_RE.finditer(snippet or ""):
        cur = (m.group("cur") or m.group("cur2") or "").lower()
        amt = (m.group("amount") or m.group("amount2") or "").replace(",", ".")
        try:
            val = float(amt)
        except ValueError:
            continue
        if val < 1 or val > 10000:
            continue
        prices.append({"currency": CURRENCY_NORMAL.get(cur, cur.upper()), "amount": val})
    return prices


def _suggested_price(values: list[float]) -> float:
    """Approximate p75 — the suggested ask for a quick second-hand sale."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_v = sorted(values)
    idx = max(0, min(len(sorted_v) - 1, int(round(0.75 * (len(sorted_v) - 1)))))
    return sorted_v[idx]


async def fetch_resale_comps(
    garment: GarmentDescription,
    *,
    api_key: str,
    num_results: int = 8,
    marketplace: str = "any",
    region: str = "eu",
    currency: str = "USD",
) -> PriceBand:
    """Search Tavily for resale comps and return a PriceBand.

    On the demo path we use the dominant-currency cluster from the result set
    (e.g. EUR if most comps are European). If the requested ``currency`` has
    enough hits, we prefer it; otherwise we fall back to the dominant one and
    return that — the frontend renders the symbol from the response.
    """
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY missing")

    query = _query_for(garment, marketplace=marketplace, region=region)
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max(3, min(10, num_results)),
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post("https://api.tavily.com/search", json=payload)
    r.raise_for_status()
    body = r.json()

    candidates: list[dict[str, Any]] = []
    sources: list[str] = []
    for res in body.get("results", []) or []:
        url = res.get("url")
        if url and url not in sources:
            sources.append(url)
        text = " ".join([res.get("title") or "", res.get("content") or ""])
        for p in _extract_prices(text):
            candidates.append({**p, "source": url})

    by_cur: dict[str, list[float]] = {}
    for c in candidates:
        by_cur.setdefault(c["currency"], []).append(float(c["amount"]))

    chosen_currency = currency
    if currency not in by_cur or len(by_cur[currency]) < 2:
        if by_cur:
            chosen_currency = max(by_cur.items(), key=lambda kv: len(kv[1]))[0]
    values = by_cur.get(chosen_currency, []) or [garment.suggested_price]

    band = PriceBand(
        min=round(min(values), 2),
        median=round(statistics.median(values), 2),
        suggested=round(_suggested_price(values), 2),
        max=round(max(values), 2),
        currency=chosen_currency,
        sources=sources[:6],
    )
    return band
