"""
Tavily service stub.

Implement with the tavily-python SDK (already tested in func_test/passed/test_tavily_price_comps.py).
"""

from __future__ import annotations

from backend.app.models import GarmentDescription, PriceBand


async def fetch_resale_comps(
    garment: GarmentDescription,
    *,
    api_key: str,
    num_results: int = 10,
    currency: str = "USD",
) -> PriceBand:
    """
    Search Tavily for recent resale prices of this garment.

    Query shape (implement on hackathon day):
      "{garment.title} {garment.brand} secondhand resale price site:vinted.com OR site:ebay.com OR site:depop.com"

    Aggregate the scraped prices into a PriceBand:
      - min   = lowest found listing price
      - median = statistical median
      - suggested = p75 (or Gemini-generated if time permits)
      - max  = highest found listing price
      - sources = list of URLs Tavily returned

    See func_test/passed/test_tavily_price_comps.py for the reference implementation.
    """
    raise NotImplementedError("implement: Tavily search → PriceBand")
