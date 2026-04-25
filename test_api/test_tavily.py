#!/usr/bin/env python3
"""
Smoke test: Tavily search API (listing comps / web grounding).
Requires TAVILY_API_KEY in FrankenFit/.env
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_api._env import load_frankenfit_env  # noqa: E402

load_frankenfit_env()


def main() -> int:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        print("Skip: TAVILY_API_KEY not set (add to FrankenFit/.env)")
        return 0

    import httpx

    payload = {
        "api_key": key,
        "query": "Zara mustard turtleneck sweater resale price Vinted",
        "search_depth": "basic",
        "max_results": 3,
    }
    r = httpx.post("https://api.tavily.com/search", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    print(json.dumps(data, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        print("FAIL:", e, file=sys.stderr)
        raise SystemExit(1)
