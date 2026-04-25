#!/usr/bin/env python3
"""
Placeholder: Entire (agent-human platform) — wire when docs + endpoint are known.
Set ENTIRE_API_KEY and ENTIRE_API_BASE if applicable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_api._env import load_frankenfit_env  # noqa: E402

load_frankenfit_env()


def main() -> int:
    key = os.environ.get("ENTIRE_API_KEY")
    base = os.environ.get("ENTIRE_API_BASE")
    if not key or not base:
        print(
            "Skip: ENTIRE_API_KEY / ENTIRE_API_BASE not set.\n"
            "  Add keys from hackathon pack, then implement a minimal agent session ping."
        )
        return 0

    print(
        "Keys present — replace this script with a real Entire request per official docs.\n"
        f"  Base: {base}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
