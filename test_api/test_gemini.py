#!/usr/bin/env python3
"""
Smoke test: Gemini text + optional 1x1 PNG multimodal ping.
Requires GEMINI_API_KEY in FrankenFit/.env
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root on path for `import test_api.*`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_api._env import load_frankenfit_env  # noqa: E402

load_frankenfit_env()

MINIMAL_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_TEST_MODEL", "gemini-3.1-flash-lite-preview"),
        help="Model id (override if your project uses another)",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Only run text ping",
    )
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Skip: GEMINI_API_KEY not set (add to FrankenFit/.env)")
        return 0

    import google.generativeai as genai

    genai.configure(api_key=key)
    model = genai.GenerativeModel(args.model)

    t = model.generate_content("Reply with exactly: pong")
    print("Text:", getattr(t, "text", t))

    if args.skip_image:
        return 0

    t2 = model.generate_content(
        [
            "Describe this image in one short phrase (it is a tiny test pattern).",
            {"mime_type": "image/png", "data": MINIMAL_PNG_1X1},
        ]
    )
    print("Image:", getattr(t2, "text", t2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001 — smoke test
        err = str(e).lower()
        if "429" in err or "quota" in err or "resource exhausted" in err:
            print(
                "WARN: API key likely works but quota/rate limit blocked the call. Retry later.\n",
                e,
                file=sys.stderr,
            )
            raise SystemExit(0)
        print("FAIL:", e, file=sys.stderr)
        raise SystemExit(1)
