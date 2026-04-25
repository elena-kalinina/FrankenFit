#!/usr/bin/env python3
"""
Pioneer (Fastino Labs) smoke test — proves the key works + prints the base-model
catalog so we can confirm `fastino/gliner2-base-v1` is available for the
Day-1 → Day-2 fine-tune loop in `func_test/test_pioneer_finetune_loop.py`.

Docs: https://agent.pioneer.ai/docs/api-reference
Auth: X-API-Key header.
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
    key = os.environ.get("PIONEER_API_KEY")
    if not key:
        print(
            "SKIP: PIONEER_API_KEY not set.\n"
            "  Get a key and docs at https://agent.pioneer.ai/docs/api-reference",
            file=sys.stderr,
        )
        return 0

    base = os.environ.get("PIONEER_API_BASE", "https://api.pioneer.ai").rstrip("/")
    headers = {"X-API-Key": key}

    import httpx

    try:
        r = httpx.get(f"{base}/base-models", headers=headers, timeout=30.0)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: GET /base-models transport error: {e}", file=sys.stderr)
        return 1

    if r.status_code >= 300:
        print(f"FAIL: GET /base-models -> {r.status_code}: {r.text}", file=sys.stderr)
        return 1

    try:
        payload = r.json()
    except Exception:  # noqa: BLE001
        print(f"FAIL: non-JSON body: {r.text[:500]}", file=sys.stderr)
        return 1

    items = payload if isinstance(payload, list) else payload.get("models") or payload.get("data") or []
    ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("model_id") or item.get("name")
            if model_id:
                ids.append(str(model_id))

    print(f"OK: Pioneer reachable. base_models.count={len(ids)}")
    if ids:
        preview = ids[:12]
        print("  preview:", ", ".join(preview))
    if any("gliner" in m.lower() for m in ids):
        print("  gliner2-style base model present \u2713 (needed for LoRA fine-tune).")
    else:
        print("  note: no 'gliner' match in base models — check the current docs for the right base id.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001
        print("FAIL:", e, file=sys.stderr)
        raise SystemExit(1)
