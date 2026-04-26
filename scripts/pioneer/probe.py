#!/usr/bin/env python3
"""
Quick side-by-side inference: baseline vs fine-tuned model.

Usage:
  python -m scripts.pioneer.probe "Y2K bedazzled bell sleeve top"
  python -m scripts.pioneer.probe --baseline=fastino/gliner2-base-v1 "Slim raw selvedge jeans"

Env:
  PIONEER_API_KEY           required
  PIONEER_TRAINED_MODEL_ID  fine-tuned model id (from scripts/pioneer/out/last_pioneer_training.json)
  PIONEER_QWEN_MODEL        Qwen model id (optional; overrides --baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env", override=True)

LABELS = ["love", "hate"]
_OUT_DIR = Path(__file__).resolve().parent / "out"
_TRAINING_META = _OUT_DIR / "last_pioneer_training.json"


def _get_trained_model_id() -> str | None:
    env = os.environ.get("PIONEER_TRAINED_MODEL_ID")
    if env:
        return env
    if _TRAINING_META.is_file():
        try:
            return json.loads(_TRAINING_META.read_text())["training_job_id"]
        except Exception:  # noqa: BLE001
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Pioneer side-by-side probe.")
    parser.add_argument("text", nargs="+", help="Garment description(s) to classify.")
    parser.add_argument(
        "--baseline",
        default=os.environ.get("PIONEER_QWEN_MODEL") or "fastino/gliner2-base-v1",
        help="Baseline model id (default: PIONEER_QWEN_MODEL or gliner2-base-v1).",
    )
    parser.add_argument(
        "--trained",
        default=_get_trained_model_id(),
        help="Fine-tuned model id (default: PIONEER_TRAINED_MODEL_ID from .env or last training).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("PIONEER_API_KEY")
    if not api_key:
        print("SKIP: PIONEER_API_KEY not set.", file=sys.stderr)
        return 0
    if not args.trained:
        print("FAIL: no trained model id. Set PIONEER_TRAINED_MODEL_ID in .env "
              "or run scripts/pioneer/train.py --phase=train first.", file=sys.stderr)
        return 1

    import httpx
    base = os.environ.get("PIONEER_API_BASE", "https://api.pioneer.ai").rstrip("/")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    texts = [" ".join(args.text)] if len(args.text) > 1 else args.text

    with httpx.Client(timeout=30.0, headers=headers) as client:
        for text in texts:
            print(f"\nProbe: {text!r}")
            for label, model_id in [("baseline", args.baseline), ("trained", args.trained)]:
                try:
                    r = client.post(f"{base}/inference", json={
                        "model_id": model_id,
                        "task": "classify_text",
                        "text": text,
                        "schema": {"categories": LABELS},
                    })
                    raw = r.json() if r.content else {}
                    res = raw.get("result") or raw.get("output") or raw
                    verdict = res.get("category") or res.get("label") or str(raw)
                    print(f"  {label:8s} [{model_id}]: {verdict}")
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:8s} [{model_id}]: ERROR — {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
