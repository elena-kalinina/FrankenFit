#!/usr/bin/env python3
"""
End-to-end demo orchestrator + cache primer.

Runs the full hackathon flow against the live backend (uvicorn on :8000):

  /v1/wardrobe/analyze       — upload 5 garment photos
  /v1/wardrobe/swipe         — like 2, dislike 3 (taste signal → live JSONL)
  /v1/upcycle/generate       — FLUX.2 multi-reference edit on the 3 dislikes
  /v1/upcycle/animate        — Hailuo I2V → primes /static/video/{sid}.mp4
                               + promotes to upcycle_hero.mp4 generic fallback
  /v1/listings/draft         — Tavily comps + Gemini copy on the upcycle
  /v1/listings/publish       — eBay sandbox VerifyAddFixedPriceItem (dry-run)
  /v1/preferences/classify   — Pioneer side-by-side (best-effort, last)

Each step's response is saved to func_test/out/demo_e2e/ for fallback JSON
generation (Phase 3 of the hackathon plan) and for backup recording artefacts.

Usage:
  PYTHONPATH=. python scripts/prime_demo_cache.py
  PYTHONPATH=. python scripts/prime_demo_cache.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
# Demo wardrobe lives in func_test/assets/latest — these are the actual photos
# the operator hand-picked for the keynote (4 keepers + 3 hates, the "perfect
# franken-bin" recipe from docs/demo_garment_dataset.md).
ASSETS_DIR = REPO_ROOT / "func_test" / "assets" / "latest"
OUT_DIR = REPO_ROOT / "func_test" / "out" / "demo_e2e"

# 4 likes / 3 dislikes — matches the dataset-guide recipe: enough loves for a
# satisfying "keepers" reveal, exactly 3 hates so the upcycle has 3 distinct
# colours/silhouettes/eras to franken-stitch into one editorial piece.
DEMO_PHOTOS = [
    ("black_turtleneck.avif", "like"),
    ("blazer.webp", "like"),
    ("leather_jacket.avif", "like"),
    ("designer_piece.jpg", "like"),
    ("athleisure.webp", "dislike"),
    ("boho_trend_chaser.webp", "dislike"),
    ("y2k_top.avif", "dislike"),
]


# Common file-suffix → IANA mime mapping. fastapi/starlette is forgiving but
# we hand the bytes to Gemini Vision via UploadFile.content_type, and Gemini
# rejects non-standard subtypes (e.g. "image/jpg" instead of "image/jpeg").
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


def _mime_for(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), f"image/{path.suffix.lstrip('.').lower()}")


def _save(name: str, payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) if not isinstance(payload, str) else payload,
        encoding="utf-8",
    )


def step_analyze(client: httpx.Client, base: str) -> dict:
    print(f"\n== /v1/wardrobe/analyze ({len(DEMO_PHOTOS)} photos, parallel Gemini Vision) ==")
    files = []
    for name, _ in DEMO_PHOTOS:
        path = ASSETS_DIR / name
        if not path.is_file():
            sys.exit(f"FAIL: missing photo {path}")
        files.append(("images", (path.name, path.read_bytes(), _mime_for(path))))
    t0 = time.time()
    r = client.post(f"{base}/v1/wardrobe/analyze", files=files, timeout=120.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    body = r.json()
    print(f"  ok in {elapsed:.1f}s  session_id={body['session_id']}  garments={len(body['garments'])}")
    for g in body["garments"]:
        print(f"    - {g['title'][:60]}  | {g['category']} / {g['style']}  | ${g['suggested_price']}")
        print(f"      roast: {g.get('roast_line', '')[:90]}")
    _save("01_analyze.json", body)
    return body


def step_swipe(client: httpx.Client, base: str, session_id: str, garments: list) -> tuple[list[str], list[str]]:
    n_like = sum(1 for _, d in DEMO_PHOTOS if d == "like")
    n_dis = sum(1 for _, d in DEMO_PHOTOS if d == "dislike")
    print(f"\n== /v1/wardrobe/swipe ({n_like} likes, {n_dis} dislikes) ==")
    likes, dislikes = [], []
    for garment, (_, direction) in zip(garments, DEMO_PHOTOS, strict=True):
        body = {
            "session_id": session_id,
            "garment_id": garment["garment_id"],
            "direction": direction,
        }
        r = client.post(f"{base}/v1/wardrobe/swipe", json=body, timeout=15.0)
        r.raise_for_status()
        out = r.json()
        bucket = likes if direction == "like" else dislikes
        bucket.append(garment["garment_id"])
        flag = "👍" if direction == "like" else "👎"
        signal = "→ jsonl" if out.get("taste_signal_appended") else ""
        print(f"  {flag} {garment['title'][:55]:<55} {signal}")
    print(f"  totals: keepers={len(likes)}  franken_bin={len(dislikes)}")
    _save("02_swipes.json", {"likes": likes, "dislikes": dislikes})
    return likes, dislikes


def step_upcycle_generate(client: httpx.Client, base: str, session_id: str, dislikes: list[str]) -> dict:
    print(f"\n== /v1/upcycle/generate (FLUX.2 flex on {len(dislikes)} garments — usually 60-180s) ==")
    body = {"session_id": session_id, "garment_ids": dislikes, "style_prompt": ""}
    t0 = time.time()
    r = client.post(f"{base}/v1/upcycle/generate", json=body, timeout=300.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    out = r.json()
    print(f"  ok in {elapsed:.1f}s  garment_id={out['garment_id']}")
    print(f"  image_url: {out['image_url']}")
    print(f"  revised_prompt: {(out.get('revised_prompt') or '')[:120]}")
    _save("03_upcycle_generate.json", out)
    return out


def step_upcycle_animate(client: httpx.Client, base: str, session_id: str, image_url: str) -> dict:
    print(f"\n== /v1/upcycle/animate (Hailuo I2V — usually 30-90s) ==")
    body = {
        "session_id": session_id,
        "image_url": image_url,
        "model": "hailuo",
        "duration": 6,
        "aspect_ratio": "9:16",
        "timeout_seconds": 180.0,
    }
    t0 = time.time()
    r = client.post(f"{base}/v1/upcycle/animate", json=body, timeout=240.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    out = r.json()
    print(f"  ok in {elapsed:.1f}s  cached={out['cached']}  video_url={out['video_url']}")
    _save("04_upcycle_animate.json", out)
    return out


def step_listing_draft(client: httpx.Client, base: str, session_id: str, garment_id: str) -> dict:
    print("\n== /v1/listings/draft (Tavily comps + Gemini copy) ==")
    body = {
        "session_id": session_id,
        "garment_id": garment_id,
        "marketplace": "ebay",
        "run_tavily": True,
    }
    t0 = time.time()
    r = client.post(f"{base}/v1/listings/draft", json=body, timeout=120.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    out = r.json()
    pb = out.get("price_band") or {}
    draft = out.get("draft") or {}
    print(f"  ok in {elapsed:.1f}s")
    print(f"  title: {draft.get('title')}")
    print(f"  price band ({pb.get('currency')}): "
          f"min={pb.get('min')} median={pb.get('median')} suggested={pb.get('suggested')} max={pb.get('max')}")
    print(f"  marketplace_copies: {[c.get('platform') for c in draft.get('marketplace_copies', [])]}")
    _save("05_listing_draft.json", out)
    return out


def step_listing_publish(client: httpx.Client, base: str, session_id: str, garment_id: str, dry_run: bool = True) -> dict:
    label = "VerifyAddFixedPriceItem (dry-run)" if dry_run else "AddFixedPriceItem (LIVE SANDBOX)"
    print(f"\n== /v1/listings/publish [{label}] ==")
    body = {"session_id": session_id, "garment_id": garment_id, "dry_run": dry_run}
    t0 = time.time()
    r = client.post(f"{base}/v1/listings/publish", json=body, timeout=60.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    out = r.json()
    print(f"  ok in {elapsed:.1f}s  ack={out.get('ack')}  item_id={out.get('item_id')}  errors={len(out.get('errors') or [])}")
    if out.get("sandbox_url"):
        print(f"  sandbox_url: {out['sandbox_url']}")
    for err in (out.get("errors") or [])[:3]:
        print(f"    {err.get('severity')} {err.get('code')}: {err.get('short')[:100]}")
    suffix = "publish" if not dry_run else "verify"
    _save(f"06_listing_{suffix}.json", out)
    return out


def step_classify(client: httpx.Client, base: str, session_id: str, garment_id: str, text: str) -> dict | None:
    print("\n== /v1/preferences/classify (Pioneer side-by-side, best-effort) ==")
    body = {
        "session_id": session_id,
        "garment_id": garment_id,
        "text": text,
        "use_trained_model": True,
    }
    try:
        t0 = time.time()
        r = client.post(f"{base}/v1/preferences/classify", json=body, timeout=45.0)
        elapsed = time.time() - t0
        r.raise_for_status()
        out = r.json()
        print(f"  ok in {elapsed:.1f}s  trained={out.get('label')} ({out.get('confidence')})  "
              f"baseline={out.get('baseline_label')} ({out.get('baseline_confidence')})  "
              f"model={out.get('model_id')}")
        _save("07_classify.json", out)
        return out
    except Exception as exc:
        print(f"  WARN: classify failed: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--skip-classify", action="store_true", help="Skip the Pioneer step.")
    parser.add_argument("--skip-publish", action="store_true", help="Skip the eBay verify call.")
    parser.add_argument("--skip-fal", action="store_true",
                        help="Skip fal.ai upcycle/generate + upcycle/animate (no FLUX.2/Hailuo cost). "
                             "Listing draft is then issued against the first liked garment instead of the upcycle.")
    parser.add_argument("--publish-live", action="store_true",
                        help="Run eBay AddFixedPriceItem against the sandbox (creates a real test listing).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Demo orchestrator → {args.base}")
    print(f"Saving responses to: {OUT_DIR.relative_to(REPO_ROOT)}/")

    with httpx.Client() as client:
        analyze = step_analyze(client, args.base)
        session_id = analyze["session_id"]
        garments = analyze["garments"]

        likes, dislikes = step_swipe(client, args.base, session_id, garments)
        if not dislikes:
            sys.exit("FAIL: no dislikes to feed upcycle.")

        if args.skip_fal:
            print("\n== fal.ai steps skipped (--skip-fal) — using first liked garment for listing draft ==")
            if not likes:
                sys.exit("FAIL: --skip-fal requires at least one LIKE; none in this run.")
            listing_garment_id = likes[0]
        else:
            upcycle = step_upcycle_generate(client, args.base, session_id, dislikes)
            listing_garment_id = upcycle["garment_id"]
            upcycle_image_url = upcycle["image_url"]
            step_upcycle_animate(client, args.base, session_id, upcycle_image_url)

        step_listing_draft(client, args.base, session_id, listing_garment_id)

        if not args.skip_publish:
            step_listing_publish(client, args.base, session_id, listing_garment_id, dry_run=not args.publish_live)

        if not args.skip_classify:
            # Use a probe known to DIVERGE between Qwen baseline and the LoRA
            # (Qwen overconfidently says love; the trained model says meh).
            # That's the exact "8B-param naive vs 109M-param taught-your-taste"
            # demo narrative, and it ships from a primed cache rather than
            # depending on the upcycle text where both models agreed today.
            probe = "Y2K bedazzled bell sleeve top with rhinestones, low rise"
            step_classify(client, args.base, session_id, listing_garment_id, probe)

    print("\n✅ Demo cache primed. Artefacts under:", OUT_DIR.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
