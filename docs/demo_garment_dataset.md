# Demo garment dataset guide

> Phone-friendly checklist for the keynote photo shoot. Read on the way to the
> closet.

## TL;DR — the recipe

| Bucket | Count | Purpose | Vibe |
|---|---|---|---|
| **LOVES** | 4 photos | Build the "keepers" reel | One per archetype: minimal, tailored, statement, designer |
| **MEHS** | 0 photos | (skipped — only matters for live Pioneer probes) | — |
| **HATES** | 3 photos | Feed the franken-bin → upcycle MP4 | Three CLASHING eras / textures / silhouettes |

**Total: 7 photos. Hard rule: exactly 3 hates.**

Why exactly 3? The upcycle prompt says "combine N tossed pieces into one
silhouette". Two pieces feels thin. Four+ blurs into mush in the FLUX.2 edit.
Three is the visual sweet spot — the AI clearly takes a colour from one, a
silhouette from another, a fabric from the third, and you can spot all three
in the final image.

---

## Why 4 LOVES (and not 3 or 5)

- **Pacing.** The swipe reel runs 7 cards in ~25 seconds. 4 likes / 3 dislikes
  = 4 ✅ ✅ ✅ ✅ + 3 ❌ ❌ ❌. Rhythmic, no awkward 50/50 split.
- **Variety.** 4 likes covers the four canonical archetypes a stylist would
  recognize: minimal, tailored, statement, luxury. 3 likes always feels like
  one is missing; 5+ starts repeating archetypes.
- **Pioneer side-by-side.** The trained LoRA learns better with 4 positive
  signals than with 3 — overnight fine-tune output is sharper.

---

## The LOVES — pick one of each archetype

| # | Archetype | Examples (pick the one you actually own) |
|---|---|---|
| 1 | **Minimal / quiet luxury** | Black turtleneck, camel coat, white tee from APC, plain wool sweater |
| 2 | **Tailored** | Navy blazer, well-cut blazer of any colour, structured trousers |
| 3 | **Statement / unusual silhouette** | Burgundy leather jacket, oversized utility coat, asymmetric anything |
| 4 | **Designer / brand-name** | Balenciaga sneakers, anything with a visible logo a stylist would clock |

Photo guidelines for LOVES:
- Hang or lay flat on a plain background (white wall, wood floor, bedsheet).
- Full garment in frame — head-to-hem if it's tall, full edge-to-edge for tops.
- Even, soft daylight. No flash. No filters.
- 1500×1500 minimum. Phone portrait mode is fine.

---

## The HATES — pick THREE that CLASH

Important: your hates need to be **visually different from each other**. If
all three are "ugly sweaters", the upcycle just looks like a sweater. The
upcycle reveal is way more dramatic when the AI has to fuse:

- 1 with **a colour pop** (neon, bright, unusual hue)
- 1 with **a heavy texture** (embroidery, sequins, thick knit, eyelet, dip-dye)
- 1 with **a defining silhouette** (bell sleeves, asymmetric, low-rise, etc.)

### Recommended hate trio (the one we tested today)

| # | Type | Why it works |
|---|---|---|
| 1 | **Athleisure** (e.g. Adidas track pants, jersey hoodie, dip-dye sweatshirt) | Brings color contrast + sporty silhouette |
| 2 | **Boho / trend-chaser piece** (eyelet blouse, prairie skirt, embroidered cotton) | Brings delicate texture + romantic shape |
| 3 | **Y2K relic** (bell sleeves, bedazzled top, low-rise jeans, rhinestones) | Brings era contrast + decoration noise |

The upcycle prompt that came out of these three:
> "Deconstructed avant-garde gown … structured bodice from a charcoal-to-cyan
> gradient sweatshirt … architectural shoulders from white embroidered cotton
> … layers of tattered, sheer pink and black floral tie-dye chiffon …"

You can SEE all three pieces in the final render. That's the goal.

### Anti-pattern (don't do this)

❌ Three plain T-shirts in different colours
❌ Three "fast-fashion sweaters"
❌ Three pieces from the same brand or era
❌ Anything covered by another garment in the photo (jackets over shirts)

If your photos don't have at least 3 distinctly clashing items, the upcycle
will look like one slightly weird sweater.

---

## Photo logistics

- **File formats accepted by the backend:** `.jpg`, `.jpeg`, `.png`, `.webp`,
  `.avif`, `.heic`, `.heif`. Phone exports work directly — no conversion.
- **Drop-zone:** `func_test/assets/latest/`. Replace whatever's there.
- **Naming:** Use lowercase, descriptive names — `black_turtleneck.avif`,
  `boho_trend_chaser.webp`, `y2k_top.avif`. The orchestrator script
  (`scripts/prime_demo_cache.py`) hard-codes filenames in `DEMO_PHOTOS`, so
  if you rename, update the list there too.
- **Re-prime the cache after the shoot:**

  ```bash
  PYTHONPATH=. python scripts/prime_demo_cache.py
  ```

  This runs the full pipeline (analyze → swipe → upcycle → animate → listing
  draft → publish dry-run → classify) against the new photos and refreshes
  `func_test/out/demo_e2e/*.json`. Total time: ~3 minutes.

---

## Self-check before walking on stage

- [ ] Exactly 7 files in `func_test/assets/latest/` (no `.DS_Store` extras)
- [ ] At least one file > 200 KB (so it's clearly not a thumbnail)
- [ ] All three hates are visually distinct (colour / texture / silhouette)
- [ ] You ran the orchestrator successfully end-to-end at least once
- [ ] `backend/static/video/upcycle_hero.mp4` is the file from the latest run
- [ ] `backend/static/upcycle/upcycle_hero.jpg` exists (cached fal still)
- [ ] `backend/static/tts/garment/` has 7 fresh `.wav` files matching the
      garment IDs in `func_test/out/demo_e2e/01_analyze.json`
- [ ] `backend/static/tts/cinematic/` has all 4 `.wav` files
      (`cold_open`, `upcycle_reveal`, `rejected_upcycle`, `resale_cheer`)

If any of these fail, re-run the orchestrator. It's idempotent.
