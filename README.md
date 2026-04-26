# Franken-Fit

Hackathon project — a sustainable second-hand fashion app. The user uploads photos of clothes they don't wear, a sarcastic stylist roasts each piece, the user swipes LIKE / DISLIKE, and the rejected items are recombined ("Franken-fitted") into a single new outfit using image-to-image plus image-to-video generation. Liked items get marketplace-ready resale listings (eBay Sandbox).

This repo is the **FastAPI backend** plus pre-rendered demo assets. The frontend is built separately in [Lovable](https://lovable.dev) and points at this backend over CORS — see *Frontend* below.

---

## What it integrates

| Provider | Purpose | Where |
|---|---|---|
| **Google Gemini** | Vision (per-photo garment JSON + roast line + stylist tip), text generation (listing copy, upcycle prompt), TTS for the live roast voiceover | `backend/app/services/gemini.py` |
| **Tavily** | Live resale price comps → min/median/suggested/max price band | `backend/app/services/tavily.py` |
| **fal.ai** | FLUX.2 Flex multi-reference edit → upcycle hero still; Hailuo I2V → upcycle MP4 | `backend/app/services/fal.py` |
| **Pioneer (Fastino)** | Side-by-side preference classification: a baseline (Qwen via OpenAI-compatible chat or `fastino/gliner2-base-v1`) versus a LoRA-fine-tuned GLiNER 2 trained on the user's own swipes | `backend/app/services/pioneer.py` |
| **eBay Sandbox** | Verify + publish resale listing via the Trading API | `backend/app/services/ebay.py` |

Resilience built in (because this runs on stage):

- Multi-stage **model fallback chains** for Gemini Vision and Gemini TTS — the primary preview model is tried first, then two secondaries, finally a stable GA model.
- Decoder vs encoder **routing** in Pioneer — Qwen-class baselines go to `/v1/chat/completions`; GLiNER-class go to `/inference`. Per-call timeouts.
- **Local mirroring** of every generated asset (upcycle hero JPG, upcycle MP4, per-garment TTS) under `backend/static/` so the demo never depends on a live CDN URL during the show.
- **Canonical fallback assets** (`backend/static/{tts/cinematic, upcycle/upcycle_hero.jpg, video/upcycle_hero.mp4, fallbacks/*.json}`) shipped with the repo so the frontend can render a complete demo even with the network unplugged.

---

## Layout

| Path | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI entry point; CORS, `/static` mount, env load (`.env` is authoritative via `load_dotenv(override=True)`) |
| `backend/app/routers/` | One router per endpoint group: `wardrobe`, `listings`, `upcycle`, `preferences`, `health` |
| `backend/app/services/` | One module per external provider; all blocking SDK calls run on `asyncio.to_thread` |
| `backend/app/models.py` | Pydantic request/response contracts — the source of truth the frontend reads from |
| `backend/app/session.py` | In-memory `DemoSession` store (single-node demo; no DB) |
| `backend/app/services/cache.py` | Static-asset path helpers + cinematic clip sync |
| `backend/static/fallbacks/` | Frontend offline-fallback JSONs (5 files, one per major endpoint) |
| `backend/static/tts/cinematic/` | 4 pre-rendered cinematic voice lines (`cold_open`, `upcycle_reveal`, `rejected_upcycle`, `resale_cheer`) |
| `backend/static/{upcycle,video}/upcycle_hero.{jpg,mp4}` | Canonical fallback hero still and animated reveal — used when fal.ai is slow or offline |
| `scripts/prime_demo_cache.py` | End-to-end orchestrator that hits every endpoint and primes the local caches |
| `scripts/pioneer/train.py` | Pioneer Day-1→Day-2 fine-tune loop (dataset → train → poll → eval → infer) |
| `scripts/pioneer/probe.py` | Quick side-by-side inference against any two Pioneer models |
| `docs/demo_garment_dataset.md` | How to prepare the 4-likes / 3-dislikes demo photoset for the best taste signal |
| `test_api/` | Lightweight provider connectivity smoke tests (Gemini, Tavily, Pioneer, end-to-end) |

---

## Quick start

```bash
git clone <this repo>
cd FrankenFit

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in: GEMINI_API_KEY, TAVILY_API_KEY, FAL_KEY, PIONEER_API_KEY,
#          PIONEER_TRAINED_MODEL_ID, EBAY_* (see .env.example for all)

uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the live OpenAPI spec.

The startup banner prints the last 6 chars of every loaded API key — use this to confirm `.env` is the source of truth (and not a stale shell export).

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/wardrobe/analyze` | Upload N photos (multipart `images[]`) → garment JSON + image URL + async TTS render per garment |
| `POST` | `/v1/wardrobe/swipe` | Record LIKE / DISLIKE; appends to `live_swipes.jsonl` for Pioneer training |
| `POST` | `/v1/upcycle/generate` | Combine 1–5 disliked garments into one upcycled hero still (fal FLUX.2 Flex). Mirrored locally to `/static/upcycle/<gid>.jpg` |
| `POST` | `/v1/upcycle/animate` | Image-to-video on the hero still (Hailuo I2V). 60s timeout → cached fallback served. |
| `POST` | `/v1/listings/draft` | Tavily price band + Gemini-generated marketplace copy (eBay / Vinted / Depop) |
| `POST` | `/v1/listings/publish` | Verify or publish to eBay Sandbox (Trading API XML) |
| `POST` | `/v1/preferences/classify` | Pioneer side-by-side: baseline (Qwen or GLiNER) vs trained LoRA |
| `GET`  | `/v1/health` | Readiness probe |
| `GET`  | `/static/...` | Uploaded photos, generated upcycle stills + videos, garment TTS, cinematic clips, fallback JSONs |

The exact request/response shapes are in `backend/app/models.py`. Paste them straight into the frontend prompts; do not rely on the OpenAPI spec for the Pydantic field descriptions — those describe behavior the spec doesn't capture (e.g. *"empty until the async render finishes; silently 404-skip"*).

---

## Pioneer — Day-1 → Day-2 fine-tune loop

The `scripts/pioneer/` module implements the hackathon's most compelling pitch beat: the app collects real taste signals during Day-1 usage (every swipe writes a row to `func_test/out/live_swipes.jsonl`), then overnight a LoRA fine-tune trains a task-specific binary classifier on those rows. Day-2 shows Qwen (7B, generic) vs fine-tuned GLiNER 2 (compact, your taste) side-by-side on the same garment.

### Classifier design

- **Binary labels** — `love` / `hate`, matching the swipe UX exactly. No `meh` class: users can never produce a neutral signal, so training on it only blurs the decision boundary.
- **Base model**: `fastino/gliner2-base-v1` (encoder, task-specific, ~10× smaller than Qwen).
- **Training type**: LoRA, 5 epochs, lr=5e-5. Typically deploys in ~2–5 min on Pioneer.

### Seed dataset

Three sources are merged, normalised, and deduped automatically:

| File | Rows | Notes |
|---|---|---|
| `func_test/out/preference-training-data.jsonl` | 50 | Hand-curated binary labels |
| `func_test/out/pioneer_style_dataset.jsonl` | 10 | Editorial seed rows |
| `func_test/out/live_swipes.jsonl` | grows | Auto-collected from `POST /v1/wardrobe/swipe` |

### Running the loop

```bash
# Full pipeline — dataset → train → poll → eval → side-by-side infer
# (use --skip-generate to skip Pioneer's /generate and go straight to training)
python -m scripts.pioneer.train --phase=all --skip-generate --dataset-name=frankenfit-binary-v3

# Or phase-by-phase:
python -m scripts.pioneer.train --phase=train  --dataset-name=frankenfit-binary-v3
python -m scripts.pioneer.train --phase=poll
python -m scripts.pioneer.train --phase=eval
python -m scripts.pioneer.train --phase=infer  --base-model=$PIONEER_QWEN_MODEL

# Quick side-by-side probe (no training needed):
python -m scripts.pioneer.probe "Y2K bedazzled bell sleeve top with rhinestones"
```

After training, set `PIONEER_TRAINED_MODEL_ID=<training_job_id>` in `.env`. The backend's `/v1/preferences/classify` endpoint will route the new model automatically.

Job metadata and local JSONL snapshots are saved to `scripts/pioneer/out/` (gitignored).

---

## Priming the demo cache

The orchestrator script runs the full path once end-to-end against a local server, populates `backend/static/uploads/`, `backend/static/tts/garment/`, `backend/static/upcycle/`, and `backend/static/video/`, then promotes the latest upcycle hero JPG and MP4 to the canonical fallback names.

```bash
# server already running on :8000
PYTHONPATH=. python -u scripts/prime_demo_cache.py

# skip fal.ai (cheap iterations):
PYTHONPATH=. python -u scripts/prime_demo_cache.py --skip-fal
```

The demo photo set lives in `func_test/assets/latest/` (which is **gitignored** — bring your own; see `docs/demo_garment_dataset.md` for the recipe).

---

## Frontend

The frontend is **not** in this repo. It is built in [Lovable](https://lovable.dev) and consumes this backend's HTTP+static surface. The integration contract is:

1. CORS is wide-open (`allow_origins=["*"]`) — point the Lovable preview straight at `http://127.0.0.1:8000`.
2. All URL fields returned by the backend (`image_url`, `tts_url`, upcycle `image_url`, animate `video_url`) are **relative** — the frontend prepends `VITE_API_BASE_URL`.
3. Cinematic TTS is `.wav`, served at `/static/tts/cinematic/{cold_open,upcycle_reveal,rejected_upcycle,resale_cheer}.wav`. Per-garment live TTS is also `.wav` at `/static/tts/garment/<garment_id>.wav` and is rendered async — the frontend should silently 404-skip until it lands.
4. The orchestrator and `backend/static/fallbacks/*.json` give the frontend a complete cached demo to fall back to if any external API misbehaves on stage.

---

## API smoke tests

`test_api/` contains standalone provider checks:

```bash
python -m test_api.test_gemini      # Gemini Vision + TTS
python -m test_api.test_tavily      # Tavily price comps
python -m test_api.test_pioneer     # Pioneer base + trained
python -m test_api.test_entire      # End-to-end one-garment dry run
```

These are *connectivity* tests — they verify keys, quotas, and routing, not business logic.

---

## Notes

- **Single-node, in-memory session.** `DemoSession` lives in process. Restart the server, sessions vanish. Fine for a demo; not for production.
- **TTS rate limits.** Each Gemini preview TTS model has its own daily quota pool (~10/day on free tier). The TTS fallback chain rotates across `gemini-2.5-flash-preview-tts`, `gemini-3.1-flash-tts-preview`, and `gemini-2.5-pro-preview-tts` — together they give effectively 3× the per-day capacity.
- **fal.ai is metered.** The `--skip-fal` flag in the orchestrator exists for exactly this reason during dev.
- **eBay Sandbox.** Set `EBAY_USER_TOKEN` to a valid sandbox user token; we publish to category `15724` (Women's Tops) by default. Adjust in `.env`.
