import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Load .env into os.environ BEFORE anything else imports vendor SDKs.
# pydantic-settings reads .env into our typed Settings, but third-party SDKs
# (fal_client, google-genai, etc.) read directly from os.environ — they need
# this explicit load to see FAL_KEY, GEMINI_API_KEY, PIONEER_API_KEY, etc.
#
# override=True is intentional: during the hackathon the operator edits .env
# and restarts uvicorn expecting the new key to win, but the parent shell
# may have stale exports from earlier sessions. We want .env to be the source
# of truth, not whatever leaked into the terminal env hours ago.
load_dotenv(override=True)

from backend.app.config import get_settings  # noqa: E402
from backend.app.routers import (  # noqa: E402
    health,
    listings,
    preferences,
    upcycle,
    wardrobe,
)
from backend.app.services.cache import (  # noqa: E402
    STATIC_DIR,
    ensure_static_dirs,
    sync_cinematic_clips,
)

settings = get_settings()

# Some SDKs (fal_client) only consult os.environ; mirror Settings -> environ
# so they always see the same key the typed config uses. We unconditionally
# overwrite (not "only if unset") because load_dotenv(override=True) above
# already put the canonical .env values into Settings — anything still in
# os.environ from the parent shell is stale and should not win.
for _env_key, _val in {
    "FAL_KEY": settings.fal_key,
    "GEMINI_API_KEY": settings.gemini_api_key,
    "PIONEER_API_KEY": settings.pioneer_api_key,
    "TAVILY_API_KEY": settings.tavily_api_key,
}.items():
    if _val:
        os.environ[_env_key] = _val

# Tiny boot diagnostic so the operator can confirm at a glance which key is
# active right now. Prints only the last 6 chars — never the full secret.
# Uses print() to match the existing [startup] lines from cache.py; logging
# wouldn't show because root handlers aren't configured at import time under
# uvicorn.
for _label, _val in (
    ("GEMINI_API_KEY", settings.gemini_api_key),
    ("FAL_KEY", settings.fal_key),
    ("PIONEER_API_KEY", settings.pioneer_api_key),
    ("TAVILY_API_KEY", settings.tavily_api_key),
):
    if _val:
        print(f"[startup] {_label} loaded: ...{_val[-6:]}", flush=True)
    else:
        print(f"[startup] {_label}: NOT SET", flush=True)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Franken-Fit API — wardrobe audit, upcycle, resale listing. "
        "Hackathon build: live Gemini Vision + TTS, Tavily price comps, "
        "fal FLUX.2 + I2V, eBay Sandbox publish, Pioneer side-by-side."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static dirs MUST exist before the StaticFiles mount, so we initialise them
# at import time. Cinematic clip sync runs again on startup to log the result
# in the uvicorn console.
ensure_static_dirs()
sync_cinematic_clips()


@app.on_event("startup")
async def _on_startup() -> None:
    clip_status = sync_cinematic_clips()
    print(f"[startup] static dir: {STATIC_DIR}")
    print(f"[startup] cinematic clips: {clip_status}")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(health.router)
app.include_router(wardrobe.router, prefix="/v1/wardrobe", tags=["wardrobe"])
app.include_router(upcycle.router, prefix="/v1/upcycle", tags=["upcycle"])
app.include_router(listings.router, prefix="/v1/listings", tags=["listings"])
app.include_router(preferences.router, prefix="/v1/preferences", tags=["preferences"])
