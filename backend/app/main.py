from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.routers import health, listings, upcycle, wardrobe
from backend.app.routers import preferences

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.0.0",
    description=(
        "Franken-Fit API — wardrobe audit, upcycle, resale listing. "
        "All feature handlers are stubs; implement during the hackathon build window "
        "(see implementation_notes.md)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(wardrobe.router, prefix="/v1/wardrobe", tags=["wardrobe"])
app.include_router(upcycle.router, prefix="/v1/upcycle", tags=["upcycle"])
app.include_router(listings.router, prefix="/v1/listings", tags=["listings"])
app.include_router(preferences.router, prefix="/v1/preferences", tags=["preferences"])
