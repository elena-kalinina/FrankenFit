"""Application settings — loaded from .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Franken-Fit API"
    debug: bool = False

    # --- Core partners ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_vision_model: str = Field(
        default="gemini-3-flash-preview",
        alias="GEMINI_VISION_MODEL",
        description=(
            "Primary vision + text Gemini model. Default is gemini-3-flash-preview "
            "(full Flash, healthy capacity). Falls back to GEMINI_VISION_FALLBACK_MODEL "
            "on transient 503/UNAVAILABLE / RESOURCE_EXHAUSTED errors."
        ),
    )
    gemini_vision_fallback_models_csv: str = Field(
        default="gemini-3.1-flash-lite-preview,gemini-2.5-flash",
        alias="GEMINI_VISION_FALLBACK_MODELS",
        description=(
            "Comma-separated chain of vision/text fallback models. Walked left-to-right "
            "when the primary 503s / 429s. Default chain ends at gemini-2.5-flash (GA, "
            "highest free-tier daily quota) so the demo never dies on Gemini availability."
        ),
    )

    @property
    def gemini_vision_fallback_models(self) -> list[str]:
        return [m.strip() for m in self.gemini_vision_fallback_models_csv.split(",") if m.strip()]
    gemini_tts_model: str = Field(
        default="gemini-2.5-flash-preview-tts",
        alias="GEMINI_TTS_MODEL",
        description=(
            "Primary Gemini TTS model. All preview-tier TTS models live on tiny "
            "per-day free quotas (~10 req/day), so we chain them via "
            "GEMINI_TTS_FALLBACK_MODELS — when the primary is exhausted, the "
            "next model in the chain serves the render."
        ),
    )
    gemini_tts_fallback_models_csv: str = Field(
        default="gemini-3.1-flash-tts-preview,gemini-2.5-pro-preview-tts",
        alias="GEMINI_TTS_FALLBACK_MODELS",
        description=(
            "Comma-separated TTS fallback chain. Walked left-to-right when the "
            "primary 503s / 429s. Each preview TTS model has its own per-day "
            "quota pool, so rotating across them effectively gives us 3× the "
            "free-tier capacity for the demo's per-garment roast renders."
        ),
    )
    gemini_tts_voice: str = Field(default="Puck", alias="GEMINI_TTS_VOICE")

    @property
    def gemini_tts_fallback_models(self) -> list[str]:
        return [m.strip() for m in self.gemini_tts_fallback_models_csv.split(",") if m.strip()]

    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # --- fal.ai (paid infrastructure) ---
    fal_key: str = Field(default="", alias="FAL_KEY")
    fal_upcycle_model: str = Field(default="fal-ai/flux-pro/v1.1-ultra", alias="FAL_UPCYCLE_MODEL")
    fal_i2v_model: str = Field(
        default="minimax/video-01",
        alias="FAL_I2V_MODEL",
        description="fal image-to-video model slug for the wow beat.",
    )

    # --- Pioneer (optional 4th-partner beat) ---
    pioneer_api_key: str = Field(default="", alias="PIONEER_API_KEY")
    pioneer_api_base: str = Field(default="https://api.pioneer.ai", alias="PIONEER_API_BASE")
    pioneer_base_model: str = Field(default="fastino/gliner2-base-v1", alias="PIONEER_BASE_MODEL")
    pioneer_qwen_model: str = Field(
        default="",
        alias="PIONEER_QWEN_MODEL",
        description="Qwen model ID from Pioneer dashboard — used as Day-1 live inference baseline.",
    )
    pioneer_trained_model_id: str = Field(
        default="941f616d-4c09-43eb-9155-80a623efde83",
        alias="PIONEER_TRAINED_MODEL_ID",
        description=(
            "Fine-tuned GLiNER model (training_job_id). "
            "Pre-hackathon test value is set as default; update after Day-1 fine-tune."
        ),
    )
    pioneer_per_call_timeout_seconds: float = Field(
        default=15.0,
        alias="PIONEER_PER_CALL_TIMEOUT_SECONDS",
        description=(
            "Per-model deadline for the Pioneer side-by-side. Each call (baseline / "
            "trained) gets its own asyncio.wait_for at this value, so a slow Qwen "
            "doesn't drag the trained response down with it. The two calls run "
            "concurrently, so total latency stays close to this number."
        ),
    )

    # --- eBay Sandbox ---
    ebay_app_id: str = Field(default="", alias="EBAY_APP_ID")
    ebay_dev_id: str = Field(default="", alias="EBAY_DEV_ID")
    ebay_cert_id: str = Field(default="", alias="EBAY_CERT_ID")
    ebay_user_token: str = Field(default="", alias="EBAY_USER_TOKEN")
    ebay_env: str = Field(default="sandbox", alias="EBAY_ENV")
    ebay_site_id: str = Field(default="0", alias="EBAY_SITE_ID")
    ebay_category_id: str = Field(default="15724", alias="EBAY_CATEGORY_ID")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
