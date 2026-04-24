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
    gemini_model: str = Field(default="gemini-2.5-flash-preview-04-17", alias="GEMINI_MODEL")
    gemini_tts_model: str = Field(default="gemini-3.1-flash-tts-preview", alias="GEMINI_TTS_MODEL")
    gemini_tts_voice: str = Field(default="Aoede", alias="GEMINI_TTS_VOICE")

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
    pioneer_trained_model_id: str = Field(
        default="",
        alias="PIONEER_TRAINED_MODEL_ID",
        description="Set to the training_job_id once the Day-1 fine-tune completes.",
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
