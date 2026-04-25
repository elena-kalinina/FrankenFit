"""Load .env from FrankenFit repo root."""

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent


def load_frankenfit_env() -> None:
    load_dotenv(_ROOT / ".env")
