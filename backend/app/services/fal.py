"""
fal.ai service — FLUX.2 multi-reference upcycle + image-to-video wow beat.

fal_client is a synchronous SDK; we run blocking calls in a worker thread and
expose async wrappers. The router layer adds the timeout / cached-fallback
policy on top of these primitives.

Reference scripts:
  - func_test/test_fal_upcycle_from_garments.py  →  upcycle_garments
  - func_test/test_fal_image_to_video.py         →  image_to_video

Pricing (as of pre-hackathon):
  hailuo  ~$0.27 / 6s @ 768P     (cheap iteration default)
  luma    ~$0.50 / 5s @ 720p     (mid-price editorial)
  kling   ~$1.40 / 5s            (gold standard for fashion cloth motion)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from backend.app.models import GarmentDescription

FLUX_FLEX_APP = "fal-ai/flux-2-flex/edit"

# I2V dispatch — keep the same normalisation as func_test/test_fal_image_to_video.py.
ASPECT_ENUMS_LUMA = {"16:9", "9:16", "4:3", "3:4", "21:9", "9:21"}


@dataclass(frozen=True)
class _I2VModelSpec:
    name: str
    app_id: str
    default_duration: str
    duration_choices: tuple[str, ...]
    supports_aspect_ratio: bool
    supports_resolution: bool
    resolution_choices: tuple[str, ...]
    default_resolution: str | None


I2V_MODELS: dict[str, _I2VModelSpec] = {
    "hailuo": _I2VModelSpec(
        name="hailuo",
        app_id="fal-ai/minimax/hailuo-02/standard/image-to-video",
        default_duration="6",
        duration_choices=("6", "10"),
        supports_aspect_ratio=False,
        supports_resolution=True,
        resolution_choices=("512P", "768P"),
        default_resolution="768P",
    ),
    "luma": _I2VModelSpec(
        name="luma",
        app_id="fal-ai/luma-dream-machine/ray-2/image-to-video",
        default_duration="5",
        duration_choices=("5", "9"),
        supports_aspect_ratio=True,
        supports_resolution=True,
        resolution_choices=("540p", "720p", "1080p"),
        default_resolution="720p",
    ),
    "kling": _I2VModelSpec(
        name="kling",
        app_id="fal-ai/kling-video/v2.1/master/image-to-video",
        default_duration="5",
        duration_choices=("5", "10"),
        supports_aspect_ratio=False,
        supports_resolution=False,
        resolution_choices=(),
        default_resolution=None,
    ),
}

DEFAULT_RUNWAY_PROMPT = (
    "Luxury fashion runway video: the model in the reference image walks forward "
    "with confident slow steps, hips and shoulders natural, fabric moving gently with motion. "
    "Dramatic spotlight and dark runway background, subtle camera dolly-in, editorial pacing, "
    "photorealistic, no on-screen text."
)


def _resolve_i2v_model(model: str) -> _I2VModelSpec:
    """Accept either the short alias (hailuo|luma|kling) or the full app id."""
    if model in I2V_MODELS:
        return I2V_MODELS[model]
    for spec in I2V_MODELS.values():
        if spec.app_id == model:
            return spec
    # Permissive default to Hailuo — cheap iteration, won't surprise anyone.
    return I2V_MODELS["hailuo"]


# ---------------------------------------------------------------------------
# Upcycle (FLUX.2 flex multi-reference edit)
# ---------------------------------------------------------------------------

def _upload_image_path(path: Path) -> str:
    import fal_client
    from PIL import Image

    suffix = path.suffix.lower()
    fmt = "png" if suffix == ".png" else "jpeg"
    im = Image.open(path).convert("RGB")
    return fal_client.upload_image(im, format=fmt)


def _upcycle_sync(
    *,
    prompt: str,
    image_urls: list[str],
) -> dict[str, Any]:
    import fal_client

    return fal_client.subscribe(
        FLUX_FLEX_APP,
        arguments={
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": "square_hd",
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        },
        with_logs=True,
    )


async def upload_image_paths(paths: list[Path]) -> list[str]:
    """Upload a batch of local images to fal storage; return the public URLs."""
    return await asyncio.gather(*(asyncio.to_thread(_upload_image_path, p) for p in paths))


async def upcycle_garments(
    image_paths: list[Path],
    prompt: str,
    *,
    session_id: str,  # noqa: ARG001 — kept for future per-session caching
    model: str = FLUX_FLEX_APP,  # noqa: ARG001 — single FLUX flex app for now
) -> dict[str, Any]:
    """Submit a multi-reference FLUX.2 upcycle edit job.

    Returns a dict with keys: image_url, revised_prompt, seed, raw.
    Raises RuntimeError if the response shape is unexpected.
    """
    if not image_paths:
        raise ValueError("upcycle_garments requires at least one image path.")
    if not prompt or not prompt.strip():
        raise ValueError("upcycle_garments requires a non-empty prompt.")

    image_urls = await upload_image_paths(image_paths)
    result = await asyncio.to_thread(_upcycle_sync, prompt=prompt, image_urls=image_urls)

    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"FLUX.2 returned no images: {result!r}")
    image_url = images[0].get("url")
    if not image_url:
        raise RuntimeError(f"FLUX.2 first image has no url: {result!r}")

    return {
        "image_url": image_url,
        "revised_prompt": result.get("prompt") or prompt,
        "seed": result.get("seed"),
        "raw": result,
    }


# ---------------------------------------------------------------------------
# Image-to-video (Hailuo / Luma / Kling)
# ---------------------------------------------------------------------------

def _build_i2v_args(
    spec: _I2VModelSpec,
    *,
    prompt: str,
    image_url: str,
    duration: str,
    aspect_ratio: str,
    resolution: str | None,
) -> dict[str, Any]:
    args: dict[str, Any] = {"prompt": prompt, "image_url": image_url, "duration": duration}
    if spec.name == "hailuo":
        if resolution:
            args["resolution"] = resolution
    elif spec.name == "luma":
        if aspect_ratio in ASPECT_ENUMS_LUMA:
            args["aspect_ratio"] = aspect_ratio
        if resolution:
            args["resolution"] = resolution
        if not args["duration"].endswith("s"):
            args["duration"] = f"{args['duration']}s"
    elif spec.name == "kling":
        # Kling has no aspect/resolution knobs; default negative_prompt is
        # applied server-side. Leave duration as bare seconds.
        pass
    return args


def _i2v_sync(*, app_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import fal_client

    return fal_client.subscribe(app_id, arguments=arguments, with_logs=True)


def _extract_video_url(result: dict[str, Any]) -> str | None:
    video = result.get("video")
    if isinstance(video, dict) and video.get("url"):
        return str(video["url"])
    if isinstance(video, str):
        return video
    videos = result.get("videos")
    if isinstance(videos, list) and videos:
        first = videos[0]
        if isinstance(first, dict) and first.get("url"):
            return str(first["url"])
        if isinstance(first, str):
            return first
    output = result.get("output")
    if isinstance(output, dict):
        v = output.get("video")
        if isinstance(v, dict) and v.get("url"):
            return str(v["url"])
        if isinstance(v, str):
            return v
    return None


async def image_to_video(
    image_url: str,
    *,
    session_id: str,  # noqa: ARG001 — kept for future per-session caching
    model: str = "hailuo",
    duration: int | str = 6,
    aspect_ratio: str = "9:16",
    prompt: str | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    """Animate a hero still with a fal.ai image-to-video model.

    Returns a dict: {video_url, model, duration_s, raw}. Raises RuntimeError
    if the endpoint returns no video URL. Caller is responsible for any
    timeout / cached-fallback policy.
    """
    spec = _resolve_i2v_model(model)
    duration_s = str(duration).rstrip("s")
    if duration_s not in spec.duration_choices:
        duration_s = spec.default_duration

    args = _build_i2v_args(
        spec,
        prompt=(prompt or DEFAULT_RUNWAY_PROMPT).strip(),
        image_url=image_url,
        duration=duration_s,
        aspect_ratio=aspect_ratio,
        resolution=resolution if spec.supports_resolution else None,
    )
    result = await asyncio.to_thread(_i2v_sync, app_id=spec.app_id, arguments=args)
    video_url = _extract_video_url(result)
    if not video_url:
        raise RuntimeError(f"I2V {spec.name} returned no video url: {result!r}")
    return {
        "video_url": video_url,
        "model": spec.app_id,
        "duration_s": int(duration_s),
        "raw": result,
    }


# ---------------------------------------------------------------------------
# Helpers — public so the router layer can download the MP4 to a cache path.
# ---------------------------------------------------------------------------

async def download_to(url: str, dest: Path, *, timeout: float = 120.0) -> int:
    """Stream a remote URL to *dest*. Returns bytes written. Creates parent dirs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                async for chunk in r.aiter_bytes(chunk_size=65536):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
    return written
