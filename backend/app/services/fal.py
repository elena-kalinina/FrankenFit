"""
fal.ai service stubs (FLUX.2 upcycle + image-to-video).

fal.ai is paid infrastructure — not a hackathon partner slide entry.
Implement with fal_client (already tested in func_test/passed/test_fal_upcycle_from_garments.py
and func_test/test_fal_image_to_video.py).

Set FAL_KEY in .env — fal_client reads it automatically.
"""

from __future__ import annotations

from backend.app.models import AnimateResponse, GarmentDescription, UpcycleResponse


async def upcycle_garments(
    garments: list[GarmentDescription],
    style_prompt: str,
    *,
    session_id: str,
    model: str = "fal-ai/flux-pro/v1.1-ultra",
) -> UpcycleResponse:
    """
    Submit a multi-reference FLUX.2 upcycle job to fal.ai.

    Steps (implement on hackathon day):
      1. For each garment, resolve a public image URL (stored in session or uploaded to fal CDN).
      2. Build the fal FLUX.2 payload with reference_image_urls + style_prompt.
      3. Call fal_client.run() or fal_client.subscribe() (async queue).
      4. Return UpcycleResponse with the output image_url + revised_prompt + seed.

    See func_test/passed/test_fal_upcycle_from_garments.py for working payload shape.
    """
    raise NotImplementedError("implement: fal FLUX.2 upcycle")


async def image_to_video(
    image_url: str,
    *,
    session_id: str,
    model: str = "minimax/video-01",
    duration: int = 10,
    aspect_ratio: str = "9:16",
) -> AnimateResponse:
    """
    Animate the upcycled hero still with a fal.ai image-to-video model.

    Supported model slugs and constraints (implement on hackathon day):
      - "minimax/video-01"                                   → max 6 s, 720p
      - "fal-ai/luma-dream-machine/ray-2"                    → max 9 s
      - "fal-ai/kling-video/v2.1/master/image-to-video"     → max 10 s, best quality for fashion

    Pre-render the MP4 at T-1 day and cache in func_test/out/ — do NOT render live on stage.
    See func_test/test_fal_image_to_video.py for reference.
    """
    raise NotImplementedError("implement: fal image-to-video")
