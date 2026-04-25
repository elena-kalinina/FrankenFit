"""
In-memory session store for the Franken-Fit demo.

Keeps the demo state — analysed garments, swipe history, artefact URLs —
in a plain module-level dict. Sufficient for a single-node hackathon demo
where the backend restarts between demo runs.

Replace with Redis / SQLite if the demo needs multi-node or persistence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SwipeEvent:
    garment_id: str
    direction: str  # "like" | "dislike"
    garment_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoSession:
    session_id: str
    garments: dict[str, dict[str, Any]] = field(default_factory=dict)
    garment_local_paths: dict[str, str] = field(default_factory=dict)
    swipes: list[SwipeEvent] = field(default_factory=list)
    upcycle_image_url: str | None = None
    upcycle_garment_id: str | None = None
    upcycle_video_url: str | None = None
    listing_draft: dict[str, Any] | None = None
    price_band: dict[str, Any] | None = None
    pioneer_job_id: str | None = None

    @property
    def keepers(self) -> list[str]:
        return [s.garment_id for s in self.swipes if s.direction == "like"]

    @property
    def franken_bin(self) -> list[str]:
        return [s.garment_id for s in self.swipes if s.direction == "dislike"]


# Module-level store — not thread-safe, fine for a single-threaded demo process.
_sessions: dict[str, DemoSession] = {}


def create_session() -> DemoSession:
    sid = str(uuid.uuid4())
    session = DemoSession(session_id=sid)
    _sessions[sid] = session
    return session


def get_session(session_id: str) -> DemoSession | None:
    return _sessions.get(session_id)


def get_or_create(session_id: str | None) -> DemoSession:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    return create_session()


def record_swipe(session_id: str, garment_id: str, direction: str, meta: dict[str, Any]) -> DemoSession:
    session = get_or_create(session_id)
    session.swipes.append(SwipeEvent(garment_id=garment_id, direction=direction, garment_meta=meta))
    if garment_id not in session.garments:
        session.garments[garment_id] = meta
    return session


def all_sessions() -> dict[str, DemoSession]:
    return dict(_sessions)
