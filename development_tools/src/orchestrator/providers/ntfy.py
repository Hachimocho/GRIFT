from __future__ import annotations

import httpx
from ..config import settings
import os


def notify(title: str, message: str, tags: list[str] | None = None) -> None:
    # Allow runtime override via env (e.g., provided by GitHub Secret at CI time)
    topic_override = os.getenv("NTFY_TOPIC")
    topic = topic_override or settings.ntfy_topic
    if not settings.ntfy_enabled or not topic:
        return
    headers = {
        "Title": title,
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    url = f"{settings.ntfy_base_url.rstrip('/')}/{topic}"
    with httpx.Client(timeout=15) as client:
        client.post(url, headers=headers, content=message)


