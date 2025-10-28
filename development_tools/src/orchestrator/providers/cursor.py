from __future__ import annotations

from typing import Any, Dict, Optional
import uuid
import httpx

from ..config import settings


class CursorClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, use_mock: Optional[bool] = None):
        self.base_url = base_url or settings.cursor_base_url
        self.api_key = api_key or settings.cursor_api_key
        self.use_mock = settings.cursor_use_mock if use_mock is None else use_mock
        self._http = httpx.Client(base_url=self.base_url, headers=self._headers(), timeout=60)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # High-level API
    def create_agent(self, name: str, instructions: str, context: Dict[str, Any]) -> str:
        if self.use_mock:
            return f"agent_{uuid.uuid4().hex[:10]}"
        # Placeholder; replace with official endpoint when available
        resp = self._http.post("/v1/agents", json={
            "name": name,
            "instructions": instructions,
            "context": context,
        })
        resp.raise_for_status()
        data = resp.json()
        return data.get("id")

    def send_message(self, agent_id: str, content: str) -> Dict[str, Any]:
        if self.use_mock:
            return {"id": f"msg_{uuid.uuid4().hex[:8]}", "content": content}
        resp = self._http.post(f"/v1/agents/{agent_id}/messages", json={
            "role": "user",
            "content": content,
        })
        resp.raise_for_status()
        return resp.json()

    def get_messages(self, agent_id: str) -> Dict[str, Any]:
        if self.use_mock:
            return {"messages": []}
        resp = self._http.get(f"/v1/agents/{agent_id}/messages")
        resp.raise_for_status()
        return resp.json()


