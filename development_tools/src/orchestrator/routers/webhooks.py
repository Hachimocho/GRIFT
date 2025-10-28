from __future__ import annotations

from fastapi import APIRouter

from ..schemas import WebhookCIEvent
from ..services.orchestrator import handle_ci_result


router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/ci")
def webhook_ci(event: WebhookCIEvent):
    handle_ci_result(repo=event.repo, pr=event.pr, conclusion=event.conclusion, run_id=str(event.run_id), summary_url=event.summary_url)
    return {"ok": True}


