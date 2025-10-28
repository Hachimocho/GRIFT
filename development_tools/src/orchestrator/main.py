from __future__ import annotations

from fastapi import FastAPI

from .services.orchestrator import init_db
from .routers.features import router as features_router
from .routers.webhooks import router as webhooks_router
from .routers.admin import router as admin_router


app = FastAPI(title="Background Agent Orchestrator", version="0.1.0")


@app.on_event("startup")
def startup_event():
    init_db()


app.include_router(features_router)
app.include_router(webhooks_router)
app.include_router(admin_router)


