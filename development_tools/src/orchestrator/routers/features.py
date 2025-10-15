from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from ..services.orchestrator import create_feature, _start_runnable_tasks
from ..db import session_scope
from ..models import FeatureRequest
from ..schemas import CreateFeatureRequest, FeatureResponse


router = APIRouter(prefix="/features", tags=["features"])


@router.post("", response_model=FeatureResponse)
def submit_feature(req: CreateFeatureRequest):
    try:
        feature_id = create_feature(prompt_yaml=req.prompt_yaml, external_id=req.external_id)
        return {"id": feature_id, "status": "created", "title": ""}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/yaml")
def submit_feature_yaml(raw_yaml: str = Body(..., media_type="text/plain")):
    try:
        feature_id = create_feature(prompt_yaml=raw_yaml, external_id=None)
        return {"id": feature_id, "status": "created", "title": ""}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/approve/{feature_id}")
def approve_feature(feature_id: int):
    with session_scope() as s:
        fr = s.get(FeatureRequest, feature_id)
        if not fr:
            raise HTTPException(status_code=404, detail="feature not found")
        if fr.status != "pending_review":
            raise HTTPException(status_code=400, detail=f"feature is {fr.status}, not pending_review")
        fr.status = "decomposed"
    _start_runnable_tasks(feature_id)
    return {"ok": True, "feature_id": feature_id, "status": "approved"}


@router.get("/{feature_id}/dag")
def get_feature_dag(feature_id: int):
    with session_scope() as s:
        fr = s.get(FeatureRequest, feature_id)
        if not fr:
            raise HTTPException(status_code=404, detail="feature not found")
        return {"feature_id": feature_id, "dag_json_path": fr.dag_json_path}


