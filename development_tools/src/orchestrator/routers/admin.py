from __future__ import annotations

from fastapi import APIRouter

from ..db import session_scope
from ..models import FeatureRequest, Task, Event


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/features")
def list_features():
    with session_scope() as s:
        rows = s.query(FeatureRequest).order_by(FeatureRequest.id.desc()).all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "status": r.status,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]


@router.get("/tasks")
def list_tasks():
    with session_scope() as s:
        rows = s.query(Task).order_by(Task.id.desc()).all()
        return [
            {
                "id": r.id,
                "feature_id": r.feature_id,
                "task_key": r.task_key,
                "status": r.status,
                "pr_number": r.pr_number,
                "pr_url": r.pr_url,
                "agent_id": r.agent_id,
                "iteration_count": r.iteration_count,
                "updated_at": r.updated_at,
            }
            for r in rows
        ]


@router.get("/events")
def list_events():
    with session_scope() as s:
        rows = s.query(Event).order_by(Event.id.desc()).limit(200).all()
        return [
            {
                "id": r.id,
                "feature_id": r.feature_id,
                "task_id": r.task_id,
                "type": r.type,
                "payload": r.payload,
                "created_at": r.created_at,
            }
            for r in rows
        ]


