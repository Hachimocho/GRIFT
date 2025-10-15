from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class FeatureRequest(Base):
    __tablename__ = "feature_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(256))
    prompt_yaml: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    dag_json_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="feature", cascade="all, delete-orphan")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[int] = mapped_column(ForeignKey("feature_requests.id", ondelete="CASCADE"))
    task_key: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(256))
    depends_on: Mapped[list[str]] = mapped_column(JSON, default=list)
    repo: Mapped[str] = mapped_column(String(256))
    target_branch: Mapped[str] = mapped_column(String(128), default="dev")
    work_branch: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    acceptance_checks: Mapped[list[str]] = mapped_column(JSON, default=list)
    paths: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    estimate_hours: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    pr_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pr_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    feature: Mapped[FeatureRequest] = relationship("FeatureRequest", back_populates="tasks")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


