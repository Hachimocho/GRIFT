from typing import Optional, List
from pydantic import BaseModel, Field


class FeaturePrompt(BaseModel):
    title: str
    business_goal: Optional[str] = None
    acceptance_criteria: List[str] = Field(default_factory=list)
    scope: dict | None = None
    repos: List[dict] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    migration: Optional[dict] = None
    test_strategy: Optional[dict] = None
    success_metrics: List[str] = Field(default_factory=list)


class DecomposedTask(BaseModel):
    id: str = Field(..., description="Stable task key emitted by decomposer")
    title: str
    depends_on: List[str] = Field(default_factory=list)
    repo: str
    paths: List[str] = Field(default_factory=list)
    acceptance_checks: List[str] = Field(default_factory=list)
    estimate_hours: Optional[int] = None


class DecompositionResult(BaseModel):
    feature_id: str
    tasks: List[DecomposedTask]


class CreateFeatureRequest(BaseModel):
    prompt_yaml: str
    external_id: Optional[str] = None


class FeatureResponse(BaseModel):
    id: int
    status: str
    title: str


class WebhookCIEvent(BaseModel):
    repo: str
    pr: int
    conclusion: str
    run_id: str | int
    summary_url: str


