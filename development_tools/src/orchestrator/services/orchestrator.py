from __future__ import annotations

from typing import List
import yaml
import json
import os

from ..config import settings
from ..db import session_scope, Base, engine, apply_migrations
from ..models import FeatureRequest, Task as TaskModel, Event
from ..schemas import FeaturePrompt
from ..providers.cursor import CursorClient
from ..providers.github import GitHubClient
from ..providers.ntfy import notify
from .decomposer import decompose_feature
from .task_agent import start_task_agent
from .validator import run_final_validator


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    apply_migrations()


def create_feature(prompt_yaml: str, external_id: str | None = None) -> int:
    parsed = yaml.safe_load(prompt_yaml)
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML must parse to a dictionary, got {type(parsed)}: {parsed}")
    feature_prompt = FeaturePrompt(**parsed)

    with session_scope() as s:
        fr = FeatureRequest(
            external_id=external_id,
            title=feature_prompt.title,
            prompt_yaml=prompt_yaml,
            status="created",
        )
        s.add(fr)
        s.flush()
        feature_db_id = fr.id

    # Decompose
    cursor = CursorClient()
    result, agent_id = decompose_feature(cursor, feature_id=f"feat-{feature_db_id}", prompt=feature_prompt)

    # Persist tasks and write DAG to a JSON file for review
    with session_scope() as s:
        fr = s.get(FeatureRequest, feature_db_id)
        assert fr is not None
        fr.status = "pending_review"
        for t in result.tasks:
            tm = TaskModel(
                feature_id=fr.id,
                task_key=t.id,
                title=t.title,
                depends_on=t.depends_on,
                repo=t.repo,
                target_branch=settings.github_default_base_branch,
                acceptance_checks=t.acceptance_checks,
                estimate_hours=t.estimate_hours,
                paths=",".join(t.paths) if t.paths else None,
                status="pending",
            )
            s.add(tm)
        s.add(Event(feature_id=fr.id, type="decomposed", payload={"agent_id": agent_id}))

        dag_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "review_dags")
        dag_dir = os.path.abspath(dag_dir)
        os.makedirs(dag_dir, exist_ok=True)
        dag_path = os.path.join(dag_dir, f"feature_{feature_db_id}_dag.json")
        dag_payload = json.dumps(result.model_dump(), indent=2)
        with open(dag_path, "w", encoding="utf-8") as f:
            f.write(dag_payload)
        fr.dag_json_path = dag_path

    notify("Feature decomposed", f"Feature {feature_db_id} DAG ready for review", ["clipboard"]) 

    # Do not start tasks until approved

    return feature_db_id


def _start_runnable_tasks(feature_db_id: int) -> None:
    cursor = CursorClient()
    gh = GitHubClient()
    with session_scope() as s:
        tasks: List[TaskModel] = (
            s.query(TaskModel)
            .filter(TaskModel.feature_id == feature_db_id)
            .order_by(TaskModel.id.asc())
            .all()
        )
        completed_keys = {t.task_key for t in tasks if t.status in {"merged", "done", "validated"}}
        for t in tasks:
            if t.status != "pending":
                continue
            if not set(t.depends_on).issubset(completed_keys):
                continue
            # Start task agent
            launch = start_task_agent(
                cursor,
                gh,
                {
                    "id": t.task_key,
                    "title": t.title,
                    "repo": t.repo,
                    "feature_id": f"feat-{feature_db_id}",
                    "acceptance_checks": t.acceptance_checks,
                },
            )
            t.agent_id = launch["agent_id"]
            t.work_branch = launch["work_branch"]
            t.pr_number = launch.get("pr_number")
            t.pr_url = launch.get("pr_url")
            t.status = "in_progress"
            s.add(Event(feature_id=feature_db_id, task_id=t.id, type="task_started", payload=launch))
    notify("Tasks started", f"Feature {feature_db_id}: started runnable tasks", ["rocket"]) 


def handle_ci_result(repo: str, pr: int, conclusion: str, run_id: str, summary_url: str) -> None:
    # Map PR back to task
    with session_scope() as s:
        task: TaskModel | None = s.query(TaskModel).filter(TaskModel.pr_number == pr).first()
        if not task:
            s.add(Event(type="ci_unknown_pr", payload={"repo": repo, "pr": pr, "conclusion": conclusion}))
            return

        s.add(Event(feature_id=task.feature_id, task_id=task.id, type="ci_result", payload={
            "repo": repo,
            "pr": pr,
            "conclusion": conclusion,
            "run_id": run_id,
            "summary_url": summary_url,
        }))

        if conclusion.lower() == "success":
            task.status = "ready_to_merge"
        else:
            # Increment iterations and decide whether to re-prompt
            task.iteration_count += 1
            if task.iteration_count >= settings.orch_max_iterations:
                task.status = "failed"
                notify("Task failed", f"Task {task.task_key} exceeded iteration limit", ["warning"])
            else:
                # Re-prompt agent would go here
                task.status = "in_progress"

    # On success, attempt DAG-ordered merges when dependencies are ready
    _attempt_merges(task.feature_id)  # type: ignore
    # Potentially start newly-unblocked tasks
    _start_runnable_tasks(task.feature_id)  # type: ignore


def _attempt_merges(feature_db_id: int) -> None:
    gh = GitHubClient()
    with session_scope() as s:
        tasks: List[TaskModel] = (
            s.query(TaskModel)
            .filter(TaskModel.feature_id == feature_db_id)
            .order_by(TaskModel.id.asc())
            .all()
        )
        merged_keys = {t.task_key for t in tasks if t.status == "merged"}
        ready_tasks = [t for t in tasks if t.status == "ready_to_merge" and set(t.depends_on).issubset(merged_keys)]
        for t in ready_tasks:
            if not t.repo or not t.pr_number:
                continue
            owner, repo = t.repo.split("/")
            result = gh.merge_pull_request(owner, repo, t.pr_number)
            if result.get("merged"):
                t.status = "merged"
                s.add(Event(feature_id=t.feature_id, task_id=t.id, type="merged", payload={"pr": t.pr_number}))

        # If all tasks merged, run final validator
        tasks = (
            s.query(TaskModel)
            .filter(TaskModel.feature_id == feature_db_id)
            .all()
        )
        if tasks and all(t.status == "merged" for t in tasks):
            fr = s.get(FeatureRequest, feature_db_id)
            if fr:
                cursor = CursorClient()
                result = run_final_validator(cursor, feature_id=f"feat-{feature_db_id}", feature_prompt_json=fr.prompt_yaml)
                s.add(Event(feature_id=feature_db_id, type="final_validation", payload=result))
                if result.get("passed"):
                    fr.status = "validated"
                    notify("Feature validated", f"Feature {feature_db_id} merged and validated.", ["white_check_mark","rocket"])
                else:
                    fr.status = "validation_failed"
                    notify("Feature validation failed", f"Feature {feature_db_id} validation failed: {result.get('details')}", ["warning"]) 


