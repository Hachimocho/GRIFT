from __future__ import annotations

from typing import Dict
import uuid
import random

from ..config import settings
from ..providers.cursor import CursorClient
from ..providers.github import GitHubClient


TASK_SYSTEM_PROMPT = (
    "You are a coding agent. Implement the task description on a new branch. "
    "Open or update a PR, run tests, and iterate on failures up to the configured limits. "
    "Keep commits minimal and descriptive."
)


def start_task_agent(cursor: CursorClient, github: GitHubClient, task_spec: Dict) -> Dict:
    task_id = task_spec["id"]
    repo_full = task_spec.get("repo", "")
    work_branch = f"feat/{task_id}/{uuid.uuid4().hex[:6]}"
    base_branch = settings.github_default_base_branch

    agent_id = cursor.create_agent(
        name=f"task-{task_id}",
        instructions=TASK_SYSTEM_PROMPT,
        context={
            "repositories": [{"repo": repo_full, "branch": base_branch}],
            "environment": {"test": "make ci"},
        },
    )

    # Send initial task message
    cursor.send_message(agent_id, content=str(task_spec))

    # Simulate PR creation in mock
    pr_number = 0
    pr_url = None
    if github.dry_run:
        pr_number = random.randint(10000, 99999)
        pr_url = f"https://github.com/{repo_full}/pull/{pr_number}"
    elif repo_full:
        owner, repo = repo_full.split("/")
        pr_data = github.create_pull_request(
            owner=owner,
            repo=repo,
            title=f"{task_spec.get('title', task_id)}",
            head=work_branch,
            base=base_branch,
            body=f"Automated task for {task_id}",
        )
        pr_number = pr_data["number"]
        pr_url = pr_data.get("html_url")
        github.add_labels(owner, repo, pr_number, [f"feature:{task_spec.get('feature_id','')}" , f"task:{task_id}"])

    return {"agent_id": agent_id, "work_branch": work_branch, "pr_number": pr_number, "pr_url": pr_url}


