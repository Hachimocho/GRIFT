from __future__ import annotations

import json
from typing import Tuple

from ..providers.cursor import CursorClient
from ..schemas import FeaturePrompt, DecompositionResult, DecomposedTask


DECOMPOSER_SYSTEM_PROMPT = (
    "You are a decomposition agent. Split the feature into atomic tasks suitable for one PR. "
    "Output JSON only matching the schema: {feature_id: string, tasks: [{id, title, depends_on[], repo, paths[], acceptance_checks[], estimate_hours?}]}. "
    "Prefer independence; specify depends_on if needed. Include paths to guide scope. No prose."
)


def decompose_feature(cursor: CursorClient, feature_id: str, prompt: FeaturePrompt) -> Tuple[DecompositionResult, str]:
    agent_id = cursor.create_agent(
        name=f"decomposer-{feature_id}",
        instructions=DECOMPOSER_SYSTEM_PROMPT,
        context={
            "repositories": prompt.repos,
        },
    )

    cursor.send_message(agent_id, content=prompt.model_dump_json())

    # For now, rely on mock; in real mode you would poll or receive webhook
    if cursor.use_mock:
        result = DecompositionResult(
            feature_id=feature_id,
            tasks=[
                {
                    "id": "task-1-initial",
                    "title": f"Setup scaffolding for {prompt.title}",
                    "depends_on": [],
                    "repo": prompt.repos[0]["repo"] if prompt.repos else "",
                    "paths": ["src/**"],
                    "acceptance_checks": ["unit: *"],
                    "estimate_hours": 1,
                }
            ],
        )
        return result, agent_id

    # In non-mock mode, poll for messages and parse JSON from content
    messages = cursor.get_messages(agent_id)
    tasks: list[DecomposedTask] = []
    try:
        # Expect the latest message from the agent to contain JSON
        if isinstance(messages, dict) and messages.get("messages"):
            content = messages["messages"][-1].get("content", "{}")
            data = json.loads(content)
            for t in data.get("tasks", []):
                tasks.append(DecomposedTask(**t))
    except Exception:
        pass
    result = DecompositionResult(feature_id=feature_id, tasks=tasks)
    return result, agent_id


