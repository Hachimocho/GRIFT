from __future__ import annotations

import json
from typing import Dict, Any

from ..providers.cursor import CursorClient


VALIDATOR_SYSTEM_PROMPT = (
    "You are a final validator. Verify the merged codebase satisfies the original feature's acceptance criteria. "
    "Run tests as needed. Output only JSON of the form {passed: boolean, details: string}."
)


def run_final_validator(cursor: CursorClient, feature_id: str, feature_prompt_json: str) -> Dict[str, Any]:
    agent_id = cursor.create_agent(
        name=f"validator-{feature_id}",
        instructions=VALIDATOR_SYSTEM_PROMPT,
        context={},
    )
    cursor.send_message(agent_id, content=feature_prompt_json)

    if cursor.use_mock:
        return {"passed": True, "details": "All acceptance criteria satisfied in mock mode."}

    # Non-mock: try to parse from last message
    messages = cursor.get_messages(agent_id)
    try:
        if isinstance(messages, dict) and messages.get("messages"):
            content = messages["messages"][-1].get("content", "{}")
            data = json.loads(content)
            if isinstance(data, dict) and "passed" in data:
                return data
    except Exception:
        pass
    return {"passed": False, "details": "Validator did not produce a parseable result."}


