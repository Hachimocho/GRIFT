## Orchestrator Architecture

Components:
- Orchestrator (FastAPI): intake, DAG orchestration, CI loop, merge prep
- DB: SQLite via SQLAlchemy for dev
- Providers: Cursor, GitHub, ntfy
- Services: decomposition, task agents, orchestration

Flow:
1. POST /features with YAML prompt
2. Decomposer agent emits tasks (DAG)
3. Task agents spawn for runnable tasks; open PRs
4. GitHub Actions posts CI results to /webhooks/ci
5. Orchestrator iterates or marks ready_to_merge
6. Final validation (future work) and notifications

References:
- Cursor Background Agents API – Overview: https://docs.cursor.com/en/background-agent/api/overview


