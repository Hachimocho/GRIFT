## Development Tools – Background Agent Orchestrator

This folder contains an orchestrator service and helper scripts to automate large-scale feature development using Cursor Background Agents, GitHub (PRs, Actions), and ntfy.sh notifications.

High-level capabilities:
- Intake standardized feature prompts
- Decompose into a DAG of atomic tasks via a decomposition agent
- Spawn task agents per node; open/iterate PRs based on CI feedback
- Merge in dependency order; run final validation; notify via ntfy

References:
- Cursor Background Agents API – Overview: https://docs.cursor.com/en/background-agent/api/overview

### Quick start (local, mock mode)
1. Create and fill `.env` from `.env.example`.
2. Install dependencies:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Run the service:
   - `./run.sh`
   - If not using a `.env` file, export variables or `set -a; source env.example; set +a` and then override.
4. Submit a feature request (from the `development_tools` directory):
   - Using the test file:
     ```bash
     cd development_tools
     curl -X POST http://localhost:8088/features/yaml \
       -H 'Content-Type: text/plain' \
       --data-binary @test_feature.yaml
     ```
   - Or using the full template:
     ```bash
     curl -X POST http://localhost:8088/features/yaml \
       -H 'Content-Type: text/plain' \
       --data-binary @src/orchestrator/templates/feature_prompt_template.yaml
     ```

Mock mode avoids real calls to Cursor/GitHub and simulates agents/PRs.

### Deploying with real integrations
- Set `CURSOR_API_KEY` and ensure repository access for Cursor background agents.
- Provide a GitHub token (prefer a GitHub App; a PAT works for testing) and disable `GITHUB_DRY_RUN`.
- Configure a GitHub Actions workflow to POST CI results to `/webhooks/ci` as documented in `docs/API.md`.

### Structure
```
development_tools/
  docs/                # Overview, API contracts, prompts
  src/orchestrator/    # FastAPI app, DB models, providers, services, routers
  requirements.txt
  run.sh
  .env.example
```

See `docs/ARCHITECTURE.md`, `docs/API.md`, and `docs/PROMPTS.md` for details.


