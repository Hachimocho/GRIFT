## Orchestrator API

### POST /features
Submit a feature prompt (YAML) for decomposition and task spawning.

Request:
```json
{ "prompt_yaml": "<YAML string>", "external_id": "optional" }
```

Response:
```json
{ "id": 1, "status": "created", "title": "" }
```

### POST /webhooks/ci
Report CI results from GitHub Actions.

Request:
```json
{
  "repo": "org/repo",
  "pr": 123,
  "conclusion": "success|failure|cancelled",
  "run_id": "<run id>",
  "summary_url": "https://github.com/org/repo/actions/runs/<id>"
}
```

Response:
```json
{ "ok": true }
```

### POST /features/yaml
Submit raw YAML as text body.

Request (Content-Type: text/plain):
```
<YAML>
```

Response:
```json
{ "id": 1, "status": "created", "title": "" }
```

### GET /features/{feature_id}/dag
Return the file path to the DAG JSON written for review and editing.

Response:
```json
{ "feature_id": 1, "dag_json_path": "/abs/path/to/review_dags/feature_1_dag.json" }
```

### POST /features/approve/{feature_id}
Approve the DAG (after manual edits to the JSON file) and start runnable tasks.

Response:
```json
{ "ok": true, "feature_id": 1, "status": "approved" }
```

### Notes
- Configure ntfy via environment to receive progress notifications.
- Cursor Background Agents API – Overview: https://docs.cursor.com/en/background-agent/api/overview


