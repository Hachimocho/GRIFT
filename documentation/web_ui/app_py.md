# `web_ui/app.py` (Web UI HTTP layer)

This document describes the architecture and responsibilities of `web_ui/app.py`.
It is intended as **high-level documentation**: how the Web UI is structured, how
requests flow through the system, and how `app.py` connects UI pages to the run
execution system.

## What `app.py` is

`web_ui/app.py` is the **Flask application entrypoint** for the HyperGraph Web UI.

It provides:

- **HTML pages** (server-rendered Jinja templates in `web_ui/templates/`)
- **JSON APIs** that the pages call via JavaScript

`app.py` is mostly “glue”: it does not implement the training logic itself. It
delegates key responsibilities to helper modules.

## Key dependencies (the real “engine” behind the UI)

- **`web_ui/config_manager.py` (`ConfigManager`)**
  - Persists experiment configurations as JSON files under `web_ui/configs/`
  - Persists configuration templates under `web_ui/config_templates/`
  - Wraps saved configs with metadata: `name`, `created`, `modified`, and an inner `config` dict

- **`web_ui/gpu_queue_manager.py` (`GPUQueueManager`)**
  - Implements the run queue + GPU scheduling
  - Starts training/testing runs as subprocesses (running `test_hierarchical.py`)
  - Tracks run metadata and logs under `web_ui/runs/`
  - Extracts results from logs and writes them back into run metadata

## What a “route” is (beginner-friendly)

In Flask, a route connects:

- a URL path (example: `/runs`)
- to a Python function that returns a response

Routes are declared like:

- `@app.route('/runs')`
- `def runs(): ...`

If the function returns `render_template(...)`, the browser receives HTML.
If it returns `jsonify(...)`, the browser receives JSON.

## Pages (HTML) provided by `app.py`

Pages are rendered from templates:

- `/` → dashboard (`index.html`)
- `/configure` → configuration UI (`configure.html`)
- `/configure/<config_name>` → edit an existing configuration (`configure.html`)
- `/runs` → run list (`runs.html`)
- `/runs/<run_id>` → run details + logs (`run_details.html`)
- `/results` → completed run results (`results.html`)
- `/templates` → configuration templates (`templates.html`)

### Template expectations and “compatibility flattening”

Run metadata may store accuracy/results in slightly different shapes depending on
where they originated (queued runs vs completed runs loaded from disk).

To keep templates simple, `app.py` sometimes “flattens” nested fields before
rendering, e.g. promoting `run["results"]["final_accuracy"]` to a top-level field.

This is why you’ll see small “patch” sections in `/` and `/results`.

## JSON APIs provided by `app.py`

### Configuration APIs

- `GET /api/configurations` → list saved configurations
- `POST /api/configurations` → save a configuration
- `GET /api/configurations/<config_name>` → fetch one configuration
- `DELETE /api/configurations/<config_name>` → delete a configuration

### Run APIs

- `POST /api/test-runs` → queue new run(s) from a saved config
- `GET /api/test-runs` → list runs
- `GET /api/test-runs/<run_id>` → fetch one run’s metadata
- `POST /api/test-runs/<run_id>/stop` → stop a running run
- `GET /api/test-runs/<run_id>/logs` → read a run log

#### Multi-run behavior (architectures × DQN models)

`POST /api/test-runs` supports queuing multiple runs if the saved configuration
specifies:

- multiple model architectures (comma-separated or list)
- multiple DQN model types (comma-separated or list)

The endpoint creates one queued run per combination by cloning the saved config
and overriding only the architecture / DQN fields for each run.

### GPU/queue status APIs

- `GET /api/gpu/status` → GPU info + queue status
- `GET /api/gpu/queue` → queue-only status
- `POST /api/gpu/check-orphaned` → detect runs marked “queued” but not actually in queue
- `POST /api/gpu/clear-queue` → stop running runs and cancel queued runs

### Cache APIs

- `GET /api/cache/status` → filesystem-based cache status for:
  - `node_cache/`
  - `graph_cache/`
- `POST /api/cache/generate` → kicks off cache generation in a background thread

These cache directories are expected at the **project root**, not inside `web_ui/`.

### Maintenance APIs

These are “repair” endpoints for metadata consistency:

- `POST /api/results/extract` → re-extract results from logs for completed runs
- `POST /api/runs/fix-status` → re-evaluate status from logs and update metadata

## Runtime model (what happens when you click “Run”)

1. Browser saves a config via `POST /api/configurations`.
2. Browser queues a run via `POST /api/test-runs`.
3. `GPUQueueManager` eventually starts a subprocess (one per run) with GPU isolation via `CUDA_VISIBLE_DEVICES`.
4. Logs are streamed to `web_ui/runs/<run_id>.log`.
5. When the process exits, metadata is updated in `web_ui/runs/<run_id>.json`.
6. The UI polls `/api/gpu/queue`, run detail endpoints, and log endpoints to show progress.

## Local development entrypoint

Running:

```bash
python web_ui/app.py
```

starts Flask in debug mode on port 5000 by default (or `$PORT`).

