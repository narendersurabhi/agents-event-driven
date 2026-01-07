#  Event-Driven Agentic Pipeline

This repo is an event-driven resume tailoring pipeline:

**Input:** raw resume text + job description text  
**Output:** tailored resume JSON + cover letter JSON + downloadable DOCX files

The pipeline is orchestrated via an in-memory event bus. Agents never call each
other directly — the orchestrator reacts to events and publishes the next events.

## Repo layout

- `api/app.py` – FastAPI app (HTTP entrypoint)
- `api/pipeline.py` – event-driven pipeline endpoints (`/pipeline/*`)
- `ui/streamlit_pipeline_app.py` – Streamlit UI (calls the API)
- `core/pipeline_orchestrator.py` – orchestrator/state machine (events only)
- `core/llm_step_worker.py` – central LLM worker + JSON repair
- `agents/*` – domain agents + workers (JD, profile, match, compose, QA, improver, cover letter, DOCX rendering)
- `templates/*.docx` – DOCX templates for resume + cover letter
- `pipeline_jobs/` – persisted job snapshots (local dev)
- `profile_cache/` – persisted extracted profiles (local dev)
- `logs/` – JSONL logs (LLM + pipeline + metrics)

## Quickstart

### 1) Clone + install

```bash
# Replace <REPO_URL> with your Git remote URL (GitHub/GitLab/etc).
git clone https://github.com/narendersurabhi/agents.git
cd agents

# (Optional) Verify your remote is set:
git remote -v

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure `.env`

Copy the example file and set values:

```bash
cp .env.example .env
```

Required config:

- `LLM_PROVIDER` – `openai`, `openai-gpt5`, `claude`, or `gemini`
- `LLM_MODEL` – model name for the chosen provider
- `LLM_TIMEOUT_SECONDS` – timeout in seconds (float)
- Provider key:
  - OpenAI: `OPENAI_API_KEY`
  - Claude: `ANTHROPIC_API_KEY`
  - Gemini: `GOOGLE_API_KEY`

Optional:

- `API_BASE_URL` – Streamlit → API URL (default `http://localhost:8000`)
- `LLM_MAX_OUTPUT_TOKENS` / `GEMINI_MAX_TOKENS` – useful for Gemini max output control
- `OBS_LOG_FILE` – override metrics log path (defaults to `logs/llm_step.log`)

Sanity-check config:

```bash
.venv/bin/python -m scripts.check_llm_config
```

### 3) Run the API

```bash
.venv/bin/uvicorn api.app:app --reload --port 8000
```

### 4) Run the Streamlit UI

In a second terminal:

```bash
source .venv/bin/activate
streamlit run ui/streamlit_pipeline_app.py
```

Open the UI, paste your resume + job description, and click **Run Pipeline**.

## Running from scripts (optional)

- `scripts/run_full_event_pipeline.py` – local event-pipeline run (no UI)
- `scripts/run_pipeline_start_demo.py` – pipeline start/resume demo

## Outputs, logs, and persistence

- Job snapshots: `pipeline_jobs/<job_id>.json`
- Profile cache: `profile_cache/` (can be bypassed from the UI via “Force profile refresh”)
- DOCX debug output: `debug_docx/` (saved by API endpoints for troubleshooting)
- Logs: `logs/*.log` (JSONL). Override a single file target via `OBS_LOG_FILE`.

## Tests

Unit tests (no live LLM calls):

```bash
.venv/bin/python -m pytest --ignore=tests/integration
```

Integration tests (may call live providers; requires network + valid keys):

```bash
.venv/bin/python -m pytest tests/integration
```

Coverage:

```bash
.venv/bin/python -m pytest
```

## Adding a new agent (pipeline step)

This project uses “Pattern B”: the orchestrator publishes `<step>.requested`,
the agent’s worker translates that into `llm_step.requested` (with `schema_text`
for JSON repair), then publishes `<step>.completed`.

Checklist:

1) **Add event constants**
   - Update `core/pipeline_events.py` with:
     - `<STEP>_REQUESTED = "<step>.requested"`
     - `<STEP>_LLM_COMPLETED = "<step>.llm.completed"`
     - `<STEP>_COMPLETED = "<step>.completed"`

2) **Define/extend the output model**
   - Add a Pydantic model in `core/models.py` for your step’s structured output.
   - Create a schema text string (Pydantic-style) in the agent module for repair:
     - `STEP_SCHEMA_TEXT = """ class MyModel(BaseModel): ... """`

3) **Implement the agent**
   - Add `agents/<step>_agent.py` (or similar) with:
     - a system prompt + `build_messages(...)`
     - `parse_result(data) -> YourModel` using `YourModel.model_validate(...)`

4) **Implement the worker**
   - Add `agents/<step>_worker.py` following the existing pattern (see `agents/jd_worker.py`):
     - listen for `<step>.requested`
     - publish `llm_step.requested` with `{messages, schema_text}` and `reply_to=<step>.llm.completed`
     - listen for `<step>.llm.completed`, parse, then publish `<step>.completed`

5) **Wire it into the orchestrator**
   - Update `core/pipeline_orchestrator.py`:
     - extend `PipelineState` to store your artifact
     - add a `run_<step>_completed()` loop to consume `<step>.completed`
     - publish the next step’s `*.requested` event based on state
     - persist snapshots via `_persist()`

6) **Start it in the API runtime**
   - Update `api/pipeline.py:_ensure_runtime()`:
     - instantiate the agent + worker
     - start worker threads (`run_<step>_requests`, `run_llm_results`)
     - start orchestrator loop thread (`run_<step>_completed`)

7) **Expose it in the UI (recommended)**
   - Update `ui/streamlit_pipeline_app.py` to display the new artifact and add the step to the stage/DAG view.

8) **Add tests**
   - Add tests under `tests/` using fake LLMs (see `tests/test_resume_qa_async.py` and `tests/test_json_repair_agent.py`).
