"""Streamlit UI for the event-driven pipeline FastAPI.

Run:
  streamlit run ui/streamlit_pipeline_app.py

Make sure your FastAPI app is running (uvicorn api.app:app) and that
the /pipeline/run endpoint is reachable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# Ensure repo root is on sys.path when running via Streamlit.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import get_config_value
from ui.api_client import ApiClient, ApiError
from ui.obs_metrics import aggregate_step_metrics, build_step_rows, read_recent_jsonl

# Configure API base URL from config/.env (env vars override .env).
# Set `API_BASE_URL=http://localhost:8000` in `.env` to override.
_API_BASE_URL = (get_config_value("API_BASE_URL", "http://localhost:8000") or "").rstrip("/")
if not _API_BASE_URL:
    _API_BASE_URL = "http://localhost:8000"

_api = ApiClient(_API_BASE_URL)


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    run_qa: bool
    run_improver: bool
    force_profile_refresh: bool


def _job_label(job: dict) -> str:
    jd = job.get("jd") or {}
    role = (jd.get("role_title") or "").strip()
    company = (jd.get("company") or "").strip()
    stage = job.get("stage", "unknown")
    job_id = job.get("job_id", "?")
    name = role or "Unknown role"
    if company:
        name = f"{name} @ {company}"
    return f"{name} — {stage} — {job_id}"


def main() -> None:
    st.set_page_config(page_title="Resume Tailoring Pipeline", layout="wide")
    st.title("Resume Tailoring Pipeline (Event-Driven)")

    options = _render_sidebar_options()
    _render_job_history_sidebar(_api)
    jd_text, resume_text = _render_inputs()
    _render_action_buttons(_api, jd_text, resume_text, options)
    _render_current_job_section(_api)
    st.markdown("---")
    _render_metrics_dashboard()


def _render_sidebar_options() -> PipelineOptions:
    st.sidebar.header("Pipeline Options")
    st.sidebar.caption(f"API: {_API_BASE_URL}")
    return PipelineOptions(
        run_qa=st.sidebar.checkbox("Run QA", value=True),
        run_improver=st.sidebar.checkbox("Run QA Improver", value=True),
        force_profile_refresh=st.sidebar.checkbox("Force profile refresh (ignore cache)", value=False),
    )


def _render_job_history_sidebar(client: ApiClient) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Job History")

    if st.sidebar.button("Refresh Jobs"):
        try:
            st.session_state["jobs"] = client.get_json("/pipeline/jobs")
        except ApiError as exc:
            st.sidebar.error(f"Failed to load jobs: {exc}")

    jobs = st.session_state.get("jobs", [])
    search_term = st.sidebar.text_input("Search jobs (by role/company)")
    filtered_jobs = _filter_jobs(jobs, search_term)

    if not filtered_jobs:
        return

    selected_job = st.sidebar.selectbox(
        "Select job to load",
        filtered_jobs,
        format_func=_job_label,
    )
    if st.sidebar.button("Load Selected Job"):
        job_id = selected_job.get("job_id")
        if job_id:
            st.session_state["job_id"] = job_id
            _fetch_and_render_status(client, job_id)


def _filter_jobs(jobs: list[dict], search_term: str) -> list[dict]:
    if not search_term:
        return jobs
    lowered = search_term.lower()
    return [j for j in jobs if lowered in _job_label(j).lower()]


def _render_inputs() -> tuple[str, str]:
    col1, col2 = st.columns(2)
    with col1:
        jd_text = st.text_area("Job Description", height=300)
    with col2:
        resume_text = st.text_area("Resume", height=300)
    return jd_text, resume_text


def _render_action_buttons(
    client: ApiClient, jd_text: str, resume_text: str, options: PipelineOptions
) -> None:
    run_col, resume_col, restart_col = st.columns(3)
    with run_col:
        _handle_run_pipeline(client, jd_text, resume_text, options)
    with resume_col:
        _handle_resume_last_job(client)
    with restart_col:
        _handle_restart_compose(client)


def _handle_run_pipeline(
    client: ApiClient, jd_text: str, resume_text: str, options: PipelineOptions
) -> None:
    if not st.button("Run Pipeline"):
        return
    if not jd_text.strip() or not resume_text.strip():
        st.error("Please provide both a Job Description and a Resume.")
        return

    payload = {
        "jd_text": jd_text,
        "resume_text": resume_text,
        "run_qa": options.run_qa,
        "run_improver": options.run_improver,
        "force_profile_refresh": options.force_profile_refresh,
    }

    with st.spinner("Running pipeline..."):
        try:
            data = client.post_json("/pipeline/run", payload=payload)
        except ApiError as exc:
            st.error(str(exc))
            return

    job_id = data.get("job_id")
    if job_id:
        st.session_state["job_id"] = job_id
        st.info(f"Job ID: {job_id}")
        _fetch_and_render_status(client, job_id)


def _handle_resume_last_job(client: ApiClient) -> None:
    if not st.button("Resume Last Job"):
        return
    job_id = st.session_state.get("job_id")
    if not job_id:
        st.warning("No previous job_id found. Run the pipeline first.")
        return

    with st.spinner(f"Resuming pipeline for job_id={job_id}..."):
        try:
            data = client.post_json(f"/pipeline/resume/{job_id}", payload=None)
        except ApiError as exc:
            st.error(f"Resume request failed: {exc}")
            return

    resumed_job_id = data.get("job_id") or job_id
    st.info(f"Resume triggered for Job ID: {resumed_job_id}")
    _fetch_and_render_status(client, resumed_job_id)


def _handle_restart_compose(client: ApiClient) -> None:
    if not st.button("Restart from Compose"):
        return
    job_id = st.session_state.get("job_id")
    if not job_id:
        st.warning("No previous job_id found. Run the pipeline first.")
        return

    with st.spinner(f"Restarting compose for job_id={job_id}..."):
        try:
            data = client.post_json(f"/pipeline/restart-compose/{job_id}", payload=None)
        except ApiError as exc:
            st.error(f"Restart compose request failed: {exc}")
            return

    restarted_job_id = data.get("job_id") or job_id
    st.info(f"Compose restart triggered for Job ID: {restarted_job_id}")
    _fetch_and_render_status(client, restarted_job_id)


def _render_current_job_section(client: ApiClient) -> None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return

    st.markdown("---")
    st.subheader("Current Job Status")

    status_col, docx_col = st.columns(2)
    with status_col:
        if st.button("Refresh Status"):
            _fetch_and_render_status(client, job_id)
    with docx_col:
        if st.button("Prepare Resume DOCX"):
            _prepare_docx(client, job_id)
        if st.button("Prepare Cover Letter DOCX"):
            _prepare_cover_docx(client, job_id)

    _render_download_buttons(job_id)


def _render_download_buttons(job_id: str) -> None:
    if "docx_bytes" in st.session_state:
        filename = st.session_state.get("docx_filename", f"resume-{job_id}.docx")
        st.download_button(
            label="Download DOCX",
            data=st.session_state["docx_bytes"],
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    if "cover_docx_bytes" in st.session_state:
        cover_filename = st.session_state.get("cover_docx_filename", f"cover-letter-{job_id}.docx")
        st.download_button(
            label="Download Cover Letter DOCX",
            data=st.session_state["cover_docx_bytes"],
            file_name=cover_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def _fetch_and_render_status(client: ApiClient, job_id: str) -> None:
    if not job_id:
        return
    try:
        data = client.get_json(f"/pipeline/status/{job_id}", timeout=10)
    except ApiError as exc:
        st.error(str(exc))
        return

    stage = data.get("stage", "unknown")
    completed = data.get("stage") == "COMPLETED"

    st.write(f"Job ID: {job_id}")
    st.write(f"Stage: **{stage}** | Completed: **{completed}**")
    _render_stage_bar(stage, data)

    _render_results(data)


def _render_stage_bar(stage: str, data: dict) -> None:
    """Visual DAG-style pipeline view for the current job."""
    run_qa = bool(data.get("run_qa", True))
    run_improver = bool(data.get("run_improver", True))

    # Map orchestrator stages to an overall progress rank.
    stage_rank_map = {
        "PENDING": 0,
        "STARTED": 0,
        "JD_COMPLETED": 1,
        "PROFILE_COMPLETED": 1,
        "MATCH_COMPLETED": 2,
        "COMPOSE_COMPLETED": 3,
        "QA_COMPLETED": 4,
        "QA_IMPROVE_COMPLETED": 5,
        "COVER_LETTER_COMPLETED": 6,
        "COMPLETED": 7,
    }
    current_rank = stage_rank_map.get(stage, 0)

    # Define DAG nodes with logical ranks (columns).
    nodes: list[tuple[str, str, int]] = [
        ("JD", "JD Analysis", 1),
        ("PROFILE", "Profile Extraction", 1),
        ("MATCH", "Match Planning", 2),
        ("COMPOSE", "Compose Resume", 3),
    ]
    if run_qa:
        nodes.append(("QA", "QA", 4))
        if run_improver:
            nodes.append(("QA_IMPROVE", "QA Improve", 5))
            nodes.append(("COVER", "Cover Letter", 6))
    nodes.append(("DONE", "Done", 7))

    # Edges for the DAG.
    edges: list[tuple[str, str]] = [
        ("JD", "MATCH"),
        ("PROFILE", "MATCH"),
        ("MATCH", "COMPOSE"),
    ]
    if run_qa:
        edges.append(("COMPOSE", "QA"))
        if run_improver:
            edges.append(("QA", "QA_IMPROVE"))
            edges.append(("QA_IMPROVE", "COVER"))
            edges.append(("COVER", "DONE"))
        else:
            edges.append(("QA", "DONE"))
    else:
        edges.append(("COMPOSE", "DONE"))

    # Build Graphviz DOT with color coding based on current_rank.
    dot_lines: list[str] = [
        "digraph G {",
        'rankdir=LR;',
        'node [shape=box, style=filled, fontname="Helvetica", fontsize=10];',
    ]

    for node_id, label, rank in nodes:
        if current_rank > rank:
            color = "palegreen"
        elif current_rank == rank:
            color = "gold"
        else:
            color = "lightgray"
        dot_lines.append(
            f'"{node_id}" [label="{label}", fillcolor="{color}"];'
        )

    for src, dst in edges:
        dot_lines.append(f'"{src}" -> "{dst}";')

    dot_lines.append("}")
    st.graphviz_chart("\n".join(dot_lines))


def _prepare_docx(client: ApiClient, job_id: str) -> None:
    if not job_id:
        return
    try:
        st.session_state["docx_bytes"] = client.get_bytes(f"/pipeline/{job_id}/docx")
    except ApiError as exc:
        st.error(f"DOCX request failed: {exc}")
        return


def _prepare_cover_docx(client: ApiClient, job_id: str) -> None:
    if not job_id:
        return
    try:
        st.session_state["cover_docx_bytes"] = client.get_bytes(
            f"/pipeline/{job_id}/cover-letter.docx"
        )
    except ApiError as exc:
        st.error(f"Cover letter DOCX request failed: {exc}")
        return


def _render_results(data: dict) -> None:
    jd = data.get("jd")
    profile = data.get("profile")
    plan = data.get("plan")
    composed = data.get("tailored")
    improved = data.get("improved")
    cover_letter = data.get("cover_letter")
    qa = data.get("qa")

    st.subheader("JD Analysis")
    if jd is not None:
        st.json(jd)
    else:
        st.write("Not available yet.")

    st.subheader("Professional Profile")
    if profile is not None:
        st.json(profile)
    else:
        st.write("Not available yet.")

    st.subheader("Resume Plan")
    if plan is not None:
        st.json(plan)
    else:
        st.write("Not available yet.")

    st.subheader("Composed Resume (before QA improver)")
    if composed is not None:
        st.json(composed)
    else:
        st.write("Not available yet.")

    st.subheader("Tailored Resume (Final)")
    final_resume = improved or composed
    if final_resume is not None:
        st.json(final_resume)
        # Use the candidate's full name to suggest a friendly resume filename.
        full_name = (final_resume.get("full_name") or "Resume").strip()
        safe_name = full_name.replace("/", "-").replace("\\", "-")
        st.session_state["docx_filename"] = f"{safe_name} Resume.docx"
    else:
        st.write("Not available yet.")

    if qa is not None:
        st.subheader("Resume QA Result")
        st.json(qa)

    if cover_letter is not None:
        st.subheader("Cover Letter")
        st.json(cover_letter)
        body = cover_letter.get("body") or ""
        full_name = (cover_letter.get("full_name") or "Cover Letter").strip()
        safe_name = full_name.replace("/", "-").replace("\\", "-")
        st.session_state["cover_docx_filename"] = f"{safe_name} Cover Letter.docx"
        st.download_button(
            label="Download Cover Letter (TXT)",
            data=body,
            file_name=f"{safe_name} Cover Letter.txt",
            mime="text/plain",
        )


def _render_metrics_dashboard() -> None:
    """Render an observability table for pipeline tasks."""
    st.subheader("Observability Metrics (Pipeline Tasks)")

    repo_root = Path(__file__).resolve().parents[1]
    log_path = Path(get_config_value("OBS_LOG_FILE") or "logs/llm_step.log")
    if not log_path.is_absolute():
        log_path = repo_root / log_path
    if not log_path.exists():
        st.info(f"No task metrics found yet (missing `{log_path}`). Run a job to generate metrics.")
        return

    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        st.error(f"Failed to read metrics log: {exc}")
        return

    records = read_recent_jsonl(text, max_lines=5000)
    stats = aggregate_step_metrics(records)
    rows = build_step_rows(stats)
    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
