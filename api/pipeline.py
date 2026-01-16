"""FastAPI router that exposes the event-driven pipeline via HTTP.

Endpoint:
  POST /pipeline/run

Body:
  {
    "jd_text": "...",
    "resume_text": "...",
    "run_qa": true,
    "run_improver": true
  }

Response:
  The payload from `pipeline.completed`, containing:
  {
    "jd": {...},
    "profile": {...},
    "plan": {...},
    "tailored": {...},
    "qa": {...} | null,
    "improved": {...}
  }
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import uuid

import anyio
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from agents.cover_letter_agent import CoverLetterAgent
from agents.cover_letter_worker import CoverLetterWorker
from agents.docx_render_agent import DocxRenderAgent
from agents.jd_analysis import JDAnalysisAgent
from agents.jd_worker import JDWorker
from agents.match_planner import MatchPlannerAgent
from agents.match_worker import MatchWorker
from agents.profile_from_resume import ProfileFromResumeAgent
from agents.profile_worker import ProfileWorker
from agents.qa_improver import QAImproveAgent
from agents.qa_improver_worker import QAImproveWorker
from agents.resume_composer import ResumeComposerAgent
from agents.resume_composer_worker import ResumeComposerWorker
from agents.resume_qa import ResumeQAAgent
from agents.resume_qa_worker import ResumeQAWorker
from core.config import get_default_model
from core.events import Event, EventBus, InMemoryEventBus
from core.llm_factory import get_sync_llm_client
from core.llm_step_worker import LLMStepWorker
from core.models import CoverLetter, TailoredResume
from core.pipeline_orchestrator import (
    PIPELINE_COMPLETED,
    PIPELINE_RESTART_COMPOSE,
    PIPELINE_RESUME,
    PIPELINE_START,
    PipelineOrchestrator,
)
from core.pipeline_store import JsonFilePipelineStore

logger = logging.getLogger(__name__)

router = APIRouter()


class PipelineRequest(BaseModel):
    jd_text: str
    resume_text: str
    run_qa: bool = True
    run_improver: bool = True
    force_profile_refresh: bool = False


@dataclass
class _PipelineRuntime:
    """Holds the shared bus, workers, and orchestrator for the API."""

    bus: EventBus
    orchestrator: PipelineOrchestrator


_runtime: _PipelineRuntime | None = None
_runtime_lock = threading.Lock()


def _ensure_runtime() -> _PipelineRuntime:
    global _runtime  # noqa: PLW0603
    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        bus = InMemoryEventBus()
        model_name = get_default_model()
        # Choose LLM implementation based on configuration (LLM_PROVIDER, etc.).
        llm = get_sync_llm_client()
        # For local development, persist job snapshots under ./pipeline_jobs
        store_root = Path("pipeline_jobs")
        store = JsonFilePipelineStore(root=store_root)

        # Agents
        jd_agent = JDAnalysisAgent(llm=llm, model=model_name)
        profile_agent = ProfileFromResumeAgent(llm=llm, model=model_name)
        match_agent = MatchPlannerAgent(llm=llm, model=model_name)
        composer_agent = ResumeComposerAgent(llm=llm, model=model_name)
        qa_agent = ResumeQAAgent(llm=llm, model=model_name)
        qa_improve_agent = QAImproveAgent(llm=llm, model=model_name)
        cover_letter_agent = CoverLetterAgent(llm=llm, model=model_name)

        # Workers
        llm_worker = LLMStepWorker(bus=bus, llm=llm, model=model_name)
        jd_worker = JDWorker(bus=bus, agent=jd_agent)
        profile_worker = ProfileWorker(bus=bus, agent=profile_agent)
        match_worker = MatchWorker(bus=bus, agent=match_agent)
        composer_worker = ResumeComposerWorker(bus=bus, agent=composer_agent)
        qa_worker = ResumeQAWorker(bus=bus, agent=qa_agent)
        qa_improve_worker = QAImproveWorker(bus=bus, agent=qa_improve_agent)
        cover_letter_worker = CoverLetterWorker(bus=bus, agent=cover_letter_agent)
        orchestrator = PipelineOrchestrator(bus=bus, store=store)

        # Start workers and orchestrator loops in background threads.
        threading.Thread(target=llm_worker.run_forever, daemon=True).start()
        threading.Thread(target=jd_worker.run_jd_requests, daemon=True).start()
        threading.Thread(target=jd_worker.run_llm_results, daemon=True).start()
        threading.Thread(target=profile_worker.run_profile_requests, daemon=True).start()
        threading.Thread(target=profile_worker.run_llm_results, daemon=True).start()
        threading.Thread(target=match_worker.run_match_requests, daemon=True).start()
        threading.Thread(target=match_worker.run_llm_results, daemon=True).start()
        threading.Thread(target=composer_worker.run_compose_requests, daemon=True).start()
        threading.Thread(target=composer_worker.run_llm_results, daemon=True).start()
        threading.Thread(target=qa_worker.run_qa_requests, daemon=True).start()
        threading.Thread(target=qa_worker.run_llm_results, daemon=True).start()
        threading.Thread(target=qa_improve_worker.run_improve_requests, daemon=True).start()
        threading.Thread(target=qa_improve_worker.run_llm_results, daemon=True).start()
        threading.Thread(target=cover_letter_worker.run_cover_letter_requests, daemon=True).start()
        threading.Thread(target=cover_letter_worker.run_llm_results, daemon=True).start()

        threading.Thread(target=orchestrator.run_pipeline_start, daemon=True).start()
        threading.Thread(target=orchestrator.run_jd_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_profile_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_match_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_compose_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_qa_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_qa_improve_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_cover_letter_completed, daemon=True).start()
        threading.Thread(target=orchestrator.run_pipeline_resume, daemon=True).start()
        threading.Thread(target=orchestrator.run_pipeline_restart_compose, daemon=True).start()

        logger.info("pipeline_runtime.started")
        _runtime = _PipelineRuntime(bus=bus, orchestrator=orchestrator)
        return _runtime


async def _wait_for_completion(bus: EventBus, correlation_id: str) -> dict:
    """Block in a worker thread until pipeline.completed arrives."""

    def _wait() -> dict:
        for event in bus.subscribe(PIPELINE_COMPLETED):
            if event.correlation_id == correlation_id:
                return event.payload
        # Should never happen with infinite subscribe loop.
        return {}

    return await anyio.to_thread.run_sync(_wait)


@router.post("/pipeline/run")
async def run_pipeline(req: PipelineRequest) -> dict:
    """Start the full pipeline for the given JD + resume.

    Returns immediately with a job_id; clients should poll /pipeline/status
    (and optionally /pipeline/resume) to track progress.
    """
    runtime = _ensure_runtime()
    bus = runtime.bus

    cid = str(uuid.uuid4())
    logger.info(
        "pipeline_api.start cid=%s run_qa=%s run_improver=%s",
        cid,
        req.run_qa,
        req.run_improver,
    )

    bus.publish(
        Event(
            type=PIPELINE_START,
            payload={
                "jd_text": req.jd_text,
                "resume_text": req.resume_text,
                "run_qa": req.run_qa,
                "run_improver": req.run_improver,
                "force_profile_refresh": req.force_profile_refresh,
            },
            correlation_id=cid,
        )
    )

    logger.info("pipeline_api.started cid=%s", cid)
    return {"job_id": cid}


@router.post("/pipeline/resume/{job_id}")
async def resume_pipeline(job_id: str) -> dict:
    """Resume a previously started pipeline for the given job_id.

    This is in-process only: it assumes the service has not restarted and
    the in-memory PipelineState for job_id is still present.
    """
    runtime = _ensure_runtime()
    bus = runtime.bus

    logger.info("pipeline_api.resume cid=%s", job_id)
    bus.publish(Event(type=PIPELINE_RESUME, payload={}, correlation_id=job_id))
    # Resume is fire-and-forget; clients should poll status.
    return {"job_id": job_id}


@router.post("/pipeline/restart-compose/{job_id}")
async def restart_compose(job_id: str) -> dict:
    """Restart from the compose step for a given job_id.

    This clears any existing tailored/qa state and re-runs compose, then lets
    the pipeline proceed through QA and QA improver as configured.
    """
    runtime = _ensure_runtime()
    bus = runtime.bus
    logger.info("pipeline_api.restart_compose cid=%s", job_id)
    bus.publish(Event(type=PIPELINE_RESTART_COMPOSE, payload={}, correlation_id=job_id))
    return {"job_id": job_id}


@router.get("/pipeline/status/{job_id}")
async def pipeline_status(job_id: str) -> dict:
    """Return the latest known status and artifacts for a job."""
    runtime = _ensure_runtime()
    snapshot = runtime.orchestrator.get_state_snapshot(job_id)
    if snapshot is None:
        # Try to load directly from the persistent store (e.g., after restart).
        snap_from_store = runtime.orchestrator.store.load(job_id)
        if snap_from_store is not None:
            snapshot = snap_from_store
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **snapshot}


@router.get("/pipeline/jobs")
async def list_jobs(limit: int = 50) -> list[dict]:
    """List recent pipeline jobs (best-effort order, most recent first)."""
    runtime = _ensure_runtime()
    jobs = runtime.orchestrator.store.list_jobs(limit=limit)
    return jobs


@router.get("/pipeline/{job_id}/docx")
async def download_docx(job_id: str) -> Response:
    """Render and download a DOCX resume for the given job_id.

    Uses the final TailoredResume (improved if present, otherwise composed)
    and a local .docx template to generate the document.
    """
    runtime = _ensure_runtime()
    snapshot = runtime.orchestrator.get_state_snapshot(job_id)
    if snapshot is None:
        snapshot = runtime.orchestrator.store.load(job_id)
    if snapshot is None:
        logger.error("docx.render.unknown_job cid=%s", job_id)
        raise HTTPException(status_code=404, detail="Unknown job_id")

    resume_data = snapshot.get("improved") or snapshot.get("tailored")
    if resume_data is None:
        logger.error(
            "docx.render.no_resume cid=%s snapshot_keys=%s",
            job_id,
            list(snapshot.keys()),
        )
        raise HTTPException(status_code=400, detail="No resume available for this job_id")

    try:
        resume = TailoredResume.model_validate(resume_data)
    except Exception as exc:  # noqa: BLE001
        logger.error("docx.render.invalid_resume cid=%s error=%s", job_id, str(exc))
        raise HTTPException(status_code=500, detail=f"Invalid TailoredResume data: {exc}") from exc

    # Load template bytes; you can change the path or make it configurable.
    template_path = Path("templates/resume_template.docx")
    if not template_path.exists():
        logger.error(
            "docx.render.template_missing cid=%s path=%s",
            job_id,
            str(template_path),
        )
        raise HTTPException(status_code=500, detail=f"Template not found at {template_path}")
    tpl_bytes = template_path.read_bytes()

    try:
        renderer = DocxRenderAgent()
        docx_bytes = renderer.render(tpl_bytes, resume)
    except Exception as exc:  # noqa: BLE001
        logger.error("docx.render.failed cid=%s error=%s", job_id, str(exc))
        raise HTTPException(status_code=500, detail=f"DOCX render failed: {exc}") from exc

    # Best-effort: save a copy locally for troubleshooting.
    try:
        debug_dir = Path("debug_docx")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{job_id}.docx"
        debug_path.write_bytes(docx_bytes)
        logger.info("docx.debug_saved cid=%s path=%s", job_id, str(debug_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("docx.debug_save_failed cid=%s error=%s", job_id, str(exc))

    filename = f"resume-{job_id}.docx"
    logger.info("docx.render.success cid=%s filename=%s", job_id, filename)
    return Response(
        content=docx_bytes,
        media_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/pipeline/{job_id}/cover-letter.docx")
async def download_cover_letter_docx(job_id: str) -> Response:
    """Render and download a DOCX cover letter for the given job_id.

    Uses the final CoverLetter stored in the pipeline state and a local .docx
    template to generate the document.
    """
    runtime = _ensure_runtime()
    snapshot = runtime.orchestrator.get_state_snapshot(job_id)
    if snapshot is None:
        snapshot = runtime.orchestrator.store.load(job_id)
    if snapshot is None:
        logger.error("cover_docx.render.unknown_job cid=%s", job_id)
        raise HTTPException(status_code=404, detail="Unknown job_id")

    cover_data = snapshot.get("cover_letter")
    if cover_data is None:
        logger.error(
            "cover_docx.render.no_cover cid=%s snapshot_keys=%s",
            job_id,
            list(snapshot.keys()),
        )
        raise HTTPException(
            status_code=400,
            detail="No cover letter available for this job_id",
        )

    try:
        cover = CoverLetter.model_validate(cover_data)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "cover_docx.render.invalid_cover cid=%s error=%s",
            job_id,
            str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Invalid CoverLetter data: {exc}",
        ) from exc

    template_path = Path("templates/cover_letter_template.docx")
    if not template_path.exists():
        logger.error(
            "cover_docx.render.template_missing cid=%s path=%s",
            job_id,
            str(template_path),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Template not found at {template_path}",
        )
    tpl_bytes = template_path.read_bytes()

    try:
        renderer = DocxRenderAgent()
        docx_bytes = renderer.render(tpl_bytes, cover)
    except Exception as exc:  # noqa: BLE001
        logger.error("cover_docx.render.failed cid=%s error=%s", job_id, str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"DOCX render failed: {exc}",
        ) from exc

    # Best-effort: save a copy locally for troubleshooting.
    try:
        debug_dir = Path("debug_docx")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{job_id}-cover.docx"
        debug_path.write_bytes(docx_bytes)
        logger.info("cover_docx.debug_saved cid=%s path=%s", job_id, str(debug_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cover_docx.debug_save_failed cid=%s error=%s",
            job_id,
            str(exc),
        )

    base_name = (cover.full_name or "Cover Letter").strip()
    safe_name = base_name.replace("/", "-").replace("\\", "-")
    filename = f"{safe_name} Cover Letter.docx"
    logger.info("cover_docx.render.success cid=%s filename=%s", job_id, filename)
    return Response(
        content=docx_bytes,
        media_type=("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
