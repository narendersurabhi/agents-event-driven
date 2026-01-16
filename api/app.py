# api/app.py
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# New router that exposes the event-driven pipeline
from api.pipeline import router as pipeline_router

# Existing router that exposes /tailor-resume
from api.tailor import router as tailor_router
from core.llm_factory import get_async_llm_client
from core.obs import (
    JsonRepoLogger,  # <- from the logging/obs module we created
    bind_log_context,
)
from core.settings import get_app_settings

SETTINGS = get_app_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ----- startup -----
    logger = JsonRepoLogger(service=SETTINGS.service_name, env=SETTINGS.app_env)
    app.state.logger = logger

    # Single shared async OpenAI client with DI-friendly logger
    app.state.llm = get_async_llm_client(logger=logger)

    logger.info("service.start", env=SETTINGS.app_env, service=SETTINGS.service_name)
    try:
        yield
    finally:
        # ----- shutdown -----
        logger.info("service.stop", env=SETTINGS.app_env, service=SETTINGS.service_name)


app = FastAPI(
    title="Resume Tailoring API",
    version=SETTINGS.app_version,
    description="JD → Plan → Compose (tailored resume) pipeline",
    lifespan=lifespan,
)


# ----- Middleware -----


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Attach a request ID to every request/response and log basic access info.
    """
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.req_id = req_id

    # Basic access log (you can expand with timing, user info, etc.)
    with bind_log_context(req_id=req_id):
        app.state.logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            client=str(request.client.host if request.client else None),
        )

        response: Response = await call_next(request)
        response.headers["x-request-id"] = req_id

        app.state.logger.info(
            "http.response",
            status_code=response.status_code,
            path=request.url.path,
        )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.cors_allowlist(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Simple health & root -----


@app.get("/", tags=["meta"])
async def root(request: Request) -> dict[str, str | None]:
    return {
        "service": SETTINGS.service_name,
        "env": SETTINGS.app_env,
        "version": app.version,
        "request_id": getattr(request.state, "req_id", None),
    }


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict[str, str]:
    # Optionally: check downstreams, env, OpenAI key present, etc.
    return {"status": "ok"}


# ----- Routers -----

app.include_router(tailor_router, tags=["tailor"])
app.include_router(pipeline_router, tags=["pipeline"])
