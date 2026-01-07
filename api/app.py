# api/app.py
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from core.obs import JsonRepoLogger  # <- from the logging/obs module we created
from core.llm_factory import get_async_llm_client

# Existing router that exposes /tailor-resume
from api.tailor import router as tailor_router
# New router that exposes the event-driven pipeline
from api.pipeline import router as pipeline_router


APP_ENV: str = os.getenv("APP_ENV", "dev")
SERVICE_NAME: str = os.getenv("SERVICE_NAME", "tailor-api")

# CORS origins (comma-separated), e.g. "http://localhost:3000,https://your.app"
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- startup -----
    logger = JsonRepoLogger(service=SERVICE_NAME, env=APP_ENV)
    app.state.logger = logger

    # Single shared async OpenAI client with DI-friendly logger
    app.state.llm = get_async_llm_client(logger=logger)

    logger.info("service.start", env=APP_ENV, service=SERVICE_NAME)
    try:
        yield
    finally:
        # ----- shutdown -----
        logger.info("service.stop", env=APP_ENV, service=SERVICE_NAME)


app = FastAPI(
    title="Resume Tailoring API",
    version=os.getenv("APP_VERSION", "0.1.0"),
    description="JD → Plan → Compose (tailored resume) pipeline",
    lifespan=lifespan,
)


# ----- Middleware -----

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Attach a request ID to every request/response and log basic access info.
    """
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.req_id = req_id

    # Basic access log (you can expand with timing, user info, etc.)
    app.state.logger.info(
        "http.request",
        req_id=req_id,
        method=request.method,
        path=request.url.path,
        client=str(request.client.host if request.client else None),
    )

    response: Response = await call_next(request)
    response.headers["x-request-id"] = req_id

    app.state.logger.info(
        "http.response",
        req_id=req_id,
        status_code=response.status_code,
        path=request.url.path,
    )
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"] if APP_ENV == "dev" else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Simple health & root -----

@app.get("/", tags=["meta"])
async def root(request: Request):
    return {
        "service": SERVICE_NAME,
        "env": APP_ENV,
        "version": app.version,
        "request_id": getattr(request.state, "req_id", None),
    }


@app.get("/healthz", tags=["meta"])
async def healthz():
    # Optionally: check downstreams, env, OpenAI key present, etc.
    return {"status": "ok"}


# ----- Routers -----

app.include_router(tailor_router, tags=["tailor"])
app.include_router(pipeline_router, tags=["pipeline"])
