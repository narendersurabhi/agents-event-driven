"""Pluggable persistence for pipeline job state.

The PipelineStore protocol allows different backends (in-memory, SQLite,
Redis, etc.) to store per-job snapshots. The orchestrator uses this to
persist and reload PipelineState by job_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import json


class PipelineStore(Protocol):
    """Abstract storage for pipeline job snapshots."""

    def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored snapshot for job_id, or None if not found."""

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        """Persist the given snapshot for job_id."""

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return up to `limit` job snapshots (backend-defined ordering)."""


@dataclass
class InMemoryPipelineStore:
    """In-memory PipelineStore implementation for dev/test.

    This is not durable across process restarts, but exercises the same
    interface as a real database-backed implementation.
    """

    _db: Dict[str, Dict[str, Any]]

    def __init__(self) -> None:
        self._db = {}

    def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._db.get(job_id)

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        snapshot = dict(state)
        snapshot.setdefault("job_id", job_id)
        self._db[job_id] = snapshot

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self._db.values())[:limit]


@dataclass
class JsonFilePipelineStore:
    """Persist job snapshots as JSON files in a local directory.

    Each job is stored as <root>/<job_id>.json. This is suitable for local
    development, keeping pipeline history across process restarts without
    requiring a full database.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def load(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(job_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, job_id: str, state: Dict[str, Any]) -> None:
        path = self._path_for(job_id)
        tmp = path.with_suffix(".json.tmp")
        snapshot = dict(state)
        snapshot.setdefault("job_id", job_id)
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in paths[:limit]:
            try:
                with p.open("r", encoding="utf-8") as f:
                    snapshots.append(json.load(f))
            except json.JSONDecodeError:
                continue
        return snapshots
