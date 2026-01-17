"""Event bus implementations for GovGuard."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from typing import Any, Protocol
from uuid import UUID


@dataclass(slots=True)
class Event:
    """Event envelope for the internal event bus."""

    event_id: UUID
    event_type: str
    candidate_id: UUID
    payload: dict[str, Any]
    correlation_id: UUID | None = None


class EventBus(Protocol):
    """Abstract event bus interface."""

    def publish(self, event: Event) -> None:
        """Publish an event to the bus."""

    def subscribe(self, event_type: str) -> Iterator[Event]:
        """Yield events of the given type as they arrive."""

    def next_event(self, event_type: str, timeout: float | None = None) -> Event | None:
        """Return the next event or None if timed out."""


class InMemoryEventBus:
    """In-process event bus with per-topic queues."""

    def __init__(self) -> None:
        self._queues: dict[str, Queue[Event]] = {}
        self._lock = Lock()

    def _queue_for(self, event_type: str) -> Queue[Event]:
        with self._lock:
            if event_type not in self._queues:
                self._queues[event_type] = Queue()
            return self._queues[event_type]

    def publish(self, event: Event) -> None:
        self._queue_for(event.event_type).put(event)

    def subscribe(self, event_type: str) -> Iterator[Event]:
        queue = self._queue_for(event_type)
        while True:
            yield queue.get()

    def next_event(self, event_type: str, timeout: float | None = None) -> Event | None:
        queue = self._queue_for(event_type)
        try:
            return queue.get(timeout=timeout)
        except Empty:
            return None
