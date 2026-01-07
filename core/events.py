"""Simple event model and in-memory event bus.

This provides a pluggable EventBus abstraction so agents and workers can
communicate via events. The default implementation is an in-process
queue-based bus; other backends (Redis, SQS, etc.) can be added later
by implementing the EventBus protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, Iterator, Optional, Protocol


@dataclass(slots=True)
class Event:
    """A single event on the bus."""

    type: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


class EventBus(Protocol):
    """Abstract event bus interface."""

    def publish(self, event: Event) -> None:
        """Publish an event to the bus."""

    def subscribe(self, event_type: str) -> Iterator[Event]:
        """Yield events of the given type as they arrive."""


class InMemoryEventBus:
    """In-process implementation of EventBus using per-type queues.

    This is suitable for local development and tests. It can be swapped
    out for a real backend (Redis, SQS, Kafka, etc.) by providing another
    EventBus implementation with the same interface.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, Queue[Event]] = {}

    def _queue_for(self, event_type: str) -> Queue[Event]:
        if event_type not in self._queues:
            self._queues[event_type] = Queue()
        return self._queues[event_type]

    def publish(self, event: Event) -> None:
        q = self._queue_for(event.type)
        q.put(event)

    def subscribe(self, event_type: str) -> Iterator[Event]:
        q = self._queue_for(event_type)
        while True:
            yield q.get()

