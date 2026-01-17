"""NATS-backed event bus implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import json
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any
from uuid import UUID

from nats.aio.client import Client as NATS

from govguard.orchestrator.event_bus import Event, EventBus


class NATSEventBus(EventBus):
    """NATS event bus with a background asyncio loop."""

    def __init__(self, url: str, subject_prefix: str = "govguard") -> None:
        self._url = url
        self._prefix = subject_prefix
        self._queues: dict[str, Queue[Event]] = {}
        self._lock = Lock()
        self._loop = asyncio.new_event_loop()
        self._client = NATS()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        await self._client.connect(servers=[self._url])

    def _queue_for(self, event_type: str) -> Queue[Event]:
        with self._lock:
            if event_type not in self._queues:
                self._queues[event_type] = Queue()
                asyncio.run_coroutine_threadsafe(
                    self._subscribe(event_type),
                    self._loop,
                )
            return self._queues[event_type]

    async def _subscribe(self, event_type: str) -> None:
        subject = f"{self._prefix}.{event_type}"

        async def handler(msg: Any) -> None:
            data = json.loads(msg.data.decode("utf-8"))
            event = Event(
                event_id=UUID(data["event_id"]),
                event_type=data["event_type"],
                candidate_id=UUID(data["candidate_id"]),
                payload=data["payload"],
                correlation_id=UUID(data["correlation_id"]) if data["correlation_id"] else None,
            )
            self._queue_for(event_type).put(event)

        await self._client.subscribe(subject, cb=handler)

    def publish(self, event: Event) -> None:
        payload = json.dumps(
            {
                "event_id": str(event.event_id),
                "event_type": event.event_type,
                "candidate_id": str(event.candidate_id),
                "payload": event.payload,
                "correlation_id": str(event.correlation_id) if event.correlation_id else None,
            }
        ).encode("utf-8")
        subject = f"{self._prefix}.{event.event_type}"
        asyncio.run_coroutine_threadsafe(self._client.publish(subject, payload), self._loop)

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
