"""Event recording helper."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from govguard.orchestrator.event_bus import Event, EventBus


@dataclass(slots=True)
class RecordingEventBus:
    """Wraps another EventBus and records published events."""

    bus: EventBus
    events: list[Event] = field(default_factory=list)

    def publish(self, event: Event) -> None:
        self.events.append(event)
        self.bus.publish(event)

    def subscribe(self, event_type: str) -> Iterator[Event]:
        return self.bus.subscribe(event_type)

    def next_event(self, event_type: str, timeout: float | None = None) -> Event | None:
        return self.bus.next_event(event_type, timeout=timeout)
