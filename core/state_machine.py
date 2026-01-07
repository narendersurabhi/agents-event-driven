from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol


class StateMachineBackend(Protocol):
    """Adapter interface so callers can swap FSM libraries without rewriting orchestration logic."""

    def add_state(self, name: str) -> None: ...
    def add_transition(self, trigger: str, source: str, dest: str, on_transition: Callable[[], None] | None = None) -> None: ...
    def set_state(self, name: str) -> None: ...
    def trigger(self, trigger: str) -> None: ...
    @property
    def state(self) -> str: ...


# ---------- Lightweight in-process backend ----------


@dataclass
class _Transition:
    trigger: str
    source: str
    dest: str
    on_transition: Callable[[], None] | None


class SimpleStateMachine(StateMachineBackend):
    """
    Minimal FSM suitable for unit tests or local orchestration.
    Not thread-safe; no async support; raises on invalid triggers.
    """

    def __init__(self) -> None:
        self._states: List[str] = []
        self._transitions: List[_Transition] = []
        self._state: Optional[str] = None

    def add_state(self, name: str) -> None:
        if name not in self._states:
            self._states.append(name)

    def add_transition(self, trigger: str, source: str, dest: str, on_transition: Callable[[], None] | None = None) -> None:
        self._transitions.append(_Transition(trigger=trigger, source=source, dest=dest, on_transition=on_transition))

    def set_state(self, name: str) -> None:
        if name not in self._states:
            raise ValueError(f"Unknown state: {name}")
        self._state = name

    def trigger(self, trigger: str) -> None:
        if self._state is None:
            raise RuntimeError("State machine not initialized; call set_state() first")
        for t in self._transitions:
            if t.trigger == trigger and t.source == self._state:
                if t.on_transition:
                    t.on_transition()
                self._state = t.dest
                return
        raise RuntimeError(f"No transition for trigger '{trigger}' from state '{self._state}'")

    @property
    def state(self) -> str:
        if self._state is None:
            raise RuntimeError("State machine not initialized; call set_state() first")
        return self._state


# ---------- Optional 'transitions' backend ----------


class TransitionsBackend(StateMachineBackend):
    """
    Adapter around the 'transitions' library (install separately) to match our interface.
    """

    def __init__(self) -> None:
        try:
            from transitions import Machine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install 'transitions' to use TransitionsBackend") from exc
        self._Machine = Machine
        self._states: List[str] = []
        self._transitions: List[Dict[str, str | Callable[[], None] | List[str]]] = []
        self._machine: Machine | None = None

    def add_state(self, name: str) -> None:
        if name not in self._states:
            self._states.append(name)

    def add_transition(self, trigger: str, source: str, dest: str, on_transition: Callable[[], None] | None = None) -> None:
        self._transitions.append({"trigger": trigger, "source": source, "dest": dest, "after": on_transition})

    def set_state(self, name: str) -> None:
        self._machine = self._Machine(model=self, states=self._states, transitions=self._transitions, initial=name)

    def trigger(self, trigger: str) -> None:
        if self._machine is None:
            raise RuntimeError("State machine not initialized; call set_state() first")
        # 'transitions' binds triggers as methods on the model
        fn = getattr(self, trigger, None)
        if not callable(fn):
            raise RuntimeError(f"No trigger '{trigger}' found")
        fn()

    @property
    def state(self) -> str:
        if self._machine is None:
            raise RuntimeError("State machine not initialized; call set_state() first")
        return self.state  # type: ignore[attr-defined]
