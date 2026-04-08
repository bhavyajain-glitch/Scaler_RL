"""
failures.py — FailureEvent and FailureInjector

Defines the fault injection engine for OnCallEnv.
"""

from __future__ import annotations

import random
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from env.services import ServiceRegistry  # Import ServiceRegistry


# ---------------------------------------------------------------------------
# FailureEvent
# ---------------------------------------------------------------------------


class FailureEvent(BaseModel):
    """Describes a single failure to be injected into a service."""

    service: str
    failure_type: Literal["cpu_spike", "oom", "crash", "latency", "db_lock"]
    severity: Literal["low", "medium", "high"]
    cascade_to: Optional[List[str]] = None  # downstream services affected
    auto_heal_after: Optional[int] = None  # steps until self-heal (None = manual only)

    class Config:
        frozen = True  # failure events are immutable


# ---------------------------------------------------------------------------
# FailureInjector
# ---------------------------------------------------------------------------


class FailureInjector(BaseModel):
    """
    Applies failure state deltas to ServiceRegistry.
    Deterministic given a seed.
    """

    _rng: random.Random = Field(default=None, exclude=True)
    _active_failures: Dict[str, List[FailureEvent]] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        object.__setattr__(self, "_rng", random.Random())

    def seed(self, seed: Optional[int] = None) -> None:
        """Seeds the internal RNG."""
        if seed is not None:
            self._rng.seed(seed)

    def inject(self, event: FailureEvent, registry: ServiceRegistry) -> None:
        """Applies a failure event to the target service."""
        service = registry.get_service(event.service)
        if not service: # Handle case where service might not exist
            print(f"Service {event.service} not found in registry. Skipping failure injection.")
            return

        service.emit_log("ERROR", f"Injecting failure: {event.failure_type} - {event.severity}")

        if event.failure_type == "cpu_spike":
            service.cpu_pct = min(100.0, service.cpu_pct + self._rng.uniform(40.0, 80.0))
        elif event.failure_type == "oom":
            service.memory_pct = min(100.0, service.memory_pct + self._rng.uniform(50.0, 90.0))
            service.status = "down" # OOM often brings down a service
            service.emit_log("CRITICAL", "Service experienced Out Of Memory and crashed.")
        elif event.failure_type == "crash":
            service.status = "down"
            service.error_rate = 1.0
            service.emit_log("CRITICAL", "Service crashed unexpectedly.")
        elif event.failure_type == "latency":
            service.latency_ms = service.latency_ms + self._rng.uniform(200.0, 800.0)
            service.error_rate = min(1.0, service.error_rate + self._rng.uniform(0.1, 0.4))
        elif event.failure_type == "db_lock":
            service.error_rate = min(1.0, service.error_rate + self._rng.uniform(0.3, 0.6))
            service.latency_ms = service.latency_ms + self._rng.uniform(300.0, 1000.0)
            service.emit_log("WARN", "Database lock contention detected.")

        # Add to active failures for auto-healing and cascading
        if event.auto_heal_after is not None or event.cascade_to:
            self._active_failures.setdefault(event.service, []).append(event)

    def cascade(self, registry: ServiceRegistry) -> None:
        """Propagates failure effects to downstream services."""
        for service_name, events in list(self._active_failures.items()):
            for event in events:
                if event.cascade_to:
                    for downstream_service_name in event.cascade_to:
                        downstream_service = registry.get_service(downstream_service_name)
                        if downstream_service and downstream_service.status == "healthy":
                            # Simple cascade: make downstream degraded/latency
                            downstream_service.error_rate = min(1.0, downstream_service.error_rate + self._rng.uniform(0.05, 0.2))
                            downstream_service.latency_ms = downstream_service.latency_ms + self._rng.uniform(50.0, 200.0)
                            downstream_service.emit_log("WARN", f"Experiencing degraded performance due to upstream failure in {service_name}.")

    def tick_heal(self, registry: ServiceRegistry) -> None:
        """Decrements auto-heal timers and resolves failures if due."""
        removed_failures = []
        for service_name, events in list(self._active_failures.items()):
            remaining_events = []
            for event in events:
                if event.auto_heal_after is not None:
                    event.auto_heal_after -= 1  # type: ignore
                    if event.auto_heal_after <= 0:
                        service = registry.get_service(service_name)
                        if service:
                            # For simplicity, a basic heal: reset some metrics
                            service.cpu_pct = 10.0
                            service.memory_pct = 20.0
                            service.error_rate = 0.0
                            service.latency_ms = 50.0
                            service.emit_log("INFO", f"Auto-healing complete for {event.failure_type} failure.")
                        removed_failures.append((service_name, event)) # Mark for removal
                    else:
                        remaining_events.append(event)
                else:
                    remaining_events.append(event)
            self._active_failures[service_name] = remaining_events

        # Remove failures that have been healed
        for service_name, event in removed_failures:
            if not self._active_failures[service_name]:
                del self._active_failures[service_name]
