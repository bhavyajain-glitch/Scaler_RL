"""
failures.py — FailureEvent and FailureInjector

Fault injection engine for OnCallEnv.
Applies failure state deltas to a ServiceRegistry and populates each service's
log ring-buffer with realistic, verbose messages (via log_templates).
Deterministic given a seed.
"""

from __future__ import annotations

import random
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, PrivateAttr

from env.services import ServiceRegistry
from env.log_templates import get_failure_logs


# ── Severity multipliers (low / medium / high) ───────────────────────────────

_SEVERITY = {
    "low":    0.4,
    "medium": 0.7,
    "high":   1.0,
}

# CPU target ranges per severity so the spike is always visibly degraded/down
_CPU_SPIKE_RANGE = {
    "low":    (55.0, 75.0),   # often healthy, occasionally degraded
    "medium": (75.0, 88.0),   # degraded
    "high":   (88.0, 99.0),   # degraded → down
}


# ── FailureEvent ──────────────────────────────────────────────────────────────

class FailureEvent(BaseModel):
    """Immutable description of a single failure to inject."""

    service: str
    failure_type: Literal["cpu_spike", "oom", "crash", "latency", "db_lock"]
    severity: Literal["low", "medium", "high"]
    cascade_to: Optional[List[str]] = None
    auto_heal_after: Optional[int] = None   # None → manual only

    model_config = {"frozen": True}


# ── FailureInjector ───────────────────────────────────────────────────────────

class FailureInjector(BaseModel):
    """
    Injects, cascades, and auto-heals failures on a ServiceRegistry.

    Internal state is kept in private attrs so the injector itself is
    never accidentally serialised into an observation.
    """

    _rng: random.Random = PrivateAttr(default_factory=random.Random)
    _active: Dict[str, List[Tuple[FailureEvent, Optional[int]]]] = PrivateAttr(
        default_factory=dict,
    )

    # ── Seed / reset ──────────────────────────────────────────────────────

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the RNG and clear all tracked failures."""
        if seed is not None:
            self._rng.seed(seed)
        self._active.clear()

    # ── Inject ────────────────────────────────────────────────────────────

    def inject(self, event: FailureEvent, registry: ServiceRegistry) -> None:
        """Apply *event* to the target service in *registry*."""
        svc = registry.get_service(event.service)
        if svc is None:
            return

        mult = _SEVERITY[event.severity]

        # ── Metric mutations ──
        if event.failure_type == "cpu_spike":
            lo, hi = _CPU_SPIKE_RANGE[event.severity]
            svc.cpu_pct = self._rng.uniform(lo, hi)

        elif event.failure_type == "oom":
            svc.memory_pct = min(100.0, svc.memory_pct + self._rng.uniform(40, 70) * mult)
            svc.error_rate = 1.0
            svc.status = "down"

        elif event.failure_type == "crash":
            svc.error_rate = 1.0
            svc.status = "down"

        elif event.failure_type == "latency":
            svc.latency_ms += self._rng.uniform(200, 800) * mult
            svc.error_rate = min(1.0, svc.error_rate + self._rng.uniform(0.1, 0.3) * mult)

        elif event.failure_type == "db_lock":
            svc.error_rate = min(1.0, svc.error_rate + self._rng.uniform(0.3, 0.6) * mult)
            svc.latency_ms += self._rng.uniform(300, 1_000) * mult

        # ── Derive status from metrics ──
        svc.derive_status()

        # ── Emit realistic log messages ──
        for level, msg in get_failure_logs(event.failure_type, event.severity, self._rng):
            svc.emit_log(level, msg)  # type: ignore[arg-type]

        # ── Track for cascading / auto-heal ──
        if event.cascade_to or event.auto_heal_after is not None:
            self._active.setdefault(event.service, []).append(
                (event, event.auto_heal_after),
            )

    # ── Manual clear ──────────────────────────────────────────────────────

    def clear_service(self, service_name: str) -> None:
        """Remove all tracked failures for *service_name*.

        Called when the agent manually heals a service (restart / rollback)
        so that cascade events sourced from that service stop firing.
        """
        self._active.pop(service_name, None)

    # ── Cascade ───────────────────────────────────────────────────────────

    def cascade(self, registry: ServiceRegistry) -> None:
        """Propagate degradation to downstream services listed in cascade_to."""
        for src_name, entries in list(self._active.items()):
            for event, _ in entries:
                if not event.cascade_to:
                    continue
                for dst_name in event.cascade_to:
                    dst = registry.get_service(dst_name)
                    if dst is None or dst.status != "healthy":
                        continue
                    dst.error_rate = min(
                        1.0, dst.error_rate + self._rng.uniform(0.05, 0.15),
                    )
                    dst.latency_ms += self._rng.uniform(50, 200)
                    dst.derive_status()
                    dst.emit_log(
                        "WARN",
                        f"Degraded by upstream failure in {src_name}. "
                        f"Fix {src_name} first before restarting this service.",
                    )

    # ── Auto-heal tick ────────────────────────────────────────────────────

    def tick_heal(self, registry: ServiceRegistry) -> None:
        """Decrement heal timers; heal services whose timer reaches zero."""
        for svc_name in list(self._active):
            remaining: List[Tuple[FailureEvent, Optional[int]]] = []
            for event, steps_left in self._active[svc_name]:
                if steps_left is None:
                    remaining.append((event, None))
                    continue
                steps_left -= 1
                if steps_left <= 0:
                    svc = registry.get_service(svc_name)
                    if svc is not None:
                        svc.reset_to_healthy()
                        svc.emit_log("INFO", f"Auto-healed after {event.failure_type}.")
                else:
                    remaining.append((event, steps_left))
            if remaining:
                self._active[svc_name] = remaining
            else:
                del self._active[svc_name]
