"""
failures.py — FailureEvent and FailureInjector

Fault injection engine for OnCallEnv.
Applies failure state deltas to a ServiceRegistry.  Deterministic given a seed.
"""

from __future__ import annotations

import random
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, PrivateAttr

from env.services import ServiceRegistry


# ── Severity multipliers (low / medium / high) ──────────────────────────────

_SEVERITY = {
    "low":    0.4,
    "medium": 0.7,
    "high":   1.0,
}


# ── FailureEvent ─────────────────────────────────────────────────────────────

class FailureEvent(BaseModel):
    """Immutable description of a single failure to inject."""

    service: str
    failure_type: Literal["cpu_spike", "oom", "crash", "latency", "db_lock"]
    severity: Literal["low", "medium", "high"]
    cascade_to: Optional[List[str]] = None
    auto_heal_after: Optional[int] = None   # None → manual only

    model_config = {"frozen": True}


# ── FailureInjector ──────────────────────────────────────────────────────────

class FailureInjector(BaseModel):
    """
    Injects, cascades, and auto-heals failures on a ServiceRegistry.

    Internal state is kept in private attrs so the injector itself is
    never accidentally serialised into an observation.
    """

    # seeded RNG for deterministic jitter
    _rng: random.Random = PrivateAttr(default_factory=random.Random)

    # (event, remaining_heal_steps)  — remaining is None when manual-only
    _active: Dict[str, List[Tuple[FailureEvent, Optional[int]]]] = PrivateAttr(
        default_factory=dict,
    )

    # ── Seed / reset ─────────────────────────────────────────────────────

    def seed(self, seed: Optional[int] = None) -> None:
        """Re-seed the RNG and clear all tracked failures."""
        if seed is not None:
            self._rng.seed(seed)
        self._active.clear()

    # ── Inject ───────────────────────────────────────────────────────────

    def inject(self, event: FailureEvent, registry: ServiceRegistry) -> None:
        """Apply *event* to the target service in *registry*."""
        svc = registry.get_service(event.service)
        if svc is None:
            return  # silently skip unknown services

        mult = _SEVERITY[event.severity]

        if event.failure_type == "cpu_spike":
            svc.cpu_pct = min(100.0, svc.cpu_pct + self._rng.uniform(30, 60) * mult)
            svc.emit_log("ERROR", f"CPU spike ({event.severity})")

        elif event.failure_type == "oom":
            svc.memory_pct = min(100.0, svc.memory_pct + self._rng.uniform(40, 70) * mult)
            svc.error_rate = 1.0
            svc.status = "down"
            svc.emit_log("CRITICAL", "Out of memory — service crashed.")

        elif event.failure_type == "crash":
            svc.error_rate = 1.0
            svc.status = "down"
            svc.emit_log("CRITICAL", "Service crashed unexpectedly.")

        elif event.failure_type == "latency":
            svc.latency_ms += self._rng.uniform(200, 800) * mult
            svc.error_rate = min(1.0, svc.error_rate + self._rng.uniform(0.1, 0.3) * mult)
            svc.emit_log("ERROR", f"Latency spike ({event.severity})")

        elif event.failure_type == "db_lock":
            svc.error_rate = min(1.0, svc.error_rate + self._rng.uniform(0.3, 0.6) * mult)
            svc.latency_ms += self._rng.uniform(300, 1000) * mult
            svc.emit_log("WARN", "Database lock contention detected.")

        svc.derive_status()

        # Track for cascading / auto-heal
        if event.cascade_to or event.auto_heal_after is not None:
            self._active.setdefault(event.service, []).append(
                (event, event.auto_heal_after),
            )

    # ── Cascade ──────────────────────────────────────────────────────────

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
                        f"Degraded — upstream {src_name} failure.",
                    )

    # ── Auto-heal tick ───────────────────────────────────────────────────

    def tick_heal(self, registry: ServiceRegistry) -> None:
        """Decrement heal timers; heal services whose timer reaches zero."""
        for svc_name in list(self._active):
            remaining: List[Tuple[FailureEvent, Optional[int]]] = []
            for event, steps_left in self._active[svc_name]:
                if steps_left is None:
                    # manual-only — keep tracking for cascade
                    remaining.append((event, None))
                    continue
                steps_left -= 1
                if steps_left <= 0:
                    svc = registry.get_service(svc_name)
                    if svc is not None:
                        svc.reset_to_healthy()
                        svc.emit_log("INFO", f"Auto-healed ({event.failure_type}).")
                else:
                    remaining.append((event, steps_left))
            if remaining:
                self._active[svc_name] = remaining
            else:
                del self._active[svc_name]
