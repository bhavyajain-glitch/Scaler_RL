"""
services.py — ServiceNode, LogEntry, and ServiceRegistry

Pure infrastructure data layer for OnCallEnv.
Knows nothing about tasks, rewards, or failures.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr


# ── Constants ────────────────────────────────────────────────────────────────

LOG_BUFFER_SIZE = 50  # max log entries kept per service

# Metric thresholds that drive automatic status transitions
CPU_DEGRADED = 80.0
CPU_DOWN = 98.0
MEM_DEGRADED = 85.0
ERR_DEGRADED = 0.20
ERR_DOWN = 0.80


# ── LogEntry ─────────────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    """A single structured log line emitted by a service."""

    timestamp: float = 0.0  # simulated clock (set by caller for determinism)
    level: Literal["INFO", "WARN", "ERROR", "CRITICAL"] = "INFO"
    message: str = ""

    model_config = {"frozen": True}


# ── ServiceNode ──────────────────────────────────────────────────────────────

class ServiceNode(BaseModel):
    """
    One simulated service (api, worker, db, …).

    Metrics are mutable — the environment mutates them directly.
    Call `derive_status()` after changing metrics to update `status`.
    """

    name: str
    status: Literal["healthy", "degraded", "down"] = "healthy"

    # Metrics
    cpu_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    memory_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=50.0, ge=0.0)

    # Operational counters
    restart_count: int = Field(default=0, ge=0)
    uptime_steps: int = Field(default=0, ge=0)

    # Ring-buffer of logs (private, excluded from serialisation)
    _logs: Deque[LogEntry] = PrivateAttr(
        default_factory=lambda: deque(maxlen=LOG_BUFFER_SIZE)
    )

    # ── Logging ──────────────────────────────────────────────────────────

    def emit_log(
        self,
        level: Literal["INFO", "WARN", "ERROR", "CRITICAL"],
        message: str,
        timestamp: float = 0.0,
    ) -> LogEntry:
        """Append a log entry to the ring buffer and return it."""
        entry = LogEntry(timestamp=timestamp, level=level, message=message)
        self._logs.append(entry)
        return entry

    def get_logs(self, last_n: Optional[int] = None) -> List[LogEntry]:
        """Return log entries, optionally limited to the most recent *last_n*."""
        logs = list(self._logs)
        if last_n is not None:
            return logs[-last_n:]
        return logs

    def clear_logs(self) -> None:
        """Wipe all stored log entries."""
        self._logs.clear()

    # ── Status ───────────────────────────────────────────────────────────

    def derive_status(self) -> None:
        """Re-derive `status` from current metric values."""
        if self.cpu_pct >= CPU_DOWN or self.error_rate >= ERR_DOWN:
            self.status = "down"
        elif (
            self.cpu_pct >= CPU_DEGRADED
            or self.memory_pct >= MEM_DEGRADED
            or self.error_rate >= ERR_DEGRADED
        ):
            self.status = "degraded"
        else:
            self.status = "healthy"

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self, last_n_logs: Optional[int] = None) -> dict:
        """Return a plain-dict snapshot suitable for agent observations."""
        return {
            "name": self.name,
            "status": self.status,
            "cpu_pct": round(self.cpu_pct, 2),
            "memory_pct": round(self.memory_pct, 2),
            "error_rate": round(self.error_rate, 4),
            "latency_ms": round(self.latency_ms, 2),
            "restart_count": self.restart_count,
            "uptime_steps": self.uptime_steps,
            "logs": [e.model_dump() for e in self.get_logs(last_n_logs)],
        }

    # ── Reset ────────────────────────────────────────────────────────────

    def reset_to_healthy(self) -> None:
        """Restore every metric to its baseline and clear logs.

        restart_count is intentionally preserved so the agent observation
        continues to reflect how many restarts occurred this episode.
        """
        self.cpu_pct = 10.0
        self.memory_pct = 20.0
        self.error_rate = 0.0
        self.latency_ms = 50.0
        self.uptime_steps = 0
        self.clear_logs()
        self.derive_status()
        self.emit_log("INFO", f"[{self.name}] Service initialised and healthy.")


# ── ServiceRegistry ──────────────────────────────────────────────────────────

class ServiceRegistry(BaseModel):
    """
    Owns every ServiceNode in the simulation.

    Provides helpers to add, query, tick, snapshot, and reset all services.
    """

    services: Dict[str, ServiceNode] = Field(default_factory=dict)

    # Internal RNG for metric decay jitter (private, excluded)
    _rng: random.Random = PrivateAttr(default_factory=random.Random)

    # ── Mutators ─────────────────────────────────────────────────────────

    def add_service(self, name: str) -> ServiceNode:
        """Create a new healthy service and register it. Returns the node."""
        node = ServiceNode(name=name)
        node.emit_log("INFO", f"[{name}] Service registered.")
        self.services[name] = node
        return node

    def get_service(self, name: str) -> Optional[ServiceNode]:
        """Look up a service by name. Returns None if not found."""
        return self.services.get(name)

    # ── Tick ──────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """
        Advance one simulated time-step for every service:
          1. Decay metrics towards healthy baselines (small random jitter).
          2. Track uptime for healthy services.
          3. Re-derive status.
        """
        for svc in self.services.values():
            # Gentle decay towards baseline
            svc.cpu_pct = max(10.0, svc.cpu_pct - self._rng.uniform(1.0, 5.0))
            svc.memory_pct = max(20.0, svc.memory_pct - self._rng.uniform(0.5, 2.0))
            svc.error_rate = max(0.0, svc.error_rate - self._rng.uniform(0.01, 0.05))
            svc.latency_ms = max(50.0, svc.latency_ms - self._rng.uniform(1.0, 5.0))

            # Uptime bookkeeping
            if svc.status == "healthy":
                svc.uptime_steps += 1
            else:
                svc.uptime_steps = 0

            svc.derive_status()

    # ── Snapshot ──────────────────────────────────────────────────────────

    def get_snapshot(self, last_n_logs: int = 5) -> Dict[str, dict]:
        """Serialisable snapshot of every service (for agent observations)."""
        return {
            name: svc.snapshot(last_n_logs)
            for name, svc in self.services.items()
        }

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        service_names: List[str],
        seed: Optional[int] = None,
    ) -> None:
        """
        Tear down existing services and create fresh healthy ones.

        If *seed* is provided the internal RNG is re-seeded for
        deterministic metric decay.
        """
        if seed is not None:
            self._rng.seed(seed)

        self.services.clear()
        for name in service_names:
            self.add_service(name)
