"""
services.py — ServiceNode, LogEntry, and ServiceRegistry

Defines the infrastructure data layer for OnCallEnv.
No knowledge of tasks, rewards, or failures lives here.
"""

from __future__ import annotations

import time
import random
from collections import deque
from typing import Deque, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_BUFFER_SIZE = 50  # max log entries kept per service

# Thresholds that drive automatic status transitions on each tick
CPU_DEGRADED_THRESHOLD = 80.0
CPU_DOWN_THRESHOLD = 98.0
MEMORY_DEGRADED_THRESHOLD = 85.0
ERROR_RATE_DEGRADED_THRESHOLD = 0.20
ERROR_RATE_DOWN_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    """A single structured log line emitted by a service."""

    timestamp: float = Field(default_factory=time.time)
    level: Literal["INFO", "WARN", "ERROR", "CRITICAL"]
    message: str

    class Config:
        frozen = True  # log entries are immutable once written


# ---------------------------------------------------------------------------
# ServiceNode
# ---------------------------------------------------------------------------

class ServiceNode(BaseModel):
    """
    Represents one simulated service (api, worker, db, …).

    All metric fields are mutable; the environment mutates them directly.
    Status is derived automatically on each tick — do not set it by hand.
    """

    name: str
    status: Literal["healthy", "degraded", "down"] = "healthy"

    # Metrics (0–100 for pct, 0.0–1.0 for ratios)
    cpu_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    memory_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=50.0, ge=0.0)

    # Operational counters
    restart_count: int = Field(default=0, ge=0)
    uptime_steps: int = Field(default=0, ge=0)  # incremented each tick while healthy

    # Internal ring-buffer of log entries (not serialized to snapshot directly)
    _logs: Deque[LogEntry] = Field(default=None, exclude=True)

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        # Pydantic v2: initialise private-ish attribute after model construction
        object.__setattr__(self, "_logs", deque(maxlen=LOG_BUFFER_SIZE))

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def emit_log(self, level: Literal["INFO", "WARN", "ERROR", "CRITICAL"], message: str) -> LogEntry:
        entry = LogEntry(level=level, message=message)
        self._logs.append(entry)
        return entry

    def get_logs(self, last_n: Optional[int] = None) -> List[LogEntry]:
        """Return recent log entries. Pass last_n to limit results."""
        logs = list(self._logs)
        if last_n is not None:
            return logs[-last_n:]
        return logs

    def clear_logs(self) -> None:
        self._logs.clear()

    # ------------------------------------------------------------------
    # Status derivation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _derive_status(self) -> Self:
        """Derives service status based on metrics thresholds."""
        if self.cpu_pct >= CPU_DOWN_THRESHOLD or self.error_rate >= ERROR_RATE_DOWN_THRESHOLD:
            self.status = "down"
        elif (
            self.cpu_pct >= CPU_DEGRADED_THRESHOLD
            or self.memory_pct >= MEMORY_DEGRADED_THRESHOLD
            or self.error_rate >= ERROR_RATE_DEGRADED_THRESHOLD
        ):
            self.status = "degraded"
        else:
            self.status = "healthy"
        return self


    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict:
        """Serializable dict for the agent observation. Logs are included as raw dicts."""
        return {
            "name": self.name,
            "status": self.status,
            "cpu_pct": round(self.cpu_pct, 2),
            "memory_pct": round(self.memory_pct, 2),
            "error_rate": round(self.error_rate, 4),
            "latency_ms": round(self.latency_ms, 2),
            "restart_count": self.restart_count,
            "uptime_steps": self.uptime_steps,
            "logs": [entry.model_dump() for entry in self.get_logs()],
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_to_healthy(self) -> None:
        """Restore this service to a clean baseline state."""
        self.status = "healthy"
        self.cpu_pct = 10.0
        self.memory_pct = 20.0
        self.error_rate = 0.0
        self.latency_ms = 50.0
        self.restart_count = 0
        self.uptime_steps = 0
        self.clear_logs()
        self.emit_log("INFO", f"[{self.name}] Service initialised and healthy.")


# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------


class ServiceRegistry(BaseModel):
    """
    Manages all services in the environment and provides methods for global state management.
    """

    services: Dict[str, ServiceNode] = Field(default_factory=dict)
    _rng: random.Random = Field(default=None, exclude=True)

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        # Pydantic v2: initialise private-ish attribute after model construction
        object.__setattr__(self, "_rng", random.Random())

    def initialize(self, service_names: List[str], seed: Optional[int] = None) -> None:
        """Initializes the registry with a given set of service names."""
        if seed is not None:
            self._rng.seed(seed)
        self.services = {name: ServiceNode(name=name) for name in service_names}

    def get_service(self, name: str) -> Optional[ServiceNode]:
        """Returns a service by name, or None if not found."""
        return self.services.get(name)

    def tick(self) -> None:
        """
        Advances simulated time for all services.
        - Decays metrics (cpu, memory, error_rate, latency)
        - Increments uptime for healthy services
        - Re-derives status for all services
        """
        for service in self.services.values():
            # Decay metrics towards healthy defaults
            service.cpu_pct = max(0.0, service.cpu_pct - self._rng.uniform(1.0, 5.0))
            service.memory_pct = max(0.0, service.memory_pct - self._rng.uniform(1.0, 3.0))
            service.error_rate = max(0.0, service.error_rate - self._rng.uniform(0.01, 0.05))
            service.latency_ms = max(50.0, service.latency_ms - self._rng.uniform(1.0, 5.0))

            if service.status == "healthy":
                service.uptime_steps += 1
            else:
                service.uptime_steps = 0  # Reset uptime if not healthy

            # Re-derive status based on new metrics
            service._derive_status()

    def get_snapshot(self) -> Dict[str, dict]:
        """Returns a serializable state of all services."""
        return {name: service.model_dump() for name, service in self.services.items()}

    def reset(self, service_names: List[str], seed: Optional[int] = None) -> None:
        """Resets the registry to an initial healthy state."""
        self.initialize(service_names, seed)
        for service in self.services.values():
            service.clear_logs()
            service.restart_count = 0
            service.uptime_steps = 0
            service.cpu_pct = 10.0
            service.memory_pct = 20.0
            service.error_rate = 0.0
            service.latency_ms = 50.0
            service._derive_status()  # Ensure status is healthy
