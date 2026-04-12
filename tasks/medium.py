"""
tasks/medium.py — Multi-service incident (Medium difficulty)

Scenario: The API service is experiencing a latency spike, AND the worker
service has run out of memory (OOM).  The agent must diagnose and fix both.
"""

from __future__ import annotations

from env.environment import TaskSpec
from env.failures import FailureEvent
from env.services import ServiceRegistry


# ── Success condition ────────────────────────────────────────────────────────

def _all_healthy(registry: ServiceRegistry) -> bool:
    return all(s.status == "healthy" for s in registry.services.values())


# ── Grader (0.0 – 1.0) ──────────────────────────────────────────────────────

def grade(steps_taken: int, max_steps: int, success: bool) -> float:
    """
    Grading rubric:
      - 0.0  if the task was not resolved
      - 1.0  if resolved in ≤ 2 steps (optimal for 2-service fix)
      - Linear decay towards 0.2 as steps approach max_steps
    """
    if not success:
        return 0.0
    optimal = 2
    if steps_taken <= optimal:
        return 1.0
    remaining_ratio = (steps_taken - optimal) / (max_steps - optimal)
    score = 1.0 - 0.8 * remaining_ratio
    return round(max(0.1, score), 2)


# ── Task spec ────────────────────────────────────────────────────────────────

TASK = TaskSpec(
    task_id="medium_api_latency_worker_oom",
    difficulty="medium",
    failure_sequence=[
        FailureEvent(
            service="api",
            failure_type="latency",
            severity="medium",
        ),
        FailureEvent(
            service="worker",
            failure_type="oom",
            severity="high",
        ),
    ],
    success_condition=_all_healthy,
    max_steps=15,
    root_cause_service="worker",
    expected_severity="sev1",
)
