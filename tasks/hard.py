"""
tasks/hard.py — Cascading failure incident (Hard difficulty)

Scenario: The database has lock contention which cascades degradation to
the worker, which in turn cascades to the API.  The agent must identify
the root cause (DB) and resolve it; fixing downstream services first won't
help because the cascade re-fires each tick.
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
      - 1.0  if resolved in ≤ 3 steps (must fix root cause + cascaded services)
      - Decays towards 0.1 as steps approach max_steps
    """
    if not success:
        return 0.0
    optimal = 3
    if steps_taken <= optimal:
        return 1.0
    remaining_ratio = (steps_taken - optimal) / (max_steps - optimal)
    score = 1.0 - 0.9 * remaining_ratio
    return round(max(0.1, score), 2)


# ── Task spec ────────────────────────────────────────────────────────────────

TASK = TaskSpec(
    task_id="hard_cascading_db_lock",
    difficulty="hard",
    failure_sequence=[
        FailureEvent(
            service="db",
            failure_type="db_lock",
            severity="high",
            cascade_to=["worker", "api"],
        ),
    ],
    success_condition=_all_healthy,
    max_steps=20,
    root_cause_service="db",
    expected_severity="sev1",
)
