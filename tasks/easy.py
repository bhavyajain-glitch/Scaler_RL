"""
tasks/easy.py — Single-service incident (Easy difficulty)

Scenario: The database service has crashed.
Objective: Restart the DB to restore all services to healthy.
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
      - 1.0  if resolved in 1 step (optimal)
      - Linear decay: 1.0 → 0.3 as steps approach max_steps
    """
    if not success:
        return 0.0
    efficiency = 1.0 - 0.7 * (steps_taken / max_steps)
    return round(max(0.1, efficiency), 2)


# ── Task spec ────────────────────────────────────────────────────────────────

TASK = TaskSpec(
    task_id="easy_db_crash",
    difficulty="easy",
    failure_sequence=[
        FailureEvent(
            service="db",
            failure_type="crash",
            severity="high",
        ),
    ],
    success_condition=_all_healthy,
    max_steps=10,
)
