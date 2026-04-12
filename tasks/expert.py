"""
tasks/expert.py — Red-herring incident (Expert difficulty)

Scenario
--------
At 03:00 UTC three things are happening simultaneously:

1. [RED HERRING] Worker — nightly-report cron job is running.
   CPU is elevated (82–90%) but the logs clearly say this is scheduled
   and expected.  The worker will AUTO-HEAL in 5 steps when the job ends.
   The correct strategy is to IGNORE the worker.

2. [REAL FAILURE] DB — out-of-memory (OOM) crash.
   Memory at ~95 %, error_rate = 1.0, status = DOWN.
   Must be fixed with restart_service(db).

3. [CASCADED] API — degraded by the DB crash.
   Upstream errors from db are causing API latency and elevated error rate.
   Fix DB first (cascade stops), then restart_service(api).

Optimal strategy (grade = 1.0):
  run_diagnostic(db)          ← read the report, confirm OOM root cause
  create_incident_ticket(db, sev1)   ← bonus points
  restart_service(db)         ← fix root cause; cascade clears
  restart_service(api)        ← fix cascaded service
                              ← worker auto-heals on its own

A naive agent that restarts everything (including the worker) scores lower
due to wasted steps.  A rule-based script has no way to distinguish the
cron logs from a real failure — an LLM that reads them does.
"""

from __future__ import annotations

from env.environment import ExtraLog, TaskSpec
from env.failures import FailureEvent
from env.services import ServiceRegistry


# ── Success condition ──────────────────────────────────────────────────────────

def _all_healthy(registry: ServiceRegistry) -> bool:
    return all(s.status == "healthy" for s in registry.services.values())


# ── Grader (0.0 – 1.0) ────────────────────────────────────────────────────────

def grade(steps_taken: int, max_steps: int, success: bool) -> float:
    """
    Grading rubric:
      - 0.0  if the task was not resolved.
      - 1.0  if resolved in ≤ 4 steps (optimal: diagnostic + ticket + fix db + fix api).
      - Linear decay towards 0.2 as steps approach max_steps.
    An LLM that reads the cron logs and skips unnecessary worker restarts
    converges much faster than one reacting to raw metrics alone.
    """
    if not success:
        return 0.0
    optimal = 4
    if steps_taken <= optimal:
        return 1.0
    remaining_ratio = (steps_taken - optimal) / (max_steps - optimal)
    score = 1.0 - 0.80 * remaining_ratio
    return round(max(0.2, score), 2)


# ── Task spec ──────────────────────────────────────────────────────────────────

TASK = TaskSpec(
    task_id="expert_red_herring",
    difficulty="expert",
    failure_sequence=[
        # Real failure: DB OOM → cascades degradation to API
        FailureEvent(
            service="db",
            failure_type="oom",
            severity="high",
            cascade_to=["api"],
        ),
        # Red herring: Worker nightly cron — auto-heals in 5 steps
        FailureEvent(
            service="worker",
            failure_type="cpu_spike",
            severity="medium",
            auto_heal_after=5,
        ),
    ],
    success_condition=_all_healthy,
    max_steps=20,
    root_cause_service="db",
    expected_severity="sev1",
    # These benign logs overlay the cpu_spike logs on the worker so an LLM
    # reading them understands the high CPU is from a scheduled job
    extra_logs=[
        ExtraLog(
            service="worker",
            level="INFO",
            message=(
                "nightly-report cron started (scheduled 03:00 UTC). "
                "Expected duration: 8–12 min."
            ),
        ),
        ExtraLog(
            service="worker",
            level="WARN",
            message=(
                "Batch job CPU elevated (84%). "
                "THIS IS EXPECTED for nightly aggregation — NO ACTION NEEDED. "
                "Worker will return to baseline when cron finishes."
            ),
        ),
        ExtraLog(
            service="worker",
            level="INFO",
            message=(
                "nightly-report progress: 7,423 / 14,847 records processed (50%). "
                "Running normally. Do not restart this service."
            ),
        ),
    ],
)
