"""
reward.py — RewardSignal and RewardCalculator

Stateless reward computation for OnCallEnv.

Reward table (per step):
  time_penalty         : -0.5  (every step — discourages procrastination)

  restart_service
    service was down → healthy : +5.0
    service was already healthy : -2.0  (anti-thrash)

  rollback
    task.is_deploy_failure=True : +7.0
    otherwise                   : -3.0  (wrong tool)

  scale_up / scale_down
    CPU > 85 % before action    : +3.0  (appropriate use)
    CPU < 50 % before action    : -1.0  (wasted action)

  run_diagnostic
    target == root_cause_service: +1.0
    any other target            : +0.3  (still somewhat useful)

  create_incident_ticket
    correct service AND severity: +4.0
    correct service only        : +1.5
    wrong service               : -1.0

  diagnosis_first bonus
    agent ran run_diagnostic or check_logs on this service
    BEFORE a fix action (restart/rollback/scale):  +1.5

  root_cause_first bonus
    agent fixes the known root-cause service while
    at least one downstream is still degraded:      +2.0

  task_resolved                : +10.0 (once, when success=True)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from pydantic import BaseModel, Field


# ── RewardSignal ──────────────────────────────────────────────────────────────

class RewardSignal(BaseModel):
    """Structured reward returned after every step."""

    score: float
    reason: str
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ── RewardCalculator ──────────────────────────────────────────────────────────

class RewardCalculator(BaseModel):
    """
    Pure-function reward calculator.

    `compute()` receives snapshot dicts (not live objects) so there is
    no dependency on environment.py or services.py at import time.
    """

    def compute(
        self,
        action: Any,           # ActionModel
        pre_state: Any,        # ObservationModel
        post_state: Any,       # ObservationModel
        task: Any,             # TaskSpec
        resolved: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Return the reward for a single step."""
        score = 0.0
        breakdown: Dict[str, float] = {}
        extra = extra or {}

        target   = action.target_service
        pre_svc  = pre_state.services.get(target, {})
        post_svc = post_state.services.get(target, {})

        diagnosed: Set[str]                     = extra.get("diagnosed_services", set())
        ticket:    Optional[Tuple[str, str]]    = extra.get("ticket")

        # ── 1. Time penalty (flat -0.5 per step) ──────────────────────────
        score += -0.5
        breakdown["time_penalty"] = -0.5

        # ── 2. Action-specific rewards ─────────────────────────────────────

        if action.action_type == "restart_service":
            if pre_svc.get("status") == "down" and post_svc.get("status") == "healthy":
                score += 5.0
                breakdown["restart_success"] = 5.0
            elif pre_svc.get("status") == "healthy":
                score -= 2.0
                breakdown["restart_wasted"] = -2.0

        elif action.action_type == "rollback":
            if getattr(task, "is_deploy_failure", False):
                score += 7.0
                breakdown["rollback_success"] = 7.0
            else:
                score -= 3.0
                breakdown["rollback_wrong"] = -3.0

        elif action.action_type in ("scale_up", "scale_down"):
            cpu = pre_svc.get("cpu_pct", 0)
            if cpu > 85:
                score += 3.0
                breakdown["scale_needed"] = 3.0
            elif cpu < 50:
                score -= 1.0
                breakdown["scale_wasted"] = -1.0

        elif action.action_type == "run_diagnostic":
            root = getattr(task, "root_cause_service", None)
            if root is None or target == root:
                score += 1.0
                breakdown["diagnostic_relevant"] = 1.0
            else:
                score += 0.3
                breakdown["diagnostic_offside"] = 0.3

        elif action.action_type == "create_incident_ticket":
            expected_svc = getattr(task, "root_cause_service", None)
            expected_sev = getattr(task, "expected_severity", None)
            ticket_svc   = target
            ticket_sev   = action.ticket_severity

            correct_svc = (expected_svc is None) or (ticket_svc == expected_svc)
            correct_sev = (expected_sev is None) or (ticket_sev == expected_sev)

            if correct_svc and correct_sev:
                score += 4.0
                breakdown["ticket_accurate"] = 4.0
            elif correct_svc:
                score += 1.5
                breakdown["ticket_service_ok"] = 1.5
            else:
                score -= 1.0
                breakdown["ticket_wrong"] = -1.0

        elif action.action_type == "check_logs":
            breakdown["check_logs"] = 0.0

        # ── 3. Diagnosis-first bonus ───────────────────────────────────────
        # Reward the pattern: run_diagnostic / check_logs → then fix
        if action.action_type in ("restart_service", "rollback", "scale_up", "scale_down"):
            if target in diagnosed:
                score += 1.5
                breakdown["diagnosis_first"] = 1.5

        # ── 4. Root-cause-first bonus ──────────────────────────────────────
        # Reward fixing the known root cause while downstream is still degraded
        if action.action_type in ("restart_service", "rollback"):
            root = getattr(task, "root_cause_service", None)
            if root and target == root and pre_svc.get("status") in ("degraded", "down"):
                downstream_struggling = any(
                    s.get("status") in ("degraded", "down")
                    for name, s in post_state.services.items()
                    if name != root
                )
                if downstream_struggling:
                    score += 2.0
                    breakdown["root_cause_first"] = 2.0

        # ── 5. Resolution bonus ────────────────────────────────────────────
        if resolved:
            score += 10.0
            breakdown["task_resolved"] = 10.0

        reason = ", ".join(f"{k}={v:+.1f}" for k, v in breakdown.items())
        return RewardSignal(score=score, reason=reason, breakdown=breakdown)
