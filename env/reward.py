"""
reward.py — RewardSignal and RewardCalculator

Stateless reward computation for OnCallEnv.
All inputs are plain dicts / Pydantic models — no circular imports.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


# ── RewardSignal ─────────────────────────────────────────────────────────────

class RewardSignal(BaseModel):
    """Structured reward returned after every step."""

    score: float
    reason: str
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ── RewardCalculator ─────────────────────────────────────────────────────────

class RewardCalculator(BaseModel):
    """
    Pure-function reward calculator.

    `compute()` receives snapshot dicts (not live objects) so there is
    no dependency on environment.py or services.py at import time.
    """

    # ── public API ───────────────────────────────────────────────────────

    def compute(
        self,
        action: Any,          # ActionModel
        pre_state: Any,       # ObservationModel  (snapshot dicts inside)
        post_state: Any,      # ObservationModel
        task: Any,            # TaskSpec
        resolved: bool = False,
    ) -> RewardSignal:
        """Return the reward for a single step."""
        score = 0.0
        breakdown: Dict[str, float] = {}

        # ── time penalty ─────────────────────────────────────────────────
        tp = self._time_penalty(post_state.step_count)
        score += tp
        breakdown["time_penalty"] = tp

        # ── action-specific rewards ──────────────────────────────────────
        target = action.target_service
        pre_svc = pre_state.services.get(target, {})
        post_svc = post_state.services.get(target, {})

        if action.action_type == "restart_service":
            if pre_svc.get("status") == "down" and post_svc.get("status") == "healthy":
                score += 5.0
                breakdown["restart_success"] = 5.0
            elif pre_svc.get("status") == "healthy":
                score -= 2.0
                breakdown["restart_wasted"] = -2.0

        elif action.action_type == "rollback":
            if getattr(task, "root_cause", None) == "deploy":
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

        elif action.action_type == "check_logs":
            breakdown["check_logs"] = 0.0

        # ── resolution bonus ─────────────────────────────────────────────
        if resolved:
            score += 10.0
            breakdown["task_resolved"] = 10.0

        reason = ", ".join(f"{k}={v:+.1f}" for k, v in breakdown.items())
        return RewardSignal(score=score, reason=reason, breakdown=breakdown)

    # ── internals ────────────────────────────────────────────────────────

    @staticmethod
    def _time_penalty(steps: int) -> float:
        """−0.5 per step elapsed."""
        return -0.5 * steps
