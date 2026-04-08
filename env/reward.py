"""
reward.py — RewardSignal and RewardCalculator

Defines the reward computation logic for OnCallEnv.
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

# Assuming these will be defined elsewhere or passed in
# from env.environment import ActionModel, ObservationModel, TaskSpec
# from env.services import ServiceRegistry, ServiceNode


# ---------------------------------------------------------------------------
# RewardSignal
# ---------------------------------------------------------------------------


class RewardSignal(BaseModel):
    """Structured reward information for agent interpretability."""

    score: float
    reason: str
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RewardCalculator
# ---------------------------------------------------------------------------


class RewardCalculator(BaseModel):
    """
    Computes rewards based on pre-state, post-state, action, and task.
    """

    # TODO: These types need to be properly imported or defined when environment.py exists
    # For now, using Any to avoid circular dependencies
    def compute(
        self,
        action: Any,  # ActionModel
        pre_state: Any,  # ObservationModel
        post_state: Any,  # ObservationModel
        task: Any,  # TaskSpec
    ) -> RewardSignal:
        """Compute the reward for a given step."""
        score = 0.0
        breakdown: Dict[str, float] = {}

        # Time penalty
        time_penalty = self._penalize_time(post_state.step_count)
        score += time_penalty
        breakdown["time_penalty"] = time_penalty

        # Action-specific rewards
        if action.action_type == "restart_service":
            target_service_name = action.target_service
            pre_service = pre_state.services.get(target_service_name)
            post_service = post_state.services.get(target_service_name)

            if pre_service and post_service:
                if pre_service.status == "down" and post_service.status == "healthy":
                    score += 5.0
                    breakdown["restart_service_success"] = 5.0
                elif pre_service.status == "healthy" and post_service.status == "healthy":
                    score -= 2.0
                    breakdown["restart_service_wasted"] = -2.0

        elif action.action_type == "rollback":
            # This is a simplified check; actual rollback logic would be more complex
            # and depend on the root cause being a 'deploy' related issue.
            # For now, let's assume the task might have a 'root_cause' attribute.
            if hasattr(task, "root_cause") and task.root_cause == "deploy":
                # And assume action.target_service is the correct one
                # This needs refinement once TaskSpec is fully defined.
                score += 7.0
                breakdown["rollback_success"] = 7.0
            else:
                score -= 3.0
                breakdown["rollback_wrong"] = -3.0

        elif action.action_type == "scale":
            target_service_name = action.target_service
            post_service = post_state.services.get(target_service_name)
            if post_service:
                if post_service.cpu_pct > 85:
                    score += 3.0
                    breakdown["scale_needed"] = 3.0
                elif post_service.cpu_pct < 50:
                    score -= 1.0
                    breakdown["scale_wasted"] = -1.0

        elif action.action_type == "check_logs":
            breakdown["check_logs_info"] = 0.0

        # Task resolved bonus
        if self._is_task_resolved(post_state, task):
            score += 10.0
            breakdown["task_resolved_bonus"] = 10.0

        return RewardSignal(score=score, reason="Reward computed for step", breakdown=breakdown)

    def _penalize_time(self, steps_elapsed: int) -> float:
        """Applies a penalty for each step elapsed."""
        return -0.5 * steps_elapsed

    def _is_task_resolved(self, post_state: Any, task: Any) -> bool:
        """Checks if all services are healthy according to task success condition."""
        # This will depend on the actual implementation of task.success_condition
        if hasattr(task, "success_condition") and callable(task.success_condition):
            return task.success_condition(post_state.services) # Assuming success_condition takes service dict
        return False
