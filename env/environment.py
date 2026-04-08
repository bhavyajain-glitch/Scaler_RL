"""
environment.py — OnCallEnv, the core RL loop

Ties together ServiceRegistry, FailureInjector, and RewardCalculator
behind a simple  reset() / step() / state()  interface.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from env.services import ServiceRegistry
from env.failures import FailureEvent, FailureInjector
from env.reward import RewardCalculator, RewardSignal


# ── I/O models ───────────────────────────────────────────────────────────────

class ObservationModel(BaseModel):
    """Immutable snapshot the agent receives each step."""

    step_count: int
    services: Dict[str, dict]
    active_alerts: List[str] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class ActionModel(BaseModel):
    """A single agent action."""

    action_type: Literal[
        "restart_service", "check_logs", "rollback", "scale_up", "scale_down",
    ]
    target_service: str


class TaskSpec(BaseModel):
    """Defines one scenario the agent must resolve."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    failure_sequence: List[FailureEvent]
    success_condition: Callable[[ServiceRegistry], bool]
    max_steps: int


# ── OnCallEnv ────────────────────────────────────────────────────────────────

DEFAULT_SERVICES = ["api", "worker", "db"]


class OnCallEnv:
    """
    Gym-style RL environment for on-call incident response.

        obs              = env.reset(seed=42)
        obs, rw, done, i = env.step(ActionModel(...))
        obs              = env.state()
    """

    def __init__(self, task: TaskSpec, service_names: Optional[List[str]] = None):
        self.task = task
        self.service_names = service_names or DEFAULT_SERVICES

        # Sub-systems
        self.registry = ServiceRegistry()
        self.injector = FailureInjector()
        self.reward_calc = RewardCalculator()

        # Episode state
        self.step_count: int = 0
        self.recent_actions: List[str] = []
        self.active_alerts: List[str] = []
        self._rng = random.Random()

    # ── reset ────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        """Start a new episode. Deterministic when *seed* is given."""
        # 1. Seed RNGs
        if seed is not None:
            self._rng.seed(seed)
        self.registry.reset(self.service_names, seed=seed)
        self.injector.seed(seed)

        # 2. Inject initial failures
        for event in self.task.failure_sequence:
            self.injector.inject(event, self.registry)

        # 3. Clear episode bookkeeping
        self.step_count = 0
        self.recent_actions.clear()
        self.active_alerts.clear()

        return self.state()

    # ── step ─────────────────────────────────────────────────────────────

    def step(
        self, action: ActionModel,
    ) -> Tuple[ObservationModel, RewardSignal, bool, dict]:
        """Execute one action and advance the simulation by one tick."""

        # 1. Pre-state snapshot
        pre = self.state()

        # 2. Execute agent action
        self._execute(action)

        # 3. Tick world (auto-heal → cascade → metric decay)
        self.injector.tick_heal(self.registry)
        self.injector.cascade(self.registry)
        self.registry.tick()
        self.step_count += 1

        # 4. Track action history (keep last 5)
        tag = f"{action.action_type}:{action.target_service}"
        self.recent_actions.append(tag)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)

        # 5. Post-state snapshot
        post = self.state()

        # 6. Done check
        success = self.task.success_condition(self.registry)
        timeout = self.step_count >= self.task.max_steps
        done = success or timeout

        # 7. Reward
        reward = self.reward_calc.compute(
            action, pre, post, self.task, resolved=success,
        )

        info = {"success": success, "timeout": timeout}
        return post, reward, done, info

    # ── state (read-only) ────────────────────────────────────────────────

    def state(self) -> ObservationModel:
        """Current observation — no side effects."""
        return ObservationModel(
            step_count=self.step_count,
            services=self.registry.get_snapshot(last_n_logs=5),
            active_alerts=list(self.active_alerts),
            recent_actions=list(self.recent_actions),
        )

    # ── action execution (private) ───────────────────────────────────────

    def _execute(self, action: ActionModel) -> None:
        """Mutate the registry according to the agent's chosen action."""
        svc = self.registry.get_service(action.target_service)
        if svc is None:
            return

        if action.action_type == "restart_service":
            svc.restart_count += 1
            svc.reset_to_healthy()
            svc.emit_log("INFO", "Service restarted by agent.")

        elif action.action_type == "rollback":
            svc.reset_to_healthy()
            svc.emit_log("INFO", "Deployment rolled back by agent.")

        elif action.action_type == "scale_up":
            svc.cpu_pct = max(10.0, svc.cpu_pct - 30.0)
            svc.derive_status()
            svc.emit_log("INFO", "Scaled up.")

        elif action.action_type == "scale_down":
            svc.cpu_pct = min(100.0, svc.cpu_pct + 10.0)
            svc.derive_status()
            svc.emit_log("INFO", "Scaled down.")

        elif action.action_type == "check_logs":
            self.active_alerts.append(
                f"Checked logs for {action.target_service}",
            )
