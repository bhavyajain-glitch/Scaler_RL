"""
environment.py — OnCallEnv, the core RL loop

Ties together ServiceRegistry, FailureInjector, and RewardCalculator
behind a simple  reset() / step() / state()  interface.

New in v2:
  - Actions: run_diagnostic, create_incident_ticket
  - Observation fields: incident_context, last_diagnostic
  - TaskSpec fields: root_cause_service, expected_severity, is_deploy_failure,
                     extra_logs (for red-herring log injection)
  - Episode tracking: diagnosed_services, ticket_filed
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field

from env.services import ServiceRegistry
from env.failures import FailureEvent, FailureInjector
from env.reward import RewardCalculator, RewardSignal


# ── Extra-log model (used for red-herring injection in TaskSpec) ──────────────

class ExtraLog(BaseModel):
    """A single log entry to inject into a service after failure injection.

    Used to overlay benign context (e.g. cron-job explanation) on top of
    metric-driven logs so that the LLM can distinguish real failures from
    expected elevated load.
    """
    service: str
    level: Literal["INFO", "WARN", "ERROR", "CRITICAL"]
    message: str

    model_config = {"frozen": True}


# ── I/O models ────────────────────────────────────────────────────────────────

class ObservationModel(BaseModel):
    """Immutable snapshot the agent receives each step."""

    step_count: int
    services: Dict[str, dict]
    active_alerts: List[str] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)
    # Rolling narrative of what has happened this episode (last 10 events)
    incident_context: List[str] = Field(default_factory=list)
    # Full text of the most recent run_diagnostic output (None if not yet run)
    last_diagnostic: Optional[str] = None

    model_config = {"frozen": True}


class ActionModel(BaseModel):
    """A single agent action."""

    action_type: Literal[
        "restart_service",
        "check_logs",
        "rollback",
        "scale_up",
        "scale_down",
        "run_diagnostic",
        "create_incident_ticket",
    ]
    target_service: str
    # Only used for create_incident_ticket
    ticket_severity: Optional[Literal["sev1", "sev2", "sev3"]] = None


class TaskSpec(BaseModel):
    """Defines one scenario the agent must resolve."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard", "expert"]
    failure_sequence: List[FailureEvent]
    success_condition: Callable[[ServiceRegistry], bool]
    max_steps: int

    # Optional metadata used by the reward calculator
    root_cause_service: Optional[str] = None   # service the ticket should name
    expected_severity: Optional[Literal["sev1", "sev2", "sev3"]] = None
    is_deploy_failure: bool = False            # True → rollback earns +7

    # Extra logs injected after failures (e.g. benign cron-job context)
    extra_logs: List[ExtraLog] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ── OnCallEnv ─────────────────────────────────────────────────────────────────

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

        self.registry     = ServiceRegistry()
        self.injector     = FailureInjector()
        self.reward_calc  = RewardCalculator()

        # Episode state
        self.step_count:       int        = 0
        self.recent_actions:   List[str]  = []
        self.active_alerts:    List[str]  = []
        self.incident_context: List[str]  = []
        self._last_diagnostic: Optional[str] = None
        self._diagnosed_services: Set[str]   = set()
        self._ticket_filed: Optional[Tuple[str, str]] = None
        self._rng = random.Random()

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        """Start a new episode.  Deterministic when *seed* is given."""
        if seed is not None:
            self._rng.seed(seed)
        self.registry.reset(self.service_names, seed=seed)
        self.injector.seed(seed)

        # Inject scenario failures
        for event in self.task.failure_sequence:
            self.injector.inject(event, self.registry)

        # Overlay any extra logs (e.g. benign cron-job context for red herrings)
        for el in self.task.extra_logs:
            svc = self.registry.get_service(el.service)
            if svc is not None:
                svc.emit_log(el.level, el.message)  # type: ignore[arg-type]

        # Clear episode bookkeeping
        self.step_count = 0
        self.recent_actions.clear()
        self.active_alerts.clear()
        self.incident_context.clear()
        self._last_diagnostic = None
        self._diagnosed_services.clear()
        self._ticket_filed = None

        return self.state()

    # ── step ───────────────────────────────────────────────────────────────

    def step(
        self, action: ActionModel,
    ) -> Tuple[ObservationModel, RewardSignal, bool, dict]:
        """Execute one action and advance the simulation by one tick."""

        pre = self.state()

        # Execute action; capture any error message
        last_action_error = self._execute(action)

        # Tick world (auto-heal → cascade → metric decay)
        self.injector.tick_heal(self.registry)
        self.injector.cascade(self.registry)
        self.registry.tick()
        self.step_count += 1

        # Track recent actions (last 5)
        tag = f"{action.action_type}:{action.target_service}"
        self.recent_actions.append(tag)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)

        # Update incident context BEFORE capturing post state so it appears
        # in the observation returned to the agent this step.
        post_services = self.registry.get_snapshot(last_n_logs=5)
        narrative = self._describe_action(action, pre.services, post_services)
        self.incident_context.append(narrative)
        if len(self.incident_context) > 10:
            self.incident_context.pop(0)

        post = self.state()

        # Done check
        success = self.task.success_condition(self.registry)
        timeout = self.step_count >= self.task.max_steps
        done    = success or timeout

        # Compute reward
        extra = {
            "diagnosed_services": set(self._diagnosed_services),
            "ticket":             self._ticket_filed,
        }
        reward = self.reward_calc.compute(
            action, pre, post, self.task, resolved=success, extra=extra,
        )

        info = {
            "success":           success,
            "timeout":           timeout,
            "last_action_error": last_action_error,
        }
        return post, reward, done, info

    # ── state (read-only) ──────────────────────────────────────────────────

    def state(self) -> ObservationModel:
        """Current observation — no side effects."""
        return ObservationModel(
            step_count=self.step_count,
            services=self.registry.get_snapshot(last_n_logs=5),
            active_alerts=list(self.active_alerts),
            recent_actions=list(self.recent_actions),
            incident_context=list(self.incident_context[-10:]),
            last_diagnostic=self._last_diagnostic,
        )

    # ── action execution (private) ─────────────────────────────────────────

    def _execute(self, action: ActionModel) -> Optional[str]:
        """Mutate the registry according to the agent's chosen action.

        Returns an error string if the action could not be applied, else None.
        """
        svc = self.registry.get_service(action.target_service)
        if svc is None:
            return f"unknown service: {action.target_service}"

        if action.action_type == "restart_service":
            svc.restart_count += 1
            svc.reset_to_healthy()
            svc.emit_log("INFO", "Service restarted by on-call engineer.")
            self.injector.clear_service(action.target_service)

        elif action.action_type == "rollback":
            svc.reset_to_healthy()
            svc.emit_log("INFO", "Deployment rolled back by on-call engineer.")
            self.injector.clear_service(action.target_service)

        elif action.action_type == "scale_up":
            svc.cpu_pct = max(10.0, svc.cpu_pct - 30.0)
            svc.derive_status()
            svc.emit_log("INFO", "Horizontal scale-up applied. CPU load reduced.")

        elif action.action_type == "scale_down":
            svc.cpu_pct = min(100.0, svc.cpu_pct + 10.0)
            svc.derive_status()
            svc.emit_log("INFO", "Scale-down applied.")

        elif action.action_type == "check_logs":
            self.active_alerts.append(f"Checked logs for {action.target_service}")
            self._diagnosed_services.add(action.target_service)

        elif action.action_type == "run_diagnostic":
            from env.log_templates import build_diagnostic_report
            report = build_diagnostic_report(
                action.target_service,
                svc.snapshot(last_n_logs=10),
                self.step_count,
            )
            self._last_diagnostic = report
            self._diagnosed_services.add(action.target_service)
            self.active_alerts.append(
                f"Diagnostic run on {action.target_service} — see last_diagnostic."
            )

        elif action.action_type == "create_incident_ticket":
            severity = action.ticket_severity or "sev3"
            self._ticket_filed = (action.target_service, severity)
            self.active_alerts.append(
                f"Incident ticket filed: service={action.target_service} severity={severity}"
            )

        return None

    # ── helpers ────────────────────────────────────────────────────────────

    def _describe_action(
        self,
        action: ActionModel,
        pre_svcs: Dict[str, dict],
        post_svcs: Dict[str, dict],
    ) -> str:
        """Return a one-line narrative entry for incident_context."""
        t    = action.target_service
        pre  = pre_svcs.get(t, {}).get("status", "?")
        post = post_svcs.get(t, {}).get("status", "?")
        at   = action.action_type

        if at in ("restart_service", "rollback"):
            return f"Step {self.step_count}: {at}({t}) — {pre} → {post}"
        if at == "run_diagnostic":
            return f"Step {self.step_count}: run_diagnostic({t}) — report available in last_diagnostic"
        if at == "create_incident_ticket":
            sev = action.ticket_severity or "unspecified"
            return f"Step {self.step_count}: create_incident_ticket({t}, {sev})"
        if at == "check_logs":
            return f"Step {self.step_count}: check_logs({t}) — {pre} status"
        if at in ("scale_up", "scale_down"):
            pre_cpu  = pre_svcs.get(t, {}).get("cpu_pct", 0)
            post_cpu = post_svcs.get(t, {}).get("cpu_pct", 0)
            return f"Step {self.step_count}: {at}({t}) — cpu {pre_cpu:.0f}% → {post_cpu:.0f}%"
        return f"Step {self.step_count}: {at}({t})"
