"""
inference.py — Baseline LLM agent for OnCallEnv

Uses the OpenAI client to observe the environment, reason about the
incident, and take actions to resolve it.  Runs all four difficulty tasks
and prints results in the required [START]/[STEP]/[END] format.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from openai import OpenAI

from env.environment import ActionModel, OnCallEnv
from tasks.easy   import TASK as EASY_TASK,   grade as easy_grade
from tasks.medium import TASK as MEDIUM_TASK, grade as medium_grade
from tasks.hard   import TASK as HARD_TASK,   grade as hard_grade
from tasks.expert import TASK as EXPERT_TASK, grade as expert_grade

# ── Environment variables (per hackathon guidelines) ──────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert SRE (Site Reliability Engineer) responding to a live incident.
You are given the current state of a distributed system with three services:
  api, worker, db

Your goal: diagnose the root cause and restore ALL services to "healthy" status
as quickly as possible.

=== AVAILABLE ACTIONS ===
Respond with EXACTLY ONE JSON object per turn. No markdown, no explanation.

Diagnostic actions (gather information first):
  {"action_type": "check_logs",      "target_service": "<name>"}
  {"action_type": "run_diagnostic",  "target_service": "<name>"}

  run_diagnostic returns a detailed report in the next observation's
  last_diagnostic field. It tells you the likely root cause and recommendation.
  ALWAYS run_diagnostic on suspicious services before fixing them.

Fix actions (use after diagnosis):
  {"action_type": "restart_service", "target_service": "<name>"}
  {"action_type": "rollback",        "target_service": "<name>"}
  {"action_type": "scale_up",        "target_service": "<name>"}
  {"action_type": "scale_down",      "target_service": "<name>"}

Reporting action (earns bonus points):
  {"action_type": "create_incident_ticket", "target_service": "<root-cause-name>",
   "ticket_severity": "<sev1|sev2|sev3>"}
  sev1 = critical (service down / data loss risk)
  sev2 = major    (severe degradation)
  sev3 = minor    (low-impact issue)

Where <name> is one of: api, worker, db

=== DECISION RULES ===
1. READ THE LOGS first. A high-CPU service with cron/batch logs is a red herring
   — do NOT restart it. Wait for it to self-heal.
2. Fix root-cause services BEFORE downstream ones.
   (If api is degraded because db is down, fix db first.)
3. run_diagnostic before restarting any service you haven't checked yet.
4. File a create_incident_ticket on the true root-cause service.
5. Do NOT restart a healthy service — you will be penalised.
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

VALID_ACTIONS  = {
    "restart_service", "check_logs", "rollback",
    "scale_up", "scale_down", "run_diagnostic", "create_incident_ticket",
}
VALID_SERVICES = {"api", "worker", "db"}
VALID_SEVERITIES = {"sev1", "sev2", "sev3"}


def _obs_to_text(obs) -> str:
    """Convert an ObservationModel into a compact text summary for the LLM."""
    lines = [f"=== Step {obs.step_count} ==="]

    for name, svc in obs.services.items():
        lines.append(
            f"\n[{name.upper()}] status={svc['status']}  "
            f"cpu={svc['cpu_pct']:.1f}%  mem={svc['memory_pct']:.1f}%  "
            f"err={svc['error_rate']:.1%}  latency={svc['latency_ms']:.0f}ms  "
            f"restarts={svc['restart_count']}"
        )
        if svc.get("logs"):
            for log in svc["logs"][-4:]:
                lines.append(f"  [{log['level']}] {log['message']}")

    if obs.active_alerts:
        lines.append(f"\nAlerts: {'; '.join(obs.active_alerts[-3:])}")

    if obs.incident_context:
        lines.append("\nIncident history:")
        for entry in obs.incident_context[-5:]:
            lines.append(f"  {entry}")

    if obs.last_diagnostic:
        lines.append("\n--- Last Diagnostic ---")
        lines.append(obs.last_diagnostic)
        lines.append("--- End Diagnostic ---")

    return "\n".join(lines)


def _parse_action(text: str) -> ActionModel | None:
    """Try to extract a valid ActionModel from the LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    action_type      = data.get("action_type", "")
    target           = data.get("target_service", "")
    ticket_severity  = data.get("ticket_severity")

    if action_type not in VALID_ACTIONS or target not in VALID_SERVICES:
        return None
    if ticket_severity is not None and ticket_severity not in VALID_SEVERITIES:
        ticket_severity = None

    return ActionModel(
        action_type=action_type,
        target_service=target,
        ticket_severity=ticket_severity,
    )


# ── Run one task ───────────────────────────────────────────────────────────────

def run_task(task, grade_fn, env_name: str = "oncall_env") -> float:
    """Execute a single task, printing [START]/[STEP]/[END] lines."""

    env     = OnCallEnv(task=task)
    obs     = env.reset(seed=42)
    rewards: list[float] = []
    steps   = 0
    success = False
    last_error: str | None = None

    print(f"[START] task={task.task_id} env={env_name} model={MODEL_NAME}", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _obs_to_text(obs)},
    ]

    try:
        while True:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            raw    = response.choices[0].message.content or ""
            action = _parse_action(raw)
            last_error = None

            if action is None:
                action     = ActionModel(action_type="check_logs", target_service="api")
                last_error = f"parse_error: {raw[:80]}"

            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(round(reward.score, 2))

            action_str = f"{action.action_type}('{action.target_service}')"
            if action.ticket_severity:
                action_str = (
                    f"{action.action_type}('{action.target_service}',"
                    f"'{action.ticket_severity}')"
                )

            err_field = (
                last_error
                or info.get("last_action_error")
                or "null"
            )

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward.score:.2f} done={'true' if done else 'false'} "
                f"error={err_field}",
                flush=True,
            )

            if done:
                success = info.get("success", False)
                break

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": _obs_to_text(obs)})

    except Exception as exc:
        last_error = str(exc)
        traceback.print_exc(file=sys.stderr)

    final_grade  = grade_fn(steps, task.max_steps, success)
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )
    return final_grade


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    tasks = [
        (EASY_TASK,   easy_grade,   "easy"),
        (MEDIUM_TASK, medium_grade, "medium"),
        (HARD_TASK,   hard_grade,   "hard"),
        (EXPERT_TASK, expert_grade, "expert"),
    ]

    grades = {}
    for task, grade_fn, label in tasks:
        g = run_task(task, grade_fn)
        grades[label] = g
        print(f"# Grade ({label}): {g}", file=sys.stderr, flush=True)

    avg = sum(grades.values()) / len(grades)
    print(f"# Average grade: {avg:.2f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
