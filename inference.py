"""
inference.py — Baseline LLM agent for OnCallEnv

Uses the OpenAI client to observe the environment, reason about the
incident, and take actions to resolve it.  Runs all three difficulty tasks
and prints results in the required [START]/[STEP]/[END] format.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from openai import OpenAI

from env.environment import ActionModel, OnCallEnv
from tasks.easy import TASK as EASY_TASK, grade as easy_grade
from tasks.medium import TASK as MEDIUM_TASK, grade as medium_grade
from tasks.hard import TASK as HARD_TASK, grade as hard_grade

# ── Environment variables (per guidelines) ───────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert SRE on-call engineer.  You are given the current state of
a distributed system with three services: api, worker, and db.

Your goal is to diagnose the incident and restore ALL services to "healthy"
status as quickly as possible.

Available actions (respond with ONE JSON object per turn):
  {"action_type": "restart_service", "target_service": "<name>"}
  {"action_type": "check_logs",      "target_service": "<name>"}
  {"action_type": "rollback",        "target_service": "<name>"}
  {"action_type": "scale_up",        "target_service": "<name>"}
  {"action_type": "scale_down",      "target_service": "<name>"}

Where <name> is one of: api, worker, db.

Rules:
- Output ONLY the JSON object.  No markdown, no explanation.
- Use check_logs to gather information before acting.
- Prioritize fixing root-cause services first.
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_ACTIONS = {"restart_service", "check_logs", "rollback", "scale_up", "scale_down"}
VALID_SERVICES = {"api", "worker", "db"}


def _obs_to_text(obs) -> str:
    """Convert an ObservationModel into a compact text summary for the LLM."""
    lines = [f"Step {obs.step_count}:"]
    for name, svc in obs.services.items():
        lines.append(
            f"  {name}: status={svc['status']}  cpu={svc['cpu_pct']}%  "
            f"mem={svc['memory_pct']}%  err={svc['error_rate']}  "
            f"latency={svc['latency_ms']}ms  restarts={svc['restart_count']}"
        )
        if svc.get("logs"):
            for log in svc["logs"][-3:]:
                lines.append(f"    [{log['level']}] {log['message']}")
    if obs.active_alerts:
        lines.append(f"  Alerts: {obs.active_alerts}")
    return "\n".join(lines)


def _parse_action(text: str) -> ActionModel | None:
    """Try to extract a valid ActionModel from the LLM response."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    action_type = data.get("action_type", "")
    target = data.get("target_service", "")

    if action_type not in VALID_ACTIONS or target not in VALID_SERVICES:
        return None

    return ActionModel(action_type=action_type, target_service=target)


# ── Run one task ─────────────────────────────────────────────────────────────

def run_task(task, grade_fn, env_name: str = "oncall_env"):
    """Execute a single task, printing [START]/[STEP]/[END] lines."""

    env = OnCallEnv(task=task)
    obs = env.reset(seed=42)
    rewards: list[float] = []
    steps = 0
    success = False
    error_msg: str | None = None

    print(f"[START] task={task.task_id} env={env_name} model={MODEL_NAME}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _obs_to_text(obs)},
    ]

    try:
        while True:
            # Ask the LLM for an action
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=150,
            )
            raw = response.choices[0].message.content or ""
            action = _parse_action(raw)
            error_msg = None

            if action is None:
                # Fallback: check_logs on api if LLM returned garbage
                action = ActionModel(action_type="check_logs", target_service="api")
                error_msg = f"parse_error: {raw[:80]}"

            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(round(reward.score, 2))

            action_str = f"{action.action_type}('{action.target_service}')"
            err_field = error_msg if error_msg else "null"
            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward.score:.2f} done={'true' if done else 'false'} "
                f"error={err_field}"
            )

            if done:
                success = info.get("success", False)
                break

            # Feed observation back
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": _obs_to_text(obs)})

    except Exception as exc:
        error_msg = str(exc)
        traceback.print_exc(file=sys.stderr)

    # Grade
    final_grade = grade_fn(steps, task.max_steps, success)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} "
        f"rewards={rewards_str}"
    )
    return final_grade


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tasks = [
        (EASY_TASK, easy_grade, "easy"),
        (MEDIUM_TASK, medium_grade, "medium"),
        (HARD_TASK, hard_grade, "hard"),
    ]

    grades = {}
    for task, grade_fn, label in tasks:
        g = run_task(task, grade_fn)
        grades[label] = g
        print(f"# Grade ({label}): {g}", file=sys.stderr)

    avg = sum(grades.values()) / len(grades)
    print(f"# Average grade: {avg:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
