"""
server/app.py — FastAPI entry point for OnCallEnv

Exposes the environment over HTTP so it can be consumed by
OpenEnv clients and the hackathon evaluation harness.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import ActionModel, OnCallEnv, ObservationModel, TaskSpec
from env.failures import FailureEvent
from env.services import ServiceRegistry
from tasks.easy import TASK as EASY_TASK, grade as easy_grade
from tasks.medium import TASK as MEDIUM_TASK, grade as medium_grade
from tasks.hard import TASK as HARD_TASK, grade as hard_grade


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="OnCallEnv",
    description="RL environment for on-call SRE incident response",
    version="1.0.0",
)

# ── Task registry ────────────────────────────────────────────────────────────

TASKS = {
    "easy_db_crash": {"task": EASY_TASK, "grade": easy_grade},
    "medium_api_latency_worker_oom": {"task": MEDIUM_TASK, "grade": medium_grade},
    "hard_cascading_db_lock": {"task": HARD_TASK, "grade": hard_grade},
}

# ── Session storage (in-memory, single-user for hackathon) ───────────────────

_sessions: Dict[str, Dict[str, Any]] = {}


# ── Request / response models ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_db_crash"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    target_service: str


class ResetResponse(BaseModel):
    session_id: str
    observation: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    observation: dict


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """List available tasks."""
    return {
        tid: {"difficulty": t["task"].difficulty, "max_steps": t["task"].max_steps}
        for tid, t in TASKS.items()
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    """Start a new episode for the given task."""
    if req.task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {req.task_id}")

    task_entry = TASKS[req.task_id]
    env = OnCallEnv(task=task_entry["task"])
    obs = env.reset(seed=req.seed)

    import uuid
    session_id = str(uuid.uuid4())[:8]
    _sessions[session_id] = {
        "env": env,
        "task_id": req.task_id,
        "grade_fn": task_entry["grade"],
        "rewards": [],
    }

    return ResetResponse(
        session_id=session_id,
        observation=obs.model_dump(),
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Execute one action in the environment."""
    session = _sessions.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    env: OnCallEnv = session["env"]
    action = ActionModel(action_type=req.action_type, target_service=req.target_service)
    obs, reward, done, info = env.step(action)

    session["rewards"].append(reward.score)

    if done:
        grade = session["grade_fn"](env.step_count, env.task.max_steps, info["success"])
        info["grade"] = grade

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.score,
        done=done,
        info=info,
    )


@app.get("/state/{session_id}", response_model=StateResponse)
def state(session_id: str):
    """Get current state without side effects."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    env: OnCallEnv = session["env"]
    return StateResponse(observation=env.state().model_dump())


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
