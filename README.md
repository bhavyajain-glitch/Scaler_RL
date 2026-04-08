---
title: OnCallEnv
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# OnCallEnv — RL Environment for On-Call Incident Response

An OpenEnv-compliant reinforcement learning environment that simulates
distributed system incidents (API, Worker, DB) for an LLM agent to diagnose
and resolve.

---

## Motivation

On-call incident response is a **real-world task** performed daily by SREs
across the industry.  This environment lets RL agents practise the core
decision loop: *observe metrics → diagnose root cause → take corrective action*.

---

## Action & Observation Spaces

### Actions

| Action | Description |
|---|---|
| `restart_service` | Restart a service (fixes crashes) |
| `check_logs` | Inspect recent logs (informational, reward = 0) |
| `rollback` | Roll back the last deployment |
| `scale_up` | Add capacity (reduces CPU pressure) |
| `scale_down` | Remove capacity |

Each action targets one of: `api`, `worker`, `db`.

### Observations

```json
{
  "step_count": 0,
  "services": {
    "api":    {"status": "healthy", "cpu_pct": 10.0, "memory_pct": 20.0, "error_rate": 0.0, "latency_ms": 50.0, "restart_count": 0, "uptime_steps": 0, "logs": [...]},
    "worker": { ... },
    "db":     { ... }
  },
  "active_alerts": [],
  "recent_actions": []
}
```

### Reward Table

| Action | Condition | Reward |
|---|---|---|
| `restart_service` | was down → now healthy | +5 |
| `restart_service` | was already healthy | −2 |
| `rollback` | root cause was deploy | +7 |
| `rollback` | wrong diagnosis | −3 |
| `scale_up/down` | CPU > 85% | +3 |
| `scale_up/down` | CPU < 50% | −1 |
| `check_logs` | always | 0 |
| *time penalty* | per step | −0.5 |
| *resolution bonus* | all services healthy | +10 |

---

## Tasks

| ID | Difficulty | Scenario | Max Steps |
|---|---|---|---|
| `easy_db_crash` | Easy | DB crashes; restart it | 10 |
| `medium_api_latency_worker_oom` | Medium | API latency + worker OOM | 15 |
| `hard_cascading_db_lock` | Hard | DB lock cascades to worker → API | 20 |

Each task has a **grader** that returns a score between **0.0** and **1.0**
based on whether the task was resolved and how many steps were used.

---

## Setup & Usage

### Local (Python)

```bash
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (requires HF_TOKEN)
export HF_TOKEN="your-token"
python inference.py
```

### Docker

```bash
docker build -t oncall-env .
docker run -p 7860:7860 oncall-env
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Start new episode `{"task_id": "...", "seed": 42}` |
| `POST` | `/step` | Take action `{"session_id": "...", "action_type": "...", "target_service": "..."}` |
| `GET` | `/state/{session_id}` | Read-only current state |

---

## Baseline Performance

| Task | Model | Steps | Grade |
|---|---|---|---|
| `easy_db_crash` | gpt-4.1-mini | 2 | 0.86 |
| `medium_api_latency_worker_oom` | gpt-4.1-mini | 4 | 0.71 |
| `hard_cascading_db_lock` | gpt-4.1-mini | 6 | 0.54 |

*(Scores may vary with different models and API endpoints.)*

---

## Project Structure

```
├── env/
│   ├── services.py      # ServiceNode, LogEntry, ServiceRegistry
│   ├── failures.py      # FailureEvent, FailureInjector
│   ├── reward.py         # RewardSignal, RewardCalculator
│   └── environment.py    # OnCallEnv (reset/step/state)
├── tasks/
│   ├── easy.py           # Single-service crash
│   ├── medium.py         # Multi-service failure
│   └── hard.py           # Cascading failure
├── server/
│   └── app.py            # FastAPI entry point
├── inference.py          # Baseline LLM agent
├── openenv.yaml          # OpenEnv manifest
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Default | Required |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4.1-mini` | No |
| `HF_TOKEN` | — | **Yes** |
