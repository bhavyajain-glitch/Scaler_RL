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
real-world distributed system incidents for an LLM agent to diagnose and resolve.

The agent must **read and interpret verbose, realistic log messages** —
not just react to numeric metrics — making this a genuine language-reasoning
challenge rather than a rule-based lookup table problem.

---

## Motivation

On-call incident response is a **real-world task** performed daily by SREs.
Unlike toy grid worlds, the agent faces:

- **Ambiguous signals** — multiple services showing symptoms, only one is the root cause.
- **Red herrings** — benign elevated load (e.g. a nightly cron job) that looks like a failure.
- **Cascading failures** — fixing the wrong service first makes things worse.
- **Language-rich observations** — log messages with deadlock TRX IDs, OOM killer PID reports,
  circuit-breaker states, and runbook-style recommendations embedded in diagnostics.

---

## Action Space

| Action | Target | Description |
|--------|--------|-------------|
| `run_diagnostic` | service | Returns a rich diagnostic report with likely cause and recommendation. **Use before fixing.** |
| `check_logs` | service | Inspect recent log ring-buffer entries. |
| `restart_service` | service | Restart a crashed/OOM service. +5 reward if it was down. |
| `rollback` | service | Roll back the last deployment (+7 if deploy-caused, −3 otherwise). |
| `scale_up` | service | Add capacity to relieve CPU pressure (+3 if CPU > 85%). |
| `scale_down` | service | Reduce capacity. |
| `create_incident_ticket` | service + severity | File a ticket naming the root-cause service and severity (`sev1`/`sev2`/`sev3`). Earns +4 if accurate. |

Each action targets one of: `api`, `worker`, `db`.

---

## Observation Space

```json
{
  "step_count": 3,
  "services": {
    "db": {
      "status": "down",
      "cpu_pct": 12.0,
      "memory_pct": 97.4,
      "error_rate": 1.0,
      "latency_ms": 50.0,
      "restart_count": 0,
      "uptime_steps": 0,
      "logs": [
        {"level": "ERROR",    "message": "Memory allocation failed: cannot allocate 524288 bytes. OOM killer scanning processes."},
        {"level": "CRITICAL", "message": "Service terminated by kernel OOM killer (exit code 137). Memory leak suspected. restart_service required."}
      ]
    },
    "worker": { "... (with cron logs explaining benign high CPU) ..." },
    "api":    { "... (degraded by db cascade) ..." }
  },
  "active_alerts": ["Diagnostic run on db — see last_diagnostic."],
  "recent_actions": ["run_diagnostic:db"],
  "incident_context": [
    "Step 1: run_diagnostic(db) — report available in last_diagnostic",
    "Step 2: create_incident_ticket(db, sev1)"
  ],
  "last_diagnostic": "DIAGNOSTIC REPORT — db (step 1)\n..."
}
```

The `last_diagnostic` field contains the full output of the most recent `run_diagnostic` call,
including a plain-English *Likely cause* and *Recommendation*.

---

## Reward Table

| Component | Condition | Reward |
|-----------|-----------|--------|
| `time_penalty` | every step | −0.5 |
| `restart_service` | was `down` → now `healthy` | +5.0 |
| `restart_service` | was already `healthy` | −2.0 |
| `rollback` | `is_deploy_failure=True` | +7.0 |
| `rollback` | wrong tool for failure | −3.0 |
| `scale_up/down` | CPU > 85% | +3.0 |
| `scale_up/down` | CPU < 50% | −1.0 |
| `run_diagnostic` | on root-cause service | +1.0 |
| `run_diagnostic` | on other service | +0.3 |
| `create_incident_ticket` | correct service + severity | +4.0 |
| `create_incident_ticket` | correct service only | +1.5 |
| `create_incident_ticket` | wrong service | −1.0 |
| `diagnosis_first` bonus | run_diagnostic/check_logs on service *before* fixing it | +1.5 |
| `root_cause_first` bonus | fixing root-cause service while downstream still degraded | +2.0 |
| `task_resolved` | all services healthy | +10.0 |

---

## Tasks

| ID | Difficulty | Scenario | Max Steps | Optimal Steps |
|----|-----------|----------|-----------|---------------|
| `easy_db_crash` | Easy | DB crashes; restart it | 10 | 1–2 |
| `medium_api_latency_worker_oom` | Medium | API latency spike + Worker OOM simultaneously | 15 | 2–3 |
| `hard_cascading_db_lock` | Hard | DB lock cascades degradation to worker then API | 20 | 3–4 |
| `expert_red_herring` | Expert | DB OOM + API cascade + worker has benign nightly cron (red herring) | 20 | 3–4 |

### Expert Task — Why It Requires Language Understanding

The expert task has three things happening at once:
- **Worker** — CPU at ~84%, but logs say *"nightly-report cron started… NO ACTION NEEDED"*
- **DB** — OOM crash with `CRITICAL: Service terminated by kernel OOM killer`
- **API** — degraded by DB cascade, logs say *"Fix upstream dependency before restarting"*

A rule-based script restarts everything and wastes 3 extra steps. An LLM that reads the logs skips the worker, identifies DB as root cause, and resolves the incident in the optimal number of steps.

---

## Setup & Usage

### Local (Python)

```bash
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline inference agent (requires HF_TOKEN)
export HF_TOKEN="your-hf-token"
export MODEL_NAME="gpt-4.1-mini"          # or any OpenAI-compatible model
export API_BASE_URL="https://api.openai.com/v1"
python inference.py
```

### Docker

```bash
docker build -t oncall-env .
docker run -p 7860:7860 -e HF_TOKEN="your-token" oncall-env
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check → `{"status": "ok"}` |
| `GET`  | `/tasks` | List all tasks with difficulty and max_steps |
| `POST` | `/reset` | Start episode: `{"task_id": "easy_db_crash", "seed": 42}` |
| `POST` | `/step`  | Take action: `{"session_id": "...", "action_type": "restart_service", "target_service": "db"}` |
| `GET`  | `/state/{session_id}` | Read-only current observation |

---

## Baseline Performance

Scores measured with `gpt-4.1-mini`, `seed=42`, `temperature=0`.

| Task | Steps | Success | Grade |
|------|-------|---------|-------|
| `easy_db_crash` | 2 | ✓ | 0.86 |
| `medium_api_latency_worker_oom` | 4 | ✓ | 0.74 |
| `hard_cascading_db_lock` | 6 | ✓ | 0.62 |
| `expert_red_herring` | 6 | ✓ | 0.68 |
| **Average** | — | — | **0.73** |

A smarter model (e.g. GPT-4o or Claude 3.5) running a chain-of-thought strategy should
approach 0.90+ average by correctly interpreting diagnostic reports and cron-job red herrings.

---

## Project Structure

```
├── env/
│   ├── log_templates.py  # Realistic per-failure-type log messages + diagnostic builder
│   ├── services.py       # ServiceNode, LogEntry, ServiceRegistry
│   ├── failures.py       # FailureEvent, FailureInjector
│   ├── reward.py         # RewardSignal, RewardCalculator
│   └── environment.py    # OnCallEnv (reset / step / state)
├── tasks/
│   ├── easy.py           # Single-service crash
│   ├── medium.py         # Multi-service simultaneous failure
│   ├── hard.py           # Cascading root-cause failure
│   └── expert.py         # Red-herring + cascading OOM (requires log reading)
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
|----------|---------|----------|
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4.1-mini` | No |
| `HF_TOKEN` | — | **Yes** |
