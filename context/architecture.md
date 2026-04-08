# OnCallEnv —  environment.py  Architecture Design

This document defines the *clean architecture* for the OnCallEnv RL environment before any code is written.
The goal is to simulate distributed system incidents (API, Worker, DB) that an LLM agent must diagnose and resolve.

---

## Layer Map


oncallenv/
├── env/
│   ├── environment.py   ← Core RL loop (OnCallEnv class)
│   ├── services.py      ← ServiceNode models + state machine
│   ├── failures.py      ← Failure injection engine
│   └── reward.py        ← Reward computation logic
├── tasks/
│   ├── easy.py          ← Task definitions (single-service)
│   ├── medium.py        ← Task definitions (multi-service)
│   └── hard.py          ← Task definitions (cascading failures)


---

## Class Design

### 1.  services.py  — The Infrastructure Layer


ServiceNode (Pydantic BaseModel)
  ├── name: str                          # "api", "worker", "db"
  ├── status: Literal["healthy", "degraded", "down"]
  ├── cpu_pct: float                     # 0–100
  ├── memory_pct: float                  # 0–100
  ├── error_rate: float                  # 0.0–1.0
  ├── latency_ms: float
  ├── restart_count: int
  └── logs: List[LogEntry]              # ring buffer, max 50

LogEntry (Pydantic BaseModel)
  ├── timestamp: float
  ├── level: Literal["INFO", "WARN", "ERROR", "CRITICAL"]
  └── message: str

ServiceRegistry
  ├── services: Dict[str, ServiceNode]
  ├── tick()                            # advances simulated time, decays metrics
  └── get_snapshot() → dict            # serializable state of all services


*Responsibility:* Pure data layer. Knows nothing about tasks or rewards.

---

### 2.  failures.py  — The Fault Injection Engine


FailureEvent (Pydantic BaseModel)
  ├── service: str
  ├── failure_type: Literal["cpu_spike", "oom", "crash", "latency", "db_lock"]
  ├── severity: Literal["low", "medium", "high"]
  ├── cascade_to: Optional[List[str]]  # downstream services affected
  └── auto_heal_after: Optional[int]   # steps until self-heal (None = manual only)

FailureInjector
  ├── inject(event: FailureEvent, registry: ServiceRegistry)
  ├── cascade(registry: ServiceRegistry)   # propagate downstream effects
  └── tick_heal(registry: ServiceRegistry) # decrement timers, auto-heal if due


*Responsibility:* Applies failure state deltas to  ServiceRegistry . Deterministic given a seed.

---

### 3.  reward.py  — The Reward Function


RewardSignal (Pydantic BaseModel)
  ├── score: float
  ├── reason: str
  └── breakdown: Dict[str, float]      # e.g. {"resolution": +5, "time_penalty": -1}

RewardCalculator
  ├── compute(action, pre_state, post_state, task) → RewardSignal
  └── _penalize_time(steps_elapsed) → float


*Reward logic (per action type):*
| Action | Condition | Reward |
|---|---|---|
|  restart_service  | service was down, now healthy | +5 |
|  restart_service  | service was healthy (wasted) | -2 |
|  rollback  | correct service + root cause was deploy | +7 |
|  rollback  | wrong service | -3 |
|  scale  | cpu_pct > 85 | +3 |
|  scale  | cpu_pct < 50 | -1 |
|  check_logs  | always informational | 0 |
| Time penalty | per step elapsed | -0.5 |
| Task resolved | all services healthy | +10 bonus |

---

### 4.  environment.py  — The RL Interface (Main Class)


ObservationModel (Pydantic BaseModel)
  ├── step_count: int
  ├── services: Dict[str, ServiceNode]
  ├── active_alerts: List[str]
  └── recent_actions: List[str]         # last 5 actions taken

ActionModel (Pydantic BaseModel)
  ├── action_type: Literal["restart_service", "check_logs", "rollback", "scale"]
  └── target_service: str

TaskSpec (Pydantic BaseModel)
  ├── task_id: str
  ├── difficulty: Literal["easy", "medium", "hard"]
  ├── failure_sequence: List[FailureEvent]
  ├── success_condition: Callable[[ServiceRegistry], bool]
  └── max_steps: int

OnCallEnv
  ├── __init__(task: TaskSpec)
  │
  ├── reset(seed: int) → ObservationModel
  │   # 1. Seed the RNG
  │   # 2. Re-initialize ServiceRegistry to all-healthy
  │   # 3. Inject the task's failure_sequence
  │   # 4. Clear action history & step count
  │   # 5. Return initial observation
  │
  ├── step(action: ActionModel) → Tuple[ObservationModel, RewardSignal, bool, dict]
  │   # 1. Validate action (Pydantic)
  │   # 2. Capture pre-state snapshot
  │   # 3. Execute action → mutate ServiceRegistry
  │   # 4. Tick environment (cascade failures, decay, auto-heals)
  │   # 5. Compute reward(pre, post, action, task)
  │   # 6. Check done: success_condition OR step_count >= max_steps
  │   # 7. Return (obs, reward, done, info)
  │
  └── state() → ObservationModel
      # Read-only current observation (no side effects)


---

## Data Flow


reset(seed)
    └─► ServiceRegistry.reset()
    └─► FailureInjector.inject(task.failure_sequence)
    └─► return state()

step(action)
    └─► pre_state = state()
    └─► _execute_action(action)         → mutates ServiceRegistry
    └─► FailureInjector.tick_heal()     → auto-heals, cascades
    └─► ServiceRegistry.tick()          → advance time, emit logs
    └─► reward = RewardCalculator.compute(...)
    └─► done = task.success_condition(registry) or timeout
    └─► return (state(), reward, done, info)


---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Pydantic models for all I/O | OpenEnv spec compliance; free validation |
|  ServiceRegistry  is stateful, mutable | Avoids deep-copying on every step |
|  ObservationModel  is immutable snapshots | Agent sees clean read-only views |
|  FailureEvent.cascade_to  | Models real dependency chains (worker → db) |
|  RewardSignal.breakdown  | Interpretable reward for debugging agent behavior |
|  LogEntry  ring buffer (50 entries) | Bounded memory; agent can use  check_logs  to read |
|  TaskSpec.success_condition  is a callable | Flexible: easy tasks check one service, hard tasks check all |
| RNG seeded in  reset()  | Deterministic replay; OpenEnv compliance |

---

## Open Questions

> [!IMPORTANT]
> *Q1: Action space* — Should  scale  accept a  replica_count  parameter, or just be binary (scale_up/scale_down)?

> [!IMPORTANT]
> *Q2: Observation verbosity* — Should  check_logs  return all 50 entries or only the last N (configurable per task)?

> [!IMPORTANT]
> *Q3: Multi-agent* — Is this single-agent only for now, or should the architecture leave a slot for a second agent role (e.g., "SRE + Manager")?

---

## Verification Plan

1.  reset()  — assert all services start healthy, step_count = 0
2.  step()  — inject a "db crash" task, assert observer sees error_rate spike
3. Reward — restart a down service, assert reward > 0
4. Done condition — resolve all services, assert  done=True 
5. Determinism —  reset(seed=42)  twice, assert identical initial states