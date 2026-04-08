# System Architecture

## Core Components

### 1. Environment Core (OpenEnv)
- step(action)
- reset(seed)
- state()

### 2. Service Simulator
Simulated services:
- Service A (API)
- Service B (Worker)
- Service C (Database)

Each service:
- status (healthy/down)
- latency
- error_rate

### 3. Failure Injector
Injects failures:
- service_down
- latency_spike
- bad_config
- dependency_failure

### 4. Log Generator
Produces logs like:
[ERROR] Service B timeout connecting to Service C
[WARN] Retry attempt 3 failed

Includes:
- noise logs
- misleading logs

### 5. Health Engine
Tracks:
- service health
- latency
- error rates

Outputs:
health_score (0.0 → 1.0)

## Flow

reset() →
  generate services →
  inject failures →
  generate logs →
  return observation

step(action) →
  apply action →
  update services →
  generate logs →
  compute reward →
  return observation