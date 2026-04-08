# Agent Interaction Model

## Agent Role
AI agent acts as an on-call engineer.

## Inputs
- logs
- metrics
- alerts

## Actions
- check_logs(service)
- restart_service(service)
- rollback(service)
- scale(service)

## Expected Behavior
1. Read logs
2. Identify failure pattern
3. Find root cause
4. Apply correct fix
5. Verify recovery

## Failure Modes
- Fixing symptom instead of root cause
- Random actions (penalized)
- Ignoring logs
