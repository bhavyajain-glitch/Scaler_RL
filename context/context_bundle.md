# PROJECT: OnCallEnv

## Goal
Simulate distributed system failures and evaluate AI debugging ability.

## Architecture
- Python-based simulation
- Services: A (API), B (Worker), C (DB)
- Failure injection system
- Log generator with noise
- Health score (0 → 1)

## Agent
- Reads logs, metrics
- Takes actions:
  - restart_service
  - check_logs
  - rollback
  - scale

## Tasks
- Easy: single failure
- Medium: dependency chain
- Hard: multi-failure + noise

## Reward
+ correct diagnosis
+ correct action
+ system recovery
- wrong action
- wasted steps

## Constraints
- 2 vCPU, 8GB RAM
- No real infrastructure
- Fully simulated

## Current Task
[UPDATE THIS BEFORE EACH SESSION]