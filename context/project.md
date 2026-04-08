# OnCallEnv — Distributed System Failure Benchmark

## Overview
OnCallEnv is an OpenEnv-compatible reinforcement learning environment designed to simulate real-world distributed system failures.

It evaluates an AI agent’s ability to:
- Diagnose issues from noisy logs
- Perform multi-step debugging
- Identify root causes
- Restore system health

## Problem
Current AI systems struggle with:
- Multi-step reasoning
- Root cause analysis
- Noisy and misleading logs
- Cascading failures

There is no standardized benchmark for real-world debugging tasks under uncertainty.

## Solution
A simulated environment where:
- Multiple services interact
- Failures are injected dynamically
- Logs and metrics evolve over time
- Agent must take actions to fix the system

## Key Features
- Lightweight simulation (no real infra)
- Deterministic seeds
- Realistic logs with noise
- Multi-step reward system
- Increasing difficulty tasks

## Success Criteria
- Agent restores system health (score → 1.0)
- Efficient debugging (fewer steps)
- Correct root cause identification