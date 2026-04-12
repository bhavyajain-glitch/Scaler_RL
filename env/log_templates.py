"""
log_templates.py — Realistic, verbose log messages per failure type.

Each failure type has a sequence of log lines that build from first warning
to peak severity.  The messages are designed so that an LLM *must read and
interpret language* rather than just reacting to raw metric numbers.

Also contains:
  - Benign (red-herring) log sets for scenarios like nightly cron jobs.
  - `build_diagnostic_report()` for the run_diagnostic action.
"""

from __future__ import annotations

import random
from typing import List, Literal, Tuple


# ── Per-failure-type log templates ───────────────────────────────────────────
# Each list is ordered first-warning → peak-severity.
# Placeholders: {pct} {used} {pid} {n} {trx} {trx_a} {trx_b}

_FAILURE_LOGS: dict[str, list[tuple[str, str]]] = {
    "cpu_spike": [
        ("WARN",
         "Load average rising: 1m=2.4 5m=3.1 15m=1.8. Worker threads at 85% saturation."),
        ("WARN",
         "Request queue depth: 847 pending. P99 latency 3.2s (threshold 500ms)."),
        ("ERROR",
         "Thread pool exhausted ({n}/512 active). Incoming requests being shed."),
        ("ERROR",
         "CPU throttling detected. Scheduling delay 240 ms. Possible hot loop in request handler."),
        ("CRITICAL",
         "Watchdog: process CPU at {pct}% for 30 s. SLA breach imminent. "
         "Consider scale_up or investigate hot loop."),
    ],
    "oom": [
        ("WARN",
         "Heap usage at 78% ({used} MB / 8192 MB). GC pressure rising. Minor GC pause: 420 ms."),
        ("WARN",
         "Full GC triggered (stop-the-world: 1.8 s). Live objects: 2.1 GB. Heap not recovering."),
        ("ERROR",
         "Memory allocation failed: cannot allocate 524288 bytes. OOM killer scanning processes."),
        ("ERROR",
         "OOM killer invoked. Killed process {pid} (OOM score 847). RSS was 6.2 GB."),
        ("CRITICAL",
         "Service terminated by kernel OOM killer (exit code 137). "
         "Memory leak suspected. restart_service required."),
    ],
    "crash": [
        ("ERROR",
         "Unhandled exception: NullPointerException at RequestHandler.process():247."),
        ("ERROR",
         "SIGSEGV received (signal 11). Core dump written to /var/crash/service.core."),
        ("CRITICAL",
         "Process {pid} terminated unexpectedly (exit code 139 — segfault). Uptime was 14 d 3 h 22 m."),
        ("CRITICAL",
         "Watchdog: service heartbeat missed for 30 s. Marking service DOWN."),
        ("CRITICAL",
         "Health check FAILED 3/3 attempts. Service not responding on port 8080. "
         "restart_service required immediately."),
    ],
    "latency": [
        ("WARN",
         "Upstream response time: P50=890 ms  P95=3400 ms  P99=8200 ms. "
         "SLA threshold P99=500 ms BREACHED."),
        ("WARN",
         "Connection pool wait time elevated: avg 1.2 s. {n}/50 connections in queue."),
        ("ERROR",
         "Request timeout after 10 s: POST /api/v2/process. Downstream not responding."),
        ("ERROR",
         "Circuit breaker OPEN: upstream error rate {pct}% (threshold 20%). Failing fast. "
         "Root cause likely in upstream dependency — check db or worker."),
        ("ERROR",
         "Retry budget exhausted (3/3 attempts). Returning 503 to {n} waiting clients. "
         "Fix upstream dependency before restarting this service."),
    ],
    "db_lock": [
        ("WARN",
         "Lock wait timeout exceeded (50 s). TRX {trx} waiting on row lock. Query rolled back."),
        ("WARN",
         "InnoDB: {n} concurrent lock waits. Longest wait: 47 s on table `orders`. Deadlock risk high."),
        ("ERROR",
         "Deadlock detected. TRX {trx_a} rolled back to release lock held by TRX {trx_b}."),
        ("ERROR",
         "DB connection pool exhausted (100/100). {n} clients queued. Avg wait: 8.4 s. "
         "Downstream services (api, worker) will start failing."),
        ("CRITICAL",
         "Write throughput: 0 TPS. {n} transactions rolled back in 60 s. "
         "DB effectively locked. restart_service on db is the fix."),
    ],
}

# Benign / red-herring logs — look alarming but need NO action
_BENIGN_LOGS: dict[str, list[tuple[str, str]]] = {
    "cron_job": [
        ("INFO",
         "nightly-report cron started (scheduled 03:00 UTC). Expected duration: 8–12 min."),
        ("INFO",
         "Batch processing: 14,847 records queued. Worker threads 8/8 active. ETA: ~9 min."),
        ("WARN",
         "Batch job CPU elevated (82%). THIS IS EXPECTED for nightly aggregation — NO ACTION NEEDED."),
        ("INFO",
         "nightly-report progress: 7,423 / 14,847 records (50%). Running normally."),
    ],
    "routine_restart": [
        ("INFO",
         "Scheduled weekly restart (maintenance window 04:00–04:05 UTC)."),
        ("INFO",
         "Graceful shutdown complete. All connections drained. Restarting…"),
        ("INFO",
         "Service started successfully. All health checks passing. Normal operation resumed."),
    ],
}


# ── Public helpers ────────────────────────────────────────────────────────────

def get_failure_logs(
    failure_type: str,
    severity: Literal["low", "medium", "high"],
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """
    Return a list of ``(level, message)`` pairs for a failure injection.

    *severity* controls how many log lines are emitted:
      low → 2 lines, medium → 3 lines, high → all lines.
    """
    templates = _FAILURE_LOGS.get(failure_type, [])
    n = {"low": 2, "medium": 3, "high": len(templates)}[severity]
    result = []
    for level, msg in templates[:n]:
        msg = msg.format(
            pct=rng.randint(85, 99),
            used=rng.randint(5_800, 7_500),
            pid=rng.randint(1_000, 9_999),
            n=rng.randint(12, 89),
            trx=f"0x{rng.randint(0x1000, 0xFFFF):04X}",
            trx_a=f"0x{rng.randint(0x1000, 0xFFFF):04X}",
            trx_b=f"0x{rng.randint(0x1000, 0xFFFF):04X}",
        )
        result.append((level, msg))
    return result


def get_benign_logs(scenario: str) -> List[Tuple[str, str]]:
    """Return benign (red-herring) log lines for a named scenario."""
    return list(_BENIGN_LOGS.get(scenario, []))


def build_diagnostic_report(
    service_name: str,
    svc: dict,
    step: int,
) -> str:
    """
    Build a rich, human-readable diagnostic report for the ``run_diagnostic`` action.

    The report intentionally requires language reasoning: an agent must read
    the 'Likely cause' and 'Recommendation' fields and act accordingly — a
    pure rule-based script cannot interpret these correctly.
    """
    cpu     = svc.get("cpu_pct", 0.0)
    mem     = svc.get("memory_pct", 0.0)
    err     = svc.get("error_rate", 0.0)
    lat     = svc.get("latency_ms", 50.0)
    restarts = svc.get("restart_count", 0)
    logs    = svc.get("logs", [])
    status  = svc.get("status", "unknown")

    # Combine recent log messages for keyword scan
    recent_text = " ".join(l.get("message", "") for l in logs[-4:]).lower()

    # Infer likely cause from metrics + log content
    if "cron" in recent_text or "batch" in recent_text or "nightly" in recent_text:
        cause = "High CPU from scheduled batch/cron job — THIS IS EXPECTED, not a failure."
        rec   = "No action required. CPU will drop when cron job completes (~10 min)."
    elif err >= 0.80:
        cause = "Total service failure — crash, OOM, or forced kill (error_rate ≥ 80%)."
        rec   = "Immediate restart_service required. Review CRITICAL log lines above."
    elif mem >= 85.0:
        cause = "Severe memory pressure — OOM risk (memory ≥ 85%). GC thrashing."
        rec   = "run scale_up to reduce load, then restart_service to reclaim heap."
    elif cpu >= 85.0:
        cause = "CPU saturation — hot loop or sudden traffic spike."
        rec   = "scale_up to relieve CPU. Check for request storms or runaway queries."
    elif lat >= 500.0:
        cause = "Latency degradation — likely upstream dependency (db / worker) is struggling."
        rec   = "Do NOT restart this service yet. Fix the upstream root cause first."
    elif err >= 0.20:
        cause = "Elevated error rate — possible config drift or bad deploy."
        rec   = "Review recent deploys. rollback if a bad deploy is confirmed."
    else:
        cause = "No anomaly detected. Service metrics within normal operating range."
        rec   = "No action required. Continue monitoring."

    recent_errors = [l for l in logs if l.get("level") in ("ERROR", "CRITICAL")]
    last_err = recent_errors[-1]["message"][:130] if recent_errors else "None."

    return (
        f"DIAGNOSTIC REPORT — {service_name}  (step {step})\n"
        f"{'─' * 54}\n"
        f"Status     : {status}\n"
        f"CPU        : {cpu:.1f}%    Memory  : {mem:.1f}%\n"
        f"Error rate : {err:.1%}     Latency : {lat:.0f} ms\n"
        f"Restarts   : {restarts}\n"
        f"{'─' * 54}\n"
        f"Likely cause  : {cause}\n"
        f"Last error    : {last_err}\n"
        f"Recommendation: {rec}\n"
        f"{'─' * 54}"
    )
