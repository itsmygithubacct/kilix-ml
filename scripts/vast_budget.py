#!/usr/bin/env python3
"""Budget-bounded orchestration for one already-rented, task-owned Vast instance.

Never provisions. Every CLI operation carries the explicitly selected ID through
the existing /home/pleb/vast-ai-deployment/vast launcher. `plan` is offline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

MAX_BUDGET_USD = 120.0
PROBE_MAX_SECONDS = 30 * 60
RESUME_DEADLINE_TOLERANCE_SECONDS = 2.0
TARGET_TOKENS = 30_000_000_000
POLL_SECONDS = 20
PHASE_STATUS_ATTEMPTS = 3


class BudgetError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def plan(rate: float, accrued: float, teardown_reserve: float, sft_reserve: float,
         acquire_max_seconds: int, probe_seconds: int | None = None) -> dict:
    for label, value in (("hourly rate", rate), ("accrued cost", accrued),
                         ("teardown reserve", teardown_reserve), ("SFT reserve", sft_reserve)):
        if not math.isfinite(value):
            raise BudgetError(f"{label} must be finite")
    if rate <= 0 or accrued < 0 or teardown_reserve <= 0 or sft_reserve <= 0:
        raise BudgetError("rate must be positive; accrued cost nonnegative; both reserves positive")
    if type(acquire_max_seconds) is not int or acquire_max_seconds < 1:
        raise BudgetError("acquisition time limit must be a positive integer number of seconds")
    if probe_seconds is None:
        probe_seconds = PROBE_MAX_SECONDS
    if type(probe_seconds) is not int or probe_seconds < 0 or probe_seconds > PROBE_MAX_SECONDS:
        raise BudgetError("probe time must be an integer from zero through the 30-minute probe cap")
    available = MAX_BUDGET_USD - accrued - teardown_reserve
    if available <= 0:
        raise BudgetError("accrued cost plus teardown reserve must be below $120")
    work_seconds = math.floor(available * 3600 / rate)
    sft_seconds = math.ceil(sft_reserve * 3600 / rate)
    if sft_reserve >= available or sft_seconds >= work_seconds:
        raise BudgetError("SFT reserve must fit inside the work budget after teardown reserve")
    teardown_seconds = math.ceil(teardown_reserve * 3600 / rate)
    if teardown_seconds < 135:
        raise BudgetError("teardown reserve must cover the 60-second destroy timeout plus graceful shutdown")
    pre_sft_seconds = work_seconds - sft_seconds
    total_seconds = math.floor((MAX_BUDGET_USD - accrued) * 3600 / rate)
    planned_pretrain_seconds = pre_sft_seconds - probe_seconds - acquire_max_seconds
    if planned_pretrain_seconds <= 0:
        raise BudgetError("budget must leave positive pretraining time after the probe and acquisition caps")
    return {
        "budget_usd": MAX_BUDGET_USD,
        "hourly_rate_usd": rate,
        "accrued_cost_usd": accrued,
        "teardown_reserve_usd": teardown_reserve,
        "sft_reserve_usd": sft_reserve,
        "work_seconds_before_teardown_reserve": work_seconds,
        "sft_reserve_seconds": sft_seconds,
        "pretrain_and_probe_seconds_before_sft_reserve": pre_sft_seconds,
        "teardown_reserve_seconds": teardown_seconds,
        "total_seconds_at_declared_rate": total_seconds,
        "probe_cap_seconds": probe_seconds,
        "acquisition_cap_seconds": acquire_max_seconds,
        "planned_pretrain_seconds_after_full_probe_and_acquisition": planned_pretrain_seconds,
        "pretrain_target_tokens": TARGET_TOKENS,
        "minimum_pretrain_tokens_per_second_after_full_probe_and_acquisition": (
            TARGET_TOKENS / planned_pretrain_seconds
        ),
    }


def redact(text: str) -> str:
    text = re.sub(r"(?i)(api[_-]?key|hf[_-]?token|access[_-]?token|password|secret)(\s*[:=]\s*)([^\s,;]+)",
                  r"\1\2[REDACTED]", text)
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"\b(?:hf_[A-Za-z0-9]{12,}|(?:sk|rk)-[A-Za-z0-9_-]{16,})\b", "[REDACTED]", text)
    return text.strip()[-1000:]


def append_ledger(path: Path, *, event: str, instance: str, rate: float,
                  accrued: float, started: float, detail: dict | None = None) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    row = {
        "time_utc": utc_now(), "event": event, "instance_id": instance,
        "elapsed_seconds": round(elapsed, 3), "hourly_rate_usd": rate,
        "accrued_cost_usd": accrued,
        "estimated_total_cost_usd": round(accrued + rate * elapsed / 3600.0, 6),
        "budget_usd": MAX_BUDGET_USD,
    }
    if detail:
        row["detail"] = detail
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Vast:
    def __init__(self, launcher: Path, instance: str):
        self.launcher = launcher
        self.instance = instance

    def call(self, *args: str, timeout: int = 180) -> str:
        command = [str(self.launcher), "--instance", self.instance, *args]
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=max(1, timeout), check=False)
        except subprocess.TimeoutExpired:
            raise BudgetError(f"Vast launcher timed out during {args[0]}") from None
        except OSError as exc:
            raise BudgetError(f"Vast launcher could not run ({type(exc).__name__})") from None
        if result.returncode:
            detail = redact(result.stderr or result.stdout or "")
            suffix = f": {detail}" if detail else ""
            raise BudgetError(f"Vast launcher {args[0]} failed with exit status {result.returncode}{suffix}")
        return result.stdout

    def status(self) -> str:
        return self.call("status", timeout=30)

    def run_remote(self, script: str, timeout: int = 180) -> str:
        return self.call("run", "--", "bash", "-lc", script, timeout=timeout)

    def download(self, source: str, destination: Path, timeout: int) -> None:
        self.call("download", source, str(destination), timeout=timeout)

    def sync_download(self, source: str, destination: Path, timeout: int) -> None:
        # Reuse the deployment launcher’s selected-instance SSH resolution and key path;
        # rsync keeps completed files and resumes interrupted files between attempts.
        loader = importlib.machinery.SourceFileLoader("vast_launcher_for_sync", str(self.launcher))
        spec = importlib.util.spec_from_loader("vast_launcher_for_sync", loader)
        if spec is None or spec.loader is None:
            raise BudgetError("could not load the existing launcher for resumable download")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.load_config()
        options, target = module.ssh_options(config, self.instance)
        destination.mkdir(parents=True, exist_ok=True)
        ssh = shlex.join(["ssh", *options])
        command = ["rsync", "-azP", "--partial", "--append-verify", "-e", ssh,
                   f"{target}:{source.rstrip('/')}/", str(destination) + "/"]
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=max(1, timeout), check=False)
        except subprocess.TimeoutExpired:
            raise BudgetError("resumable checkpoint transfer timed out") from None
        except OSError as exc:
            raise BudgetError(f"resumable checkpoint transfer could not run ({type(exc).__name__})") from None
        if result.returncode:
            detail = redact(result.stderr or result.stdout or "")
            raise BudgetError("resumable checkpoint transfer failed" + (f": {detail}" if detail else ""))

    def destroy(self, timeout: int = 60) -> None:
        # One explicit instance ID is prepended by call(); --all is never used.
        self.call("destroy", "--yes", timeout=timeout)


def verify_selected_instance(status: str, instance: str, declared_rate: float) -> float:
    found = None
    for line in status.splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        if fields and fields[0] == instance and len(fields) >= 4:
            try:
                found = float(fields[3])
            except ValueError:
                continue
    if found is None:
        raise BudgetError("selected instance ID was not present in Vast status output")
    # The project launcher displays dph_total (or its available fallback) to 4 decimals.
    if abs(found - declared_rate) > 0.00015:
        raise BudgetError(f"declared rate ${declared_rate:.4f}/hr differs from status ${found:.4f}/hr")
    return found


def remote_dir(instance: str) -> str:
    return f"/tmp/kilix-ml-budget-{re.sub(r'[^A-Za-z0-9_-]', '_', instance)}"


def phase_start(vast: Vast, instance: str, phase: str, command: str,
                seconds: int, progress_file: str = "", checkpoint_dir: str = "") -> None:
    if seconds <= 0:
        raise BudgetError(f"no execution time remains for {phase}")
    d = remote_dir(instance)
    env = "export KILIX_TARGET_TOKENS=30000000000"
    if progress_file:
        env += " KILIX_PROGRESS_FILE=" + shlex.quote(progress_file)
    if checkpoint_dir:
        env += " KILIX_CHECKPOINT_DIR=" + shlex.quote(checkpoint_dir)
    env += " KILIX_STAGE_SECONDS=" + str(seconds)
    runner = env + "; set +e; bash -lc " + shlex.quote(command) + "; rc=$?; printf '%s\\n' \"$rc\" >" + shlex.quote(d + "/" + phase + ".exit") + "; exit \"$rc\""
    watchdog = (
        "sleep " + str(seconds) + "; d=" + shlex.quote(d) + "; "
        "p=$(cat \"$d/" + phase + ".pid\" 2>/dev/null || true); "
        "expected=$(cat \"$d/" + phase + ".start\" 2>/dev/null || true); "
        "current=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "group=0; kill -0 -- -\"$p\" 2>/dev/null && group=1 || true; safe=0; "
        "if [ -n \"$expected\" ] && [ \"$current\" = \"$expected\" ] && [ \"$group\" = 1 ]; then safe=1; "
        "elif [ \"$group\" = 1 ] && { [ ! -e \"/proc/$p/stat\" ] || [ \"$pg\" != \"$p\" ]; }; then safe=1; fi; "
        "if [ \"$safe\" = 1 ]; then printf 'deadline\\n' >\"$d/" + phase + ".timeout\"; "
        "kill -INT -- -\"$p\" 2>/dev/null || true; sleep 60; "
        "current=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "group=0; kill -0 -- -\"$p\" 2>/dev/null && group=1 || true; safe=0; "
        "if [ -n \"$expected\" ] && [ \"$current\" = \"$expected\" ] && [ \"$group\" = 1 ]; then safe=1; "
        "elif [ \"$group\" = 1 ] && { [ ! -e \"/proc/$p/stat\" ] || [ \"$pg\" != \"$p\" ]; }; then safe=1; fi; "
        "if [ \"$safe\" = 1 ]; then kill -TERM -- -\"$p\" 2>/dev/null || true; sleep 15; "
        "current=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "group=0; kill -0 -- -\"$p\" 2>/dev/null && group=1 || true; safe=0; "
        "if [ -n \"$expected\" ] && [ \"$current\" = \"$expected\" ] && [ \"$group\" = 1 ]; then safe=1; "
        "elif [ \"$group\" = 1 ] && { [ ! -e \"/proc/$p/stat\" ] || [ \"$pg\" != \"$p\" ]; }; then safe=1; fi; "
        "if [ \"$safe\" = 1 ]; then kill -KILL -- -\"$p\" 2>/dev/null || true; fi; fi; fi"
    )
    script = (
        "set -eu; d=" + shlex.quote(d) + "; mkdir -p \"$d\"; "
        "rm -f \"$d/" + phase + ".pid\" \"$d/" + phase + ".start\" \"$d/" + phase + ".exit\" \"$d/" + phase + ".timeout\" \"$d/" + phase + ".watchdog.pid\"; "
        "setsid bash -lc " + shlex.quote(runner) + " >\"$d/" + phase + ".log\" 2>&1 < /dev/null & "
        "p=$!; printf '%s\\n' \"$p\" >\"$d/" + phase + ".pid\"; "
        "start=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "[ -n \"$start\" ] || { kill -TERM -- -\"$p\" 2>/dev/null || true; exit 1; }; "
        "printf '%s\\n' \"$start\" >\"$d/" + phase + ".start\"; "
        "nohup bash -c " + shlex.quote(watchdog) + " >/dev/null 2>&1 < /dev/null & "
        "printf '%s\\n' \"$!\" >\"$d/" + phase + ".watchdog.pid\"; printf 'started\\n'"
    )
    vast.run_remote(script, timeout=20)


def phase_status(vast: Vast, instance: str, phase: str, progress_file: str = "",
                 timeout: int = 15) -> tuple[bool, int | None, int | None, bool]:
    d = remote_dir(instance)
    progress = ""
    if progress_file:
        progress = (
            " if [ -r " + shlex.quote(progress_file) + " ]; then "
            "v=$(cat " + shlex.quote(progress_file) + " | head -c 24); "
            "case $v in ''|*[!0-9]*) v=0;; esac; printf 'TOKENS=%s\\n' \"$v\"; fi;"
        )
    script = (
        "d=" + shlex.quote(d) + "; p=$(cat \"$d/" + phase + ".pid\" 2>/dev/null || true); "
        "expected=$(cat \"$d/" + phase + ".start\" 2>/dev/null || true); "
        "current=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "group=0; kill -0 -- -\"$p\" 2>/dev/null && group=1 || true; alive=0; "
        "if [ -n \"$expected\" ] && [ \"$current\" = \"$expected\" ] && [ \"$group\" = 1 ]; then alive=1; "
        "elif [ \"$group\" = 1 ] && { [ ! -e \"/proc/$p/stat\" ] || [ \"$pg\" != \"$p\" ]; }; then alive=1; fi; "
        "if [ \"$alive\" = 1 ]; then echo RUNNING; else echo FINISHED; fi; "
        "if [ -r \"$d/" + phase + ".exit\" ]; then printf 'EXIT='; head -c 16 \"$d/" + phase + ".exit\"; fi; "
        "if [ -r \"$d/" + phase + ".timeout\" ]; then echo TIMEOUT; fi; " + progress
    )
    output = vast.run_remote(script, timeout=timeout)
    running = re.search(r"(?m)^RUNNING\s*$", output) is not None
    match = re.search(r"TOKENS=(\d+)", output)
    tokens = int(match.group(1)) if match else None
    match = re.search(r"EXIT=(\d+)", output)
    exit_code = int(match.group(1)) if match else None
    timed_out = "TIMEOUT" in output
    return running, tokens, exit_code, timed_out


def phase_status_retry(vast: Vast, instance: str, phase: str, progress_file: str = "",
                       timeout: int = 15, attempts: int = PHASE_STATUS_ATTEMPTS):
    """Retry transient transport/API polling failures without changing remote state."""
    error = None
    for attempt in range(attempts):
        try:
            return phase_status(vast, instance, phase, progress_file, timeout=timeout)
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 2))
    raise BudgetError(f"{phase} status polling failed after {attempts} attempts: "
                      f"{type(error).__name__}: {redact(str(error))}") from None


def retrieve_probe_tail(vast: Vast, instance: str) -> str:
    """Best-effort bounded remote stdout evidence, fetched through the selected ID."""
    d = remote_dir(instance)
    script = ("d=" + shlex.quote(d) + "; "
              "for f in probe.log probe.metrics probe_metrics.json metrics.json; do "
              "if [ -r \"$d/$f\" ]; then echo ===$f===; tail -c 4096 \"$d/$f\"; fi; done; true")
    return redact(vast.run_remote(script, timeout=10))[-6000:]


def disarm_watchdog(vast: Vast, instance: str, phase: str) -> None:
    d = remote_dir(instance)
    script = (
        "d=" + shlex.quote(d) + "; w=$(cat \"$d/" + phase + ".watchdog.pid\" 2>/dev/null || true); "
        "case $w in ''|*[!0-9]*) ;; *) kill -TERM \"$w\" 2>/dev/null || true;; esac; true"
    )
    vast.run_remote(script, timeout=15)


def stop_phase(vast: Vast, instance: str, phase: str) -> None:
    d = remote_dir(instance)
    script = (
        "d=" + shlex.quote(d) + "; p=$(cat \"$d/" + phase + ".pid\" 2>/dev/null || true); "
        "expected=$(cat \"$d/" + phase + ".start\" 2>/dev/null || true); "
        "current=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "group=0; kill -0 -- -\"$p\" 2>/dev/null && group=1 || true; safe=0; "
        "if [ -n \"$expected\" ] && [ \"$current\" = \"$expected\" ] && [ \"$group\" = 1 ]; then safe=1; "
        "elif [ \"$group\" = 1 ] && { [ ! -e \"/proc/$p/stat\" ] || [ \"$pg\" != \"$p\" ]; }; then safe=1; fi; "
        "if [ \"$safe\" = 1 ]; then kill -INT -- -\"$p\" 2>/dev/null || true; fi; true"
    )
    vast.run_remote(script, timeout=15)
    # Give the worker at least a minute to flush a checkpoint on SIGINT.
    for _ in range(60):
        running, _, _, _ = phase_status_retry(vast, instance, phase, timeout=10)
        if not running:
            disarm_watchdog(vast, instance, phase)
            return
        time.sleep(1)
    vast.run_remote(_signal_phase_script(instance, phase, "TERM"), timeout=15)
    for _ in range(15):
        running, _, _, _ = phase_status_retry(vast, instance, phase, timeout=10)
        if not running:
            disarm_watchdog(vast, instance, phase)
            return
        time.sleep(1)
    vast.run_remote(_signal_phase_script(instance, phase, "KILL"), timeout=15)
    for _ in range(5):
        running, _, _, _ = phase_status_retry(vast, instance, phase, timeout=10)
        if not running:
            disarm_watchdog(vast, instance, phase)
            return
        time.sleep(1)
    raise BudgetError(f"{phase} process did not quiesce; checkpoint pull withheld")


def _signal_phase_script(instance: str, phase: str, sig: str) -> str:
    d = remote_dir(instance)
    return (
        "d=" + shlex.quote(d) + "; p=$(cat \"$d/" + phase + ".pid\" 2>/dev/null || true); "
        "expected=$(cat \"$d/" + phase + ".start\" 2>/dev/null || true); "
        "current=$(awk '{print $22}' \"/proc/$p/stat\" 2>/dev/null || true); "
        "pg=$(ps -o pgid= -p \"$p\" 2>/dev/null | tr -d ' '); "
        "group=0; kill -0 -- -\"$p\" 2>/dev/null && group=1 || true; safe=0; "
        "if [ -n \"$expected\" ] && [ \"$current\" = \"$expected\" ] && [ \"$group\" = 1 ]; then safe=1; "
        "elif [ \"$group\" = 1 ] && { [ ! -e \"/proc/$p/stat\" ] || [ \"$pg\" != \"$p\" ]; }; then safe=1; fi; "
        "if [ \"$safe\" = 1 ]; then kill -" + sig + " -- -\"$p\" 2>/dev/null || true; fi; true"
    )


def monitor(vast: Vast, instance: str, phase: str, deadline: float, started: float,
            progress_file: str, ledger: Path, rate: float, accrued: float,
            event_details: dict | None = None) -> tuple[bool, int | None, bool]:
    last_tokens = None
    while time.monotonic() < deadline:
        remaining = max(1, int(deadline - time.monotonic()))
        running, tokens, exit_code, watchdog_timeout = phase_status_retry(
            vast, instance, phase, progress_file, timeout=min(15, remaining))
        if time.monotonic() >= deadline:
            if running:
                stop_phase(vast, instance, phase)
            else:
                disarm_watchdog(vast, instance, phase)
            append_ledger(ledger, event="stage_deadline", instance=instance, rate=rate,
                          accrued=accrued, started=started,
                          detail={"phase": phase, **(event_details or {})})
            return False, tokens, True
        if tokens is not None and tokens != last_tokens:
            last_tokens = tokens
            append_ledger(ledger, event="progress", instance=instance, rate=rate,
                          accrued=accrued, started=started,
                          detail={"phase": phase, "tokens_seen": tokens})
        if not running:
            disarm_watchdog(vast, instance, phase)
            if exit_code is None:
                raise BudgetError(f"{phase} ended without an exit status; remote log retained at {remote_dir(instance)}/{phase}.log")
            if watchdog_timeout:
                return False, tokens, True
            if exit_code != 0:
                raise BudgetError(f"{phase} exited with status {exit_code}; remote log retained at {remote_dir(instance)}/{phase}.log")
            reached = phase == "base_pretrain" and tokens is not None and tokens >= TARGET_TOKENS
            return reached, tokens, False
        if phase == "base_pretrain" and tokens is not None and tokens >= TARGET_TOKENS:
            stop_phase(vast, instance, phase)
            append_ledger(ledger, event="target_reached", instance=instance, rate=rate,
                          accrued=accrued, started=started,
                          detail={"tokens_seen": tokens, "target_tokens": TARGET_TOKENS})
            return True, tokens, False
        time.sleep(min(POLL_SECONDS, max(1, deadline - time.monotonic())))
    stop_phase(vast, instance, phase)
    append_ledger(ledger, event="stage_deadline", instance=instance, rate=rate,
                  accrued=accrued, started=started,
                  detail={"phase": phase, **(event_details or {})})
    return False, last_tokens, True


def verify_local_manifest(checkpoint: Path, timeout: int = 300) -> tuple[bool, str]:
    manifest = checkpoint / "SHA256SUMS"
    if not checkpoint.is_dir() or not manifest.is_file():
        return False, "checkpoint directory or SHA256SUMS is missing"
    if not manifest.read_text(encoding="utf-8", errors="replace").strip():
        return False, "SHA256SUMS is empty"
    try:
        result = subprocess.run(["sha256sum", "--check", "--strict", "SHA256SUMS"],
                                cwd=checkpoint, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=max(1, timeout), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"local SHA256SUMS verification could not run ({type(exc).__name__})"
    if result.returncode:
        message = redact(result.stdout + "\n" + result.stderr)
        return False, message or f"sha256sum exited {result.returncode}"
    return True, "all listed files verified"


def remaining_timeout(deadline: float, ceiling: int) -> int:
    return max(1, min(ceiling, math.floor(deadline - time.monotonic())))


def _ledger_epoch(row: dict) -> float:
    value = row.get("time_utc")
    if not isinstance(value, str):
        raise BudgetError("probe resume ledger row has no valid UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BudgetError("probe resume ledger row has no valid UTC timestamp") from None
    if parsed.tzinfo is None:
        raise BudgetError("probe resume ledger timestamp has no timezone")
    return parsed.timestamp()


def validate_probe_resume(ledger_path: Path, instance: str, resume_until: float,
                          rate: float, supplied_accrued: float,
                          now_epoch: float | None = None) -> dict:
    """Validate a handoff to an existing probe without changing the remote phase."""
    if not math.isfinite(resume_until):
        raise BudgetError("--resume-probe-until must be a finite epoch timestamp")
    now_epoch = time.time() if now_epoch is None else now_epoch
    remaining = resume_until - now_epoch
    if remaining <= 0 or remaining > PROBE_MAX_SECONDS:
        raise BudgetError("probe resume deadline must be in the future and no more than 30 minutes away")
    if not ledger_path.is_file():
        raise BudgetError("probe resume requires the existing controller ledger")
    rows = []
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        raise BudgetError("probe resume ledger is unreadable or malformed") from None
    if not rows:
        raise BudgetError("probe resume ledger is empty")
    if any(row.get("instance_id") != instance for row in rows):
        raise BudgetError("probe resume ledger contains a different instance ID")

    probe_indices = [i for i, row in enumerate(rows)
                     if row.get("event") == "phase_start"
                     and isinstance(row.get("detail"), dict)
                     and row["detail"].get("phase") == "probe"]
    if not probe_indices:
        raise BudgetError("probe resume ledger has no probe phase_start")
    probe_index = probe_indices[-1]
    probe_row = rows[probe_index]
    begin_indices = [i for i, row in enumerate(rows[:probe_index])
                     if row.get("event") == "begin"
                     and isinstance(row.get("detail"), dict)
                     and row["detail"].get("task_owned_confirmation") == "matched"]
    if not begin_indices:
        raise BudgetError("probe resume ledger has no prior matched task-owned begin")
    begin_row = rows[begin_indices[-1]]
    for row in rows[probe_index + 1:]:
        event = row.get("event")
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        if event == "finish" or (event == "phase_start" and detail.get("phase") != "probe"):
            raise BudgetError("probe resume ledger contains a later phase or finish event")
        if event == "stage_deadline" and detail.get("phase") == "probe":
            raise BudgetError("probe resume ledger records that the probe deadline already expired")
    max_seconds = probe_row["detail"].get("max_seconds")
    if type(max_seconds) is not int or not 1 <= max_seconds <= PROBE_MAX_SECONDS:
        raise BudgetError("probe resume ledger has an invalid probe duration")
    phase_epoch = _ledger_epoch(probe_row)
    exact_deadline = phase_epoch + max_seconds
    if abs(resume_until - exact_deadline) > RESUME_DEADLINE_TOLERANCE_SECONDS:
        raise BudgetError("--resume-probe-until does not match the recorded probe deadline")
    begin_epoch = _ledger_epoch(begin_row)
    try:
        original_rate = float(begin_row["hourly_rate_usd"])
        original_accrued = float(begin_row["accrued_cost_usd"])
    except (KeyError, TypeError, ValueError):
        raise BudgetError("probe resume ledger has invalid original billing details") from None
    if abs(original_rate - rate) > 0.00015:
        raise BudgetError("probe resume hourly rate differs from the original controller ledger")
    calculated_accrued = original_accrued + max(0.0, now_epoch - begin_epoch) * rate / 3600.0
    return {"deadline_epoch": exact_deadline, "remaining_seconds": remaining,
            "accrued_cost_usd": max(supplied_accrued, calculated_accrued),
            "calculated_accrued_cost_usd": calculated_accrued,
            "probe_phase_index": probe_index}


def execute(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.instance):
        raise BudgetError("instance ID must contain only letters, digits, underscore or hyphen")
    if args.confirm_task_owned_id != args.instance or args.confirm_destroy_id != args.instance:
        raise BudgetError("both ownership confirmations must exactly match --instance")
    if not Path(args.launcher).is_file():
        raise BudgetError("existing Vast launcher was not found")
    if not shutil.which("rsync"):
        raise BudgetError("local rsync is required for resumable checkpoint retrieval")
    if not Path(args.local_checkpoint).is_absolute():
        raise BudgetError("local checkpoint destination must be absolute")
    if Path(args.local_checkpoint).expanduser().exists():
        raise BudgetError("local checkpoint destination already exists; choose a new path")
    for label, remote in (("checkpoint", args.checkpoint_remote), ("progress", args.progress_file), ("SFT progress directory", args.sft_progress_dir)):
        if not remote.startswith("/") or remote == "/":
            raise BudgetError(f"{label} file/path must be a non-root absolute path")

    resume_until = getattr(args, "resume_probe_until", None)
    resume_info = None
    probe_plan_seconds = PROBE_MAX_SECONDS
    if resume_until is not None:
        ledger_for_resume = Path(args.ledger).expanduser().resolve()
        resume_info = validate_probe_resume(ledger_for_resume, args.instance, resume_until,
                                            args.hourly_rate, args.accrued_cost)
        args.accrued_cost = resume_info["accrued_cost_usd"]
        probe_plan_seconds = math.ceil(resume_info["remaining_seconds"])
    envelope = plan(args.hourly_rate, args.accrued_cost, args.teardown_reserve_usd,
                    args.sft_reserve_usd, args.acquire_max_seconds,
                    probe_seconds=probe_plan_seconds)
    started = time.monotonic()
    ledger = Path(args.ledger).expanduser().resolve()
    vast = Vast(Path(args.launcher).resolve(), args.instance)
    verify_selected_instance(vast.status(), args.instance, args.hourly_rate)
    resume_status = None
    if resume_info is not None:
        resume_status = phase_status_retry(vast, args.instance, "probe", timeout=15)
        running, _, exit_code, watchdog_timeout = resume_status
        if watchdog_timeout or not (running or exit_code == 0):
            raise BudgetError("remote probe is neither running nor successfully exited; refusing handoff")
        remaining = resume_info["deadline_epoch"] - time.time()
        if remaining <= 0 or remaining > PROBE_MAX_SECONDS:
            raise BudgetError("original probe deadline expired or is outside the 30-minute handoff window")
    append_ledger(ledger, event="begin", instance=args.instance, rate=args.hourly_rate,
                  accrued=args.accrued_cost, started=started,
                  detail={"budget": envelope, "task_owned_confirmation": "matched",
                          "probe_resumed": resume_info is not None})
    if resume_info is None:
        print(f"Selected instance verified; budget reserves ${args.sft_reserve_usd:.2f} for Kilix SFT and ${args.teardown_reserve_usd:.2f} for checkpoint/teardown.")
    else:
        append_ledger(ledger, event="phase_resume", instance=args.instance, rate=args.hourly_rate,
                      accrued=args.accrued_cost, started=started,
                      detail={"phase": "probe", "original_deadline_epoch": resume_info["deadline_epoch"],
                              "remote_running": resume_status[0], "remote_exit_code": resume_status[2]})
        print(f"Selected instance verified; reattached to the existing probe with {max(0, int(resume_info['deadline_epoch'] - time.time()))}s remaining.")

    hard_deadline = started + envelope["total_seconds_at_declared_rate"]
    work_deadline = started + envelope["work_seconds_before_teardown_reserve"]
    pre_sft_deadline = work_deadline - envelope["sft_reserve_seconds"]
    current_phase = None
    phase_active = False
    training_started = False
    token_count = None
    pretrain_target = False
    pretrain_progress = 0
    pretrain_was_controlled = False
    failure: Exception | None = None
    sft_started: list[str] = []
    sft_finished: list[str] = []
    try:
        # Probe, acquisition and base pretraining all share the pre-SFT deadline.
        if resume_info is not None:
            probe_deadline = time.monotonic() + max(0.0, resume_info["deadline_epoch"] - time.time())
            probe_seconds = max(0, math.ceil(probe_deadline - time.monotonic()))
        else:
            probe_deadline = min(pre_sft_deadline, time.monotonic() + PROBE_MAX_SECONDS)
            probe_seconds = max(0, int(probe_deadline - time.monotonic()))
        if probe_seconds:
            current_phase = "probe"
            phase_active = True
            if resume_info is None:
                append_ledger(ledger, event="phase_start", instance=args.instance, rate=args.hourly_rate,
                              accrued=args.accrued_cost, started=started,
                              detail={"phase": current_phase, "max_seconds": probe_seconds})
                phase_start(vast, args.instance, current_phase, args.probe_command, probe_seconds)
            monitor(vast, args.instance, current_phase, probe_deadline,
                    started, "", ledger, args.hourly_rate, args.accrued_cost)
            phase_active = False
            current_phase = None

        acquire_seconds = min(args.acquire_max_seconds, max(0, int(pre_sft_deadline - time.monotonic())))
        if not acquire_seconds:
            raise BudgetError("no budgeted time remains to acquire and prepare the training corpus")
        current_phase = "acquire_data"
        append_ledger(ledger, event="phase_start", instance=args.instance, rate=args.hourly_rate,
                      accrued=args.accrued_cost, started=started,
                      detail={"phase": current_phase, "max_seconds": acquire_seconds})
        phase_active = True
        phase_start(vast, args.instance, current_phase, args.acquire_command, acquire_seconds)
        _, _, acquisition_timed_out = monitor(vast, args.instance, current_phase,
            min(pre_sft_deadline, time.monotonic() + acquire_seconds), started, "",
            ledger, args.hourly_rate, args.accrued_cost)
        phase_active = False
        current_phase = None
        if acquisition_timed_out:
            raise BudgetError("data acquisition exceeded its bounded stage time; refusing to train on a partial corpus")

        pretrain_seconds = max(0, int(pre_sft_deadline - time.monotonic()))
        if not pretrain_seconds:
            raise BudgetError("no budgeted time remains for base pretraining after probe and acquisition")
        current_phase = "base_pretrain"
        append_ledger(ledger, event="phase_start", instance=args.instance, rate=args.hourly_rate,
                      accrued=args.accrued_cost, started=started,
                      detail={"phase": current_phase, "max_seconds": pretrain_seconds,
                              "target_tokens": TARGET_TOKENS})
        phase_active = True
        training_started = True
        phase_start(vast, args.instance, current_phase, args.train_command,
                    pretrain_seconds, args.progress_file, args.checkpoint_remote)
        pretrain_target, token_count, pretrain_was_controlled = monitor(vast, args.instance, current_phase,
            pre_sft_deadline, started, args.progress_file,
            ledger, args.hourly_rate, args.accrued_cost)
        phase_active = False
        pretrain_progress = token_count or 0
        current_phase = None

        # Sequential generic, Kilix-specific, then synthetic-home SFT stages share the held reserve.
        if failure is None and pretrain_progress > 0:
            sft_stages = [
                ("generic_sft", args.generic_sft_command, "generic_sft"),
                ("kilix_sft", args.kilix_sft_command, "kilix_sft"),
                ("synthetic_home", args.synthetic_home_command, "synthetic_home"),
            ]
            for index, (stage, command, progress_key) in enumerate(sft_stages):
                remaining_sft = max(0, int(work_deadline - time.monotonic()))
                if remaining_sft <= 0:
                    failure = BudgetError(f"no SFT reserve remained for {stage}")
                    break
                seconds = max(1, remaining_sft // (len(sft_stages) - index))
                stage_deadline = min(work_deadline, time.monotonic() + seconds)
                progress_file = args.sft_progress_dir.rstrip("/") + "/" + progress_key + ".progress"
                current_phase = stage
                sft_started.append(stage)
                append_ledger(ledger, event="phase_start", instance=args.instance, rate=args.hourly_rate,
                              accrued=args.accrued_cost, started=started,
                              detail={"phase": stage, "max_seconds": seconds})
                phase_active = True
                phase_start(vast, args.instance, current_phase, command, seconds,
                            progress_file, args.checkpoint_remote)
                _, _, stage_timed_out = monitor(vast, args.instance, current_phase, stage_deadline,
                    started, progress_file, ledger, args.hourly_rate, args.accrued_cost,
                    event_details={"stage": stage})
                phase_active = False
                current_phase = None
                if stage_timed_out:
                    failure = BudgetError(f"{stage} did not finish within its reserved stage window")
                    break
                sft_finished.append(stage)
    except (Exception, KeyboardInterrupt) as exc:
        failure = exc if isinstance(exc, Exception) else BudgetError("controller interrupted")
        safe_cause = redact(f"{type(failure).__name__}: {failure}")
        append_ledger(ledger, event="controller_failure", instance=args.instance,
                      rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                      detail={"phase": current_phase, "error_type": type(failure).__name__,
                              "cause": safe_cause})
        print(f"Controller failure during {current_phase or 'setup'}: {safe_cause}", file=sys.stderr)
        if current_phase and phase_active:
            if current_phase == "probe" and not training_started:
                try:
                    evidence = retrieve_probe_tail(vast, args.instance)
                    if evidence:
                        append_ledger(ledger, event="probe_failure_evidence", instance=args.instance,
                                      rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                                      detail={"tail": evidence})
                except Exception as evidence_exc:
                    append_ledger(ledger, event="probe_failure_evidence", instance=args.instance,
                                  rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                                  detail={"error_type": type(evidence_exc).__name__,
                                          "cause": redact(str(evidence_exc))})
            try:
                stop_phase(vast, args.instance, current_phase)
                phase_active = False
            except Exception:
                pass
    if failure is None and pretrain_progress <= 0:
        failure = BudgetError("base pretraining made no counted progress")
    if failure is None and len(sft_finished) != 3:
        failure = BudgetError("the complete generic/Kilix/synthetic-home SFT sequence did not finish")

    pulled = False
    verified = False
    verification_detail = "not attempted"
    local = Path(args.local_checkpoint).expanduser().resolve()
    # If a transient controller error left state uncertain, keep retrying graceful
    # quiescence while budget remains; never hash a live checkpoint tree.
    while phase_active and remaining_timeout(hard_deadline, 10_000) > 90:
        try:
            stop_phase(vast, args.instance, current_phase)
            phase_active = False
        except Exception:
            time.sleep(min(15, max(1, remaining_timeout(hard_deadline, 10_000) - 90)))
    # Never inspect or transfer the checkpoint unless the last process group is known quiescent.
    if phase_active:
        failure = failure or BudgetError("active remote process could not be confirmed quiescent; checkpoint pull withheld")
        verification_detail = "pull withheld because process-group quiescence is unconfirmed"
        append_ledger(ledger, event="checkpoint_hash_verification", instance=args.instance,
                      rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                      detail={"checkpoint_pulled": False, "verified": False, "detail": verification_detail})
    elif not training_started:
        verification_detail = "no checkpoint expected because training never started"
        append_ledger(ledger, event="checkpoint_hash_verification", instance=args.instance,
                      rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                      detail={"checkpoint_pulled": False, "verified": False, "detail": verification_detail})
    else:
        checksum = (
            "set -eu; cd " + shlex.quote(args.checkpoint_remote) + "; "
            "find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS"
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        # The $8 teardown reserve is used for resumable rsync retries and checksum checks.
        # Keep trying until only a minute remains for exact-instance destroy.
        attempt = 0
        while remaining_timeout(hard_deadline, 10_000) > 60:
            attempt += 1
            try:
                vast.run_remote(checksum, timeout=remaining_timeout(hard_deadline, 600))
                vast.sync_download(args.checkpoint_remote, local,
                                   timeout=max(1, remaining_timeout(hard_deadline, 600) - 60))
                pulled = True
                verified, verification_detail = verify_local_manifest(
                    local, timeout=max(1, remaining_timeout(hard_deadline, 300) - 60))
            except Exception as exc:
                pulled = local.is_dir()
                verified = False
                verification_detail = f"transfer/verification attempt {attempt} failed ({type(exc).__name__})"
            append_ledger(ledger, event="checkpoint_hash_verification", instance=args.instance,
                          rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                          detail={"checkpoint_pulled": pulled, "verified": verified,
                                  "attempt": attempt, "detail": verification_detail})
            if verified:
                break
            time.sleep(min(15, max(0, remaining_timeout(hard_deadline, 10_000) - 60)))
        if not verified:
            verification_detail = verification_detail + "; teardown reserve nearly exhausted"
            if attempt == 0:
                append_ledger(ledger, event="checkpoint_hash_verification", instance=args.instance,
                              rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                              detail={"checkpoint_pulled": False, "verified": False,
                                      "attempt": 0, "detail": verification_detail})
            if failure is None:
                failure = BudgetError("checkpoint transfer/hash verification failed within teardown reserve")

    destroyed = False
    if training_started and not verified and remaining_timeout(hard_deadline, 10_000) > 60:
        # Unverified trained weights stay on the rental while the bounded reserve remains.
        # This also covers the case where quiescence could not be confirmed.
        while remaining_timeout(hard_deadline, 10_000) > 60:
            time.sleep(min(15, max(1, remaining_timeout(hard_deadline, 10_000) - 60)))
    try:
        # Exact-instance destroy only. Constrain API wait to remaining hard envelope where possible.
        vast.destroy(timeout=remaining_timeout(hard_deadline, 60))
        destroyed = True
        append_ledger(ledger, event="destroyed", instance=args.instance,
                      rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                      detail={"hash_verified_before_destroy": verified})
    except Exception as exc:
        if failure is None:
            failure = exc
        append_ledger(ledger, event="destroy_failed", instance=args.instance,
                      rate=args.hourly_rate, accrued=args.accrued_cost, started=started,
                      detail={"error_type": type(exc).__name__,
                              "hash_verified_before_destroy": verified})
        print(f"Could not destroy instance {args.instance}; billing may continue. When the controller connection is restored, run: {args.launcher} --instance {args.instance} destroy --yes", file=sys.stderr)

    elapsed = time.monotonic() - started
    estimate = args.accrued_cost + args.hourly_rate * elapsed / 3600
    append_ledger(ledger, event="finish", instance=args.instance, rate=args.hourly_rate,
                  accrued=args.accrued_cost, started=started,
                  detail={"checkpoint_pulled": pulled, "checkpoint_hash_verified": verified,
                          "hash_verification_detail": verification_detail,
                          "destroyed": destroyed, "pretrain_target_reached": pretrain_target,
                          "pretrain_tokens_seen": token_count, "pretrain_target_reached": pretrain_target,
                          "pretrain_stopped_at_deadline": pretrain_was_controlled,
                          "sft_stages_started": sft_started, "sft_stages_finished": sft_finished,
                          "estimated_total_cost_usd": round(estimate, 6)})
    print(f"Ended after {elapsed:.1f}s; estimate ${estimate:.4f}/$120; pretrain_tokens={token_count}; pretrain_30B={pretrain_target}; SFT_finished={len(sft_finished)}/3; checkpoint_pulled={pulled}; SHA256SUMS_verified={verified}; destroyed={destroyed}.")
    if failure:
        raise BudgetError(f"run ended with {type(failure).__name__}: {redact(str(failure))}; see ledger and retained remote stage log if applicable") from None
    return 0 if destroyed and pulled and verified else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--instance", required=True, help="explicit existing task-owned Vast instance ID")
    common.add_argument("--hourly-rate", type=float, required=True, help="full displayed $/hour from `vast --instance ID status`")
    common.add_argument("--accrued-cost", type=float, required=True, help="current billed estimate before this script starts")
    common.add_argument("--teardown-reserve-usd", type=float, default=8.0, help="held for checkpoint pull and teardown")
    common.add_argument("--sft-reserve-usd", type=float, default=15.0, help="withheld before pretraining for three SFT stages")
    common.add_argument("--acquire-max-seconds", type=int, default=3600, help="maximum remote data acquisition/preparation stage time")
    common.add_argument("--confirm-task-owned-id", required=True, help="repeat the task-owned instance ID")
    common.add_argument("--confirm-destroy-id", required=True, help="repeat the exact instance ID authorized for teardown")
    common.add_argument("--launcher", default=str(Path.home() / "vast-ai-deployment" / "vast"))
    common.add_argument("--ledger", default="vast-budget-ledger.jsonl")
    p = commands.add_parser("plan", parents=[common], help="offline cost simulation; does not contact Vast")
    p.set_defaults(handler="plan")
    run = commands.add_parser("run", parents=[common], help="run probe, data acquisition, base pretraining, three SFT stages, checkpoint verification, and teardown on one rental")
    run.add_argument("--probe-command", required=True, help="remote probe command; output stays on the instance")
    run.add_argument("--acquire-command", required=True, help="bounded corpus acquisition/preparation command on the same instance")
    run.add_argument("--train-command", required=True, help="generic pretrainer; must write decimal tokens seen to KILIX_PROGRESS_FILE")
    run.add_argument("--progress-file", required=True, help="remote decimal pretraining-token counter")
    run.add_argument("--generic-sft-command", required=True, help="generic SFT stage command")
    run.add_argument("--kilix-sft-command", required=True, help="Kilix-specific SFT stage command")
    run.add_argument("--synthetic-home-command", required=True, help="synthetic-home SFT stage command")
    run.add_argument("--sft-progress-dir", required=True, help="remote directory for per-stage decimal progress counters")
    run.add_argument("--checkpoint-remote", required=True, help="remote checkpoint directory to retrieve")
    run.add_argument("--local-checkpoint", required=True, help="new absolute local checkpoint destination")
    run.add_argument("--resume-probe-until", type=float,
                     help="reattach to the ledgered probe until its original UTC epoch deadline")
    run.set_defaults(handler="run")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.handler == "plan":
            if args.confirm_task_owned_id != args.instance or args.confirm_destroy_id != args.instance:
                raise BudgetError("both ownership confirmations must exactly match --instance")
            envelope = plan(args.hourly_rate, args.accrued_cost,
                            args.teardown_reserve_usd, args.sft_reserve_usd,
                            args.acquire_max_seconds)
            print(json.dumps({"instance_id": args.instance, **envelope,
                              "vast_contacted": False, "resource_mutation": False}, indent=2))
            return 0
        return execute(args)
    except BudgetError as exc:
        print(f"vast-budget: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("vast-budget: interrupted; the remote watchdog stops the active stage, but controller teardown may need retry", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
