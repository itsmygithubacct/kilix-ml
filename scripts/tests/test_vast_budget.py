"""Offline simulation coverage for vast_budget.py; never contacts Vast."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "vast_budget.py"
spec = importlib.util.spec_from_file_location("vast_budget", SCRIPT)
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)


class VastBudgetTests(unittest.TestCase):
    def test_phase_status_retries_transient_transport_errors_three_times(self):
        good = (True, 10, None, False)
        with patch.object(vb, "phase_status", side_effect=[OSError("network flap"),
                                                             vb.BudgetError("ssh reset"), good]), \
             patch.object(vb.time, "sleep") as sleep:
            self.assertEqual(vb.phase_status_retry(object(), "12345", "probe"), good)
        self.assertEqual(sleep.call_count, 2)

    def test_phase_status_retry_surfaces_sanitized_original_failure(self):
        with patch.object(vb, "phase_status", side_effect=vb.BudgetError("API token=hf_12345678901234567890 unavailable")), \
             patch.object(vb.time, "sleep") as sleep:
            with self.assertRaisesRegex(vb.BudgetError, "failed after 3 attempts") as caught:
                vb.phase_status_retry(object(), "12345", "probe")
        self.assertIn("API token=[REDACTED]", str(caught.exception))
        self.assertEqual(sleep.call_count, 2)

    def test_probe_failure_tail_is_guarded_and_redacted(self):
        class Remote:
            instance = None
            script = None
            def run_remote(self, script, timeout=10):
                self.script = script
                self.timeout = timeout
                return "traceback\napi_key=abc123\nlast metrics"
        remote = Remote()
        result = vb.retrieve_probe_tail(remote, "12345")
        self.assertIn("api_key=[REDACTED]", result)
        self.assertIn("probe.log", remote.script)
        self.assertEqual(remote.timeout, 10)

    @staticmethod
    def _probe_ledger(path, instance="12345", *, now=2_000_000_000.0,
                      phase="probe", later_phase=None):
        stamp = lambda value: vb.dt.datetime.fromtimestamp(
            value, vb.dt.timezone.utc).isoformat()
        rows = [
            {"time_utc": stamp(now - 1200), "event": "begin", "instance_id": instance,
             "hourly_rate_usd": 1.25, "accrued_cost_usd": 5.0,
             "detail": {"task_owned_confirmation": "matched"}},
            {"time_utc": stamp(now - 1000), "event": "phase_start", "instance_id": instance,
             "hourly_rate_usd": 1.25, "accrued_cost_usd": 5.0,
             "detail": {"phase": phase, "max_seconds": 1800}},
        ]
        if later_phase:
            rows.append({"time_utc": stamp(now - 900), "event": "phase_start",
                         "instance_id": instance, "hourly_rate_usd": 1.25,
                         "accrued_cost_usd": 5.0,
                         "detail": {"phase": later_phase, "max_seconds": 600}})
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return now - 1000 + 1800

    def test_probe_resume_validates_ledger_deadline_and_recalculates_accrual(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"
            now = 2_000_000_000.0
            until = self._probe_ledger(ledger, now=now)
            result = vb.validate_probe_resume(ledger, "12345", until, 1.25, 4.0,
                                              now_epoch=now)
            self.assertEqual(result["deadline_epoch"], until)
            self.assertEqual(result["remaining_seconds"], 800)
            self.assertGreater(result["accrued_cost_usd"], 5.0)
            with self.assertRaisesRegex(vb.BudgetError, "does not match"):
                vb.validate_probe_resume(ledger, "12345", until + 10, 1.25, 5.0,
                                          now_epoch=now)
            with self.assertRaisesRegex(vb.BudgetError, "future"):
                vb.validate_probe_resume(ledger, "12345", until, 1.25, 5.0,
                                          now_epoch=until + 1)

    def test_probe_resume_rejects_instance_phase_and_later_stage_mismatches(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"
            now = 2_000_000_000.0
            until = self._probe_ledger(ledger, now=now, phase="acquire_data")
            with self.assertRaisesRegex(vb.BudgetError, "no probe phase_start"):
                vb.validate_probe_resume(ledger, "12345", until, 1.25, 5.0, now_epoch=now)
            until = self._probe_ledger(ledger, now=now, later_phase="acquire_data")
            with self.assertRaisesRegex(vb.BudgetError, "later phase or finish"):
                vb.validate_probe_resume(ledger, "12345", until, 1.25, 5.0, now_epoch=now)
            until = self._probe_ledger(ledger, instance="99999", now=now)
            with self.assertRaisesRegex(vb.BudgetError, "different instance ID"):
                vb.validate_probe_resume(ledger, "12345", until, 1.25, 5.0, now_epoch=now)

    def test_resume_probe_reattaches_without_relaunch_and_checks_status_before_begin(self):
        status = "ID           STATUS       GPU                      $/HR  LABEL\n12345        running      RTX 4090                 1.2500  kilix ml\n"

        class SimulatedVast:
            events = []
            def __init__(self, launcher, instance):
                self.instance = instance
                SimulatedVast.events = []
            def status(self): return status
            def run_remote(self, script, timeout=180): self.events.append("remote-checksum"); return ""
            def sync_download(self, source, destination, timeout):
                destination.mkdir(parents=True, exist_ok=True)
                weight = destination / "weights.bin"
                weight.write_bytes(b"resumed checkpoint")
                digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                (destination / "SHA256SUMS").write_text(digest + "  ./weights.bin\n")
                self.events.append("rsync")
            def destroy(self, timeout=60): self.events.append("destroy")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launcher = root / "vast"
            launcher.write_text("mock launcher")
            ledger = root / "ledger.jsonl"
            until = self._probe_ledger(ledger, now=time.time())
            events = []
            phase_calls = []
            original_append = vb.append_ledger

            def append_spy(path, **kwargs):
                events.append("ledger-" + kwargs["event"])
                return original_append(path, **kwargs)

            def phase_status_spy(*args, **kwargs):
                events.append("status-probe")
                return True, None, None, False

            def start_spy(vast, instance, phase, command, seconds, progress_file="", checkpoint_dir=""):
                phase_calls.append(phase)

            def monitor_spy(vast, instance, phase, *args, **kwargs):
                if phase == "base_pretrain": return False, 9_000_000_000, True
                return False, None, False

            args = type("Args", (), {
                "instance": "12345", "confirm_task_owned_id": "12345", "confirm_destroy_id": "12345",
                "launcher": str(launcher), "local_checkpoint": str(root / "pulled"),
                "checkpoint_remote": "/workspace/checkpoint", "progress_file": "/workspace/tokens",
                "sft_progress_dir": "/workspace/sft-progress", "hourly_rate": 1.25,
                "accrued_cost": 5.0, "teardown_reserve_usd": 8.0, "sft_reserve_usd": 15.0,
                "acquire_max_seconds": 3600, "ledger": str(ledger),
                "probe_command": "probe", "acquire_command": "acquire", "train_command": "pretrain",
                "generic_sft_command": "generic", "kilix_sft_command": "kilix",
                "synthetic_home_command": "synthetic", "resume_probe_until": until,
            })()
            with patch.object(vb, "Vast", SimulatedVast), patch.object(vb, "phase_status", phase_status_spy), \
                 patch.object(vb, "phase_start", start_spy), patch.object(vb, "monitor", monitor_spy), \
                 patch.object(vb, "append_ledger", side_effect=append_spy), \
                 patch.object(vb.shutil, "which", return_value="/usr/bin/rsync"), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(vb.execute(args), 0)
            self.assertNotIn("probe", phase_calls)
            self.assertEqual(phase_calls[0], "acquire_data")
            self.assertLess(events.index("status-probe"), events.index("ledger-begin"))
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertTrue(any(row["event"] == "phase_resume" for row in records))
            latest_begin = [row for row in records if row["event"] == "begin"][-1]
            self.assertGreater(latest_begin["accrued_cost_usd"], 5.0)

    def test_cost_budget_reserves_sft_acquisition_and_teardown(self):
        result = vb.plan(1.25, 20, 8, 15, 3600)
        self.assertEqual(result["work_seconds_before_teardown_reserve"], 264960)
        self.assertEqual(result["sft_reserve_seconds"], 43200)
        self.assertEqual(result["pretrain_and_probe_seconds_before_sft_reserve"], 221760)
        self.assertEqual(result["planned_pretrain_seconds_after_full_probe_and_acquisition"], 216360)
        self.assertEqual(result["total_seconds_at_declared_rate"], 288000)
        for values in ((0, 0, 8, 15, 3600), (1, 120, 8, 15, 3600),
                       (1, 0, 8, 113, 3600), (1, 0, 8, 15, -1),
                       (300, 0, 8, 15, 3600)):
            with self.assertRaises(vb.BudgetError):
                vb.plan(*values)

    def test_status_requires_selected_id_and_full_displayed_rate(self):
        text = "ID           STATUS       GPU                      $/HR  LABEL\n12345        running      RTX 4090                 1.2500  kilix ml\n"
        self.assertEqual(vb.verify_selected_instance(text, "12345", 1.25), 1.25)
        with self.assertRaises(vb.BudgetError):
            vb.verify_selected_instance(text, "99999", 1.25)
        with self.assertRaises(vb.BudgetError):
            vb.verify_selected_instance(text, "12345", 0.25)

    def test_single_id_destroy_arguments_only(self):
        client = vb.Vast.__new__(vb.Vast)
        client.instance = "12345"
        client.launcher = Path("/fake/vast")
        calls = []
        client.call = lambda *args, **kwargs: calls.append((client.instance, *args)) or ""
        client.destroy()
        self.assertEqual(calls, [("12345", "destroy", "--yes")])
        self.assertNotIn("--all", calls[0])

    def test_resumable_pull_uses_exact_instance_ssh_and_partial_append(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            launcher = root / "vast"
            launcher.write_text("def load_config(): return {}\ndef ssh_options(config, instance):\n    assert instance == '12345'\n    return ['-i', '/key'], 'root@host'\n")
            client = vb.Vast(launcher, "12345")
            with patch.object(vb.subprocess, "run", return_value=type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()) as run:
                client.sync_download("/workspace/checkpoint", root / "pulled", 45)
            command = run.call_args.args[0]
            self.assertEqual(command[0], "rsync")
            self.assertIn("--partial", command)
            self.assertIn("--append-verify", command)
            self.assertIn("root@host:/workspace/checkpoint/", command)
            self.assertEqual(run.call_args.kwargs["timeout"], 45)

    def test_remote_phase_includes_deadline_and_pid_identity_watchdog(self):
        class Capture:
            script = ""
            def run_remote(self, script, timeout=20):
                self.script = script
                return "started\n"
        client = Capture()
        vb.phase_start(client, "12345", "base_pretrain", "python train.py", 900,
                       "/workspace/tokens_seen", "/workspace/checkpoint")
        self.assertIn("KILIX_STAGE_SECONDS=900", client.script)
        self.assertIn("KILIX_TARGET_TOKENS=30000000000", client.script)
        self.assertIn("KILIX_CHECKPOINT_DIR=/workspace/checkpoint", client.script)
        self.assertIn(".start", client.script)
        self.assertIn(".watchdog.pid", client.script)
        self.assertIn(".timeout", client.script)
        self.assertIn(".exit", client.script)
        self.assertIn("rc=$?", client.script)
        self.assertIn("base_pretrain.exit", client.script)
        self.assertIn("ps -o pgid=", client.script)
        self.assertIn("setsid", client.script)

    def test_stop_phase_waits_sixty_seconds_before_escalating(self):
        class FakeRemote:
            scripts = []
            def run_remote(self, script, timeout=15):
                self.scripts.append(script)
                return ""
        remote = FakeRemote()
        statuses = [(True, None, None, False)] * 59 + [(False, None, 0, False)]
        with patch.object(vb, "phase_status", side_effect=statuses), \
             patch.object(vb, "disarm_watchdog") as disarm, patch.object(vb.time, "sleep") as sleep:
            vb.stop_phase(remote, "12345", "base_pretrain")
        self.assertEqual(sleep.call_count, 59)
        self.assertIn("kill -INT", remote.scripts[0])
        self.assertEqual(len(remote.scripts), 1)  # no TERM/KILL after graceful exit
        disarm.assert_called_once_with(remote, "12345", "base_pretrain")

    def test_nonzero_early_stage_exit_is_an_error(self):
        class FailedRemote:
            def run_remote(self, script, timeout=15):
                return "FINISHED\nEXIT=17\n"
        with self.assertRaisesRegex(vb.BudgetError, "status 17"):
            vb.monitor(FailedRemote(), "12345", "base_pretrain",
                       time.monotonic() + 2, time.monotonic(), "",
                       Path("unused-ledger"), 1.0, 0.0)

    def test_base_pretrain_target_detection_uses_actual_stage_name(self):
        class FinishedRemote:
            def run_remote(self, script, timeout=15):
                return "FINISHED\nEXIT=0\nTOKENS=30000000000\n"
        with patch.object(vb, "phase_status", return_value=(False, 30_000_000_000, 0, False)), \
             patch.object(vb, "disarm_watchdog"):
            result = vb.monitor(FinishedRemote(), "12345", "base_pretrain",
                                time.monotonic() + 2, time.monotonic(), "/tokens",
                                Path("unused-ledger"), 1.0, 0.0)
        self.assertEqual(result, (True, 30_000_000_000, False))

    def test_deadline_stops_phase_and_records_timeout(self):
        stopped = []
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"
            with patch.object(vb, "phase_status", return_value=(True, 5, None, False)), \
                 patch.object(vb, "stop_phase", side_effect=lambda _v, _i, stage: stopped.append(stage)):
                result = vb.monitor(object(), "12345", "base_pretrain",
                                    time.monotonic() + 0.02, time.monotonic(), "",
                                    ledger, 1.0, 0.0)
            records = [json.loads(line) for line in ledger.read_text().splitlines()]
            self.assertEqual(result, (False, 5, True))
            self.assertEqual(stopped, ["base_pretrain"])
            self.assertEqual(records[-1]["event"], "stage_deadline")

    def test_sha256_manifest_passes_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoint"
            root.mkdir()
            weight = root / "weights.bin"
            weight.write_bytes(b"model weights")
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            (root / "SHA256SUMS").write_text(f"{digest}  ./weights.bin\n")
            self.assertTrue(vb.verify_local_manifest(root)[0])
            weight.write_bytes(b"changed")
            valid, detail = vb.verify_local_manifest(root)
            self.assertFalse(valid)
            self.assertTrue(detail)

    def test_plan_mode_is_explicitly_non_mutating(self):
        args = ["plan", "--instance", "SIMULATED-4", "--hourly-rate", "1.25",
                "--accrued-cost", "20", "--confirm-task-owned-id", "SIMULATED-4",
                "--confirm-destroy-id", "SIMULATED-4", "--launcher", "/no/such/launcher"]
        with patch("sys.argv", [str(SCRIPT), *args]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(vb.main(), 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["vast_contacted"])
        self.assertFalse(payload["resource_mutation"])

    def test_limited_base_pretraining_flows_through_three_sft_stages(self):
        status = "ID           STATUS       GPU                      $/HR  LABEL\n12345        running      RTX 4090                 1.2500  kilix ml\n"

        class SimulatedVast:
            events = []
            def __init__(self, launcher, instance):
                self.instance = instance
                SimulatedVast.events = []
            def status(self):
                return status
            def run_remote(self, script, timeout=180):
                self.events.append("remote-checksum")
                return ""
            def sync_download(self, source, destination, timeout):
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "weights.bin").write_bytes(b"checkpoint")
                digest = hashlib.sha256((destination / "weights.bin").read_bytes()).hexdigest()
                (destination / "SHA256SUMS").write_text(digest + "  ./weights.bin\n")
                self.events.append("rsync-resumable")
            def download(self, source, destination, timeout):
                destination = Path(destination)
                if source.endswith("/SHA256SUMS"):
                    digest = hashlib.sha256((destination.parent / "weights.bin").read_bytes()).hexdigest()
                    (destination.parent / "SHA256SUMS").write_text(digest + "  ./weights.bin\n")
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / "weights.bin").write_bytes(b"checkpoint")
                self.events.append("download")
            def destroy(self, timeout=60):
                self.events.append("destroy-exact-id")

        def fake_start(vast, instance, stage, command, seconds, progress_file="", checkpoint_dir=""):
            SimulatedVast.events.append("start-" + stage)
            if stage in {"base_pretrain", "generic_sft", "kilix_sft", "synthetic_home"}:
                assert checkpoint_dir == "/workspace/checkpoint"

        def fake_monitor(vast, instance, stage, deadline, started, progress, ledger, rate, accrued,
                         event_details=None):
            if stage == "base_pretrain":
                return False, 9_000_000_000, True  # stopped short at its budget boundary
            return False, None, False

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = type("Args", (), {
                "instance": "12345", "confirm_task_owned_id": "12345", "confirm_destroy_id": "12345",
                "launcher": "/usr/bin/true", "local_checkpoint": str(root / "pulled"),
                "checkpoint_remote": "/workspace/checkpoint", "progress_file": "/workspace/tokens",
                "sft_progress_dir": "/workspace/sft-progress", "hourly_rate": 1.25,
                "accrued_cost": 0.0, "teardown_reserve_usd": 8.0, "sft_reserve_usd": 15.0,
                "acquire_max_seconds": 3600, "ledger": str(root / "ledger.jsonl"),
                "probe_command": "probe", "acquire_command": "acquire", "train_command": "pretrain",
                "generic_sft_command": "generic", "kilix_sft_command": "kilix",
                "synthetic_home_command": "synthetic",
            })()
            with patch.object(vb, "Vast", SimulatedVast), patch.object(vb, "phase_start", fake_start), \
                 patch.object(vb, "monitor", fake_monitor), contextlib.redirect_stdout(io.StringIO()):
                result = vb.execute(args)
            self.assertEqual(result, 0)
            starts = [event for event in SimulatedVast.events if event.startswith("start-")]
            self.assertEqual(starts, ["start-probe", "start-acquire_data", "start-base_pretrain",
                                      "start-generic_sft", "start-kilix_sft", "start-synthetic_home"])
            records = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines()]
            verification = max(i for i, row in enumerate(records) if row["event"] == "checkpoint_hash_verification")
            destroy = next(i for i, row in enumerate(records) if row["event"] == "destroyed")
            self.assertTrue(records[verification]["detail"]["verified"])
            self.assertLess(verification, destroy)
            self.assertTrue(records[-1]["detail"]["pretrain_stopped_at_deadline"])
            self.assertEqual(SimulatedVast.events.count("rsync-resumable"), 1)

    def test_retry_rsync_hash_before_destroy_and_hold_unverified_until_cap(self):
        status = "ID           STATUS       GPU                      $/HR  LABEL\n12345        running      RTX 4090                 1.2500  kilix ml\n"
        class RetryVast:
            events = []
            def __init__(self, launcher, instance):
                self.instance = instance
                RetryVast.events = []
            def status(self): return status
            def run_remote(self, script, timeout=180): self.events.append("remote-checksum"); return ""
            def sync_download(self, source, destination, timeout):
                destination.mkdir(parents=True, exist_ok=True)
                weight = destination / "weights.bin"
                weight.write_bytes(b"good checkpoint")
                digest = hashlib.sha256(weight.read_bytes()).hexdigest()
                (destination / "SHA256SUMS").write_text(digest + "  ./weights.bin\n")
                self.events.append("rsync")
            def destroy(self, timeout=60): self.events.append("destroy")

        def fake_start(*args, **kwargs): pass
        def fake_monitor(vast, instance, stage, *args, **kwargs):
            if stage == "base_pretrain": return False, 9_000_000_000, True
            return False, None, False
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = type("Args", (), {
                "instance": "12345", "confirm_task_owned_id": "12345", "confirm_destroy_id": "12345",
                "launcher": "/usr/bin/true", "local_checkpoint": str(root / "pulled"),
                "checkpoint_remote": "/workspace/checkpoint", "progress_file": "/workspace/tokens",
                "sft_progress_dir": "/workspace/sft-progress", "hourly_rate": 1.25,
                "accrued_cost": 0.0, "teardown_reserve_usd": 8.0, "sft_reserve_usd": 15.0,
                "acquire_max_seconds": 3600, "ledger": str(root / "ledger.jsonl"),
                "probe_command": "probe", "acquire_command": "acquire", "train_command": "pretrain",
                "generic_sft_command": "generic", "kilix_sft_command": "kilix",
                "synthetic_home_command": "synthetic", "sft_progress_dir": "/workspace/sft-progress",
            })()
            with patch.object(vb, "Vast", RetryVast), patch.object(vb, "phase_start", fake_start), \
                 patch.object(vb, "monitor", fake_monitor), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(vb.time, "sleep"):
                self.assertEqual(vb.execute(args), 0)
            self.assertLess(RetryVast.events.index("rsync"), RetryVast.events.index("destroy"))

        # With only the last minute of the hard deadline available, do not start a pull;
        # record an unverified checkpoint and destroy the selected instance at the cap.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args.ledger = str(root / "ledger.jsonl")
            args.local_checkpoint = str(root / "not-pulled")
            with patch.object(vb, "Vast", RetryVast), patch.object(vb, "phase_start", fake_start), \
                 patch.object(vb, "monitor", fake_monitor), patch.object(vb, "remaining_timeout", return_value=60), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(vb.BudgetError):
                    vb.execute(args)
            self.assertNotIn("rsync", RetryVast.events)
            self.assertIn("destroy", RetryVast.events)
            records = [json.loads(line) for line in Path(args.ledger).read_text().splitlines()]
            verification = next(row for row in records if row["event"] == "checkpoint_hash_verification")
            destroyed = next(row for row in records if row["event"] == "destroyed")
            self.assertFalse(verification["detail"]["verified"])
            self.assertLess(records.index(verification), records.index(destroyed))

    def test_unconfirmed_quiescence_withholds_checkpoint_pull(self):
        status = "ID           STATUS       GPU                      $/HR  LABEL\n12345        running      RTX 4090                 1.2500  kilix ml\n"
        class UncertainVast:
            events = []
            def __init__(self, launcher, instance): self.instance = instance; UncertainVast.events = []
            def status(self): return status
            def sync_download(self, source, destination, timeout): self.events.append("pull")
            def destroy(self, timeout=60): self.events.append("destroy")
        def fake_monitor(vast, instance, stage, *args, **kwargs):
            if stage == "base_pretrain": raise vb.BudgetError("simulated controller loss")
            return False, None, False
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = type("Args", (), {
                "instance": "12345", "confirm_task_owned_id": "12345", "confirm_destroy_id": "12345",
                "launcher": "/usr/bin/true", "local_checkpoint": str(root / "pulled"),
                "checkpoint_remote": "/workspace/checkpoint", "progress_file": "/workspace/tokens",
                "sft_progress_dir": "/workspace/sft-progress", "hourly_rate": 1.25,
                "accrued_cost": 0.0, "teardown_reserve_usd": 8.0, "sft_reserve_usd": 15.0,
                "acquire_max_seconds": 3600, "ledger": str(root / "ledger.jsonl"),
                "probe_command": "probe", "acquire_command": "acquire", "train_command": "pretrain",
                "generic_sft_command": "generic", "kilix_sft_command": "kilix",
                "synthetic_home_command": "synthetic",
            })()
            with patch.object(vb, "Vast", UncertainVast), patch.object(vb, "phase_start"), \
                 patch.object(vb, "monitor", fake_monitor), patch.object(vb, "stop_phase", side_effect=vb.BudgetError("still active")), \
                 patch.object(vb, "remaining_timeout", return_value=60), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(vb.BudgetError): vb.execute(args)
            self.assertNotIn("pull", UncertainVast.events)
            self.assertIn("destroy", UncertainVast.events)
            records = [json.loads(line) for line in Path(args.ledger).read_text().splitlines()]
            verification = next(row for row in records if row["event"] == "checkpoint_hash_verification")
            self.assertIn("quiescence is unconfirmed", verification["detail"]["detail"])


if __name__ == "__main__":
    unittest.main()
