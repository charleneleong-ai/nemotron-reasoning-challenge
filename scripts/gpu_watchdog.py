"""GPU watchdog daemon: poll nvidia-smi and kill a training PID on pathological usage.

Thresholds (post a 3-min warmup grace, no kills before then):
  - hang:               util < 8%  sustained 5 min
  - wasted-compute:     util < 35% sustained 15 min
  - undersized-config:  PEAK mem (monotonic max-so-far) < 50% of total for 30 min
                        (uses peak so an eval spike that ever hit the budget saves the run)

On a fired condition it appends a structured kill_reason JSON line to the log, SIGTERMs the
PID, waits a short grace, then SIGKILLs if still alive. Polls every ~30s.

Run detached on the A100 box (survives SSH death):
    setsid nohup python -u scripts/gpu_watchdog.py --pid <PID> </dev/null \
        >>logs/gpu_watchdog.out 2>&1 & disown
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

POLL_SECONDS = 30.0
WARMUP_GRACE_S = 3 * 60
TERM_GRACE_S = 10.0

HANG_UTIL = 8.0
HANG_WINDOW_S = 5 * 60
WASTED_UTIL = 35.0
WASTED_WINDOW_S = 15 * 60
UNDERSIZED_MEM_FRAC = 0.50
UNDERSIZED_WINDOW_S = 30 * 60

DEFAULT_LOG = "logs/gpu_watchdog.jsonl"


@dataclass
class GpuSample:
    util_pct: float
    mem_used_mib: float
    mem_total_mib: float

    @property
    def mem_frac(self) -> float:
        return self.mem_used_mib / self.mem_total_mib if self.mem_total_mib > 0 else 0.0


def query_gpu(gpu_index: int) -> GpuSample | None:
    """Return a GpuSample, or None if nvidia-smi is unavailable / unparseable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        _emit_raw({"event": "nvidia_smi_error", "error": str(exc)})
        return None

    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        _emit_raw({"event": "parse_error", "raw": line})
        return None
    try:
        return GpuSample(float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        _emit_raw({"event": "parse_error", "raw": line})
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _emit_raw(record: dict[str, object]) -> None:
    record = {"ts": time.time(), **record}
    print(json.dumps(record), flush=True)


@dataclass
class _Window:
    """Tracks when a sustained condition first became true; fires once the window elapses."""

    window_s: float
    start: float | None = None

    def update(self, condition: bool, now: float) -> bool:
        if condition:
            if self.start is None:
                self.start = now
            return now - self.start >= self.window_s
        self.start = None
        return False


@dataclass
class Watchdog:
    pid: int
    log_path: Path
    gpu_index: int = 0
    started_at: float = field(default_factory=time.monotonic)
    peak_mem_frac: float = 0.0
    _hang: _Window = field(init=False)
    _wasted: _Window = field(init=False)
    _undersized: _Window = field(init=False)

    def __post_init__(self) -> None:
        self._hang = _Window(HANG_WINDOW_S)
        self._wasted = _Window(WASTED_WINDOW_S)
        self._undersized = _Window(UNDERSIZED_WINDOW_S)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, record: dict[str, object]) -> None:
        record = {"ts": time.time(), "pid": self.pid, **record}
        line = json.dumps(record)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")
            fh.flush()
        print(line, flush=True)

    def _check(self, sample: GpuSample, now: float) -> str | None:
        """Return a kill_reason string if a threshold fired, else None."""
        self.peak_mem_frac = max(self.peak_mem_frac, sample.mem_frac)

        if self._hang.update(sample.util_pct < HANG_UTIL, now):
            return "hang"
        if self._wasted.update(sample.util_pct < WASTED_UTIL, now):
            return "wasted_compute"
        # Undersized only fires if the PEAK has never reached the budget — an eval
        # spike that ever crossed UNDERSIZED_MEM_FRAC permanently clears this.
        if self._undersized.update(self.peak_mem_frac < UNDERSIZED_MEM_FRAC, now):
            return "undersized_config"
        return None

    def _kill(self, reason: str, sample: GpuSample) -> None:
        self._log(
            {
                "event": "kill",
                "kill_reason": reason,
                "util_pct": sample.util_pct,
                "mem_frac": round(sample.mem_frac, 4),
                "peak_mem_frac": round(self.peak_mem_frac, 4),
            }
        )
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            self._log({"event": "already_dead", "kill_reason": reason})
            return

        deadline = time.monotonic() + TERM_GRACE_S
        while time.monotonic() < deadline:
            if not pid_alive(self.pid):
                self._log({"event": "terminated", "kill_reason": reason})
                return
            time.sleep(0.5)

        try:
            os.kill(self.pid, signal.SIGKILL)
            self._log({"event": "sigkill", "kill_reason": reason})
        except ProcessLookupError:
            self._log({"event": "terminated", "kill_reason": reason})

    def run(self) -> int:
        self._log(
            {
                "event": "start",
                "gpu_index": self.gpu_index,
                "warmup_grace_s": WARMUP_GRACE_S,
                "poll_s": POLL_SECONDS,
            }
        )
        while True:
            if not pid_alive(self.pid):
                self._log({"event": "target_exited"})
                return 0

            sample = query_gpu(self.gpu_index)
            now = time.monotonic()
            if sample is None:
                time.sleep(POLL_SECONDS)
                continue

            if now - self.started_at < WARMUP_GRACE_S:
                # Still observe peak mem during warmup, but never kill.
                self.peak_mem_frac = max(self.peak_mem_frac, sample.mem_frac)
                time.sleep(POLL_SECONDS)
                continue

            reason = self._check(sample, now)
            if reason is not None:
                self._kill(reason, sample)
                return 1
            time.sleep(POLL_SECONDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU watchdog for a training PID.")
    parser.add_argument("--pid", type=int, required=True, help="Training PID to watch.")
    parser.add_argument(
        "--log", type=str, default=DEFAULT_LOG, help="JSONL kill-reason log path."
    )
    parser.add_argument(
        "--gpu-index", type=int, default=0, help="nvidia-smi GPU index to poll."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    watchdog = Watchdog(pid=args.pid, log_path=Path(args.log), gpu_index=args.gpu_index)
    return watchdog.run()


if __name__ == "__main__":
    sys.exit(main())
