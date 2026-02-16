"""Event-loop watchdog: detects stuck ACP processes and auto-captures diagnostics.

Design:
- A *daemon thread* (not asyncio task) checks a heartbeat timestamp.
- The asyncio event loop bumps the heartbeat every tick via a periodic coroutine.
- If the heartbeat goes stale (event loop stuck), the thread:
  1. Dumps all-thread Python traceback via faulthandler
  2. Logs the incident
  3. Kills the process with SIGKILL (clean exit is unlikely if the loop is stuck)

Why a thread?  If the event loop itself is the thing spinning, an asyncio.Task
would never get scheduled.  A separate OS thread is the only reliable observer.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import threading
import time
from io import TextIOWrapper
from pathlib import Path

log = logging.getLogger(__name__)

# Defaults — caller can override via constructor.
_DEFAULT_HEARTBEAT_INTERVAL_S = 5  # how often the event loop should ping
_DEFAULT_STALE_THRESHOLD_S = 60  # how long before we declare "stuck"
_DEFAULT_CHECK_INTERVAL_S = 10  # how often the watchdog thread checks


class Watchdog:
    """Daemon-thread watchdog that kills the process if the event loop stops responding."""

    def __init__(
        self,
        *,
        fault_file: TextIOWrapper | None = None,
        stale_threshold_s: float = _DEFAULT_STALE_THRESHOLD_S,
        check_interval_s: float = _DEFAULT_CHECK_INTERVAL_S,
    ) -> None:
        self._fault_file = fault_file
        self._stale_threshold_s = stale_threshold_s
        self._check_interval_s = check_interval_s
        self._last_heartbeat: float = time.monotonic()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def heartbeat(self) -> None:
        """Called from the asyncio event loop to signal liveness."""
        with self._lock:
            self._last_heartbeat = time.monotonic()

    def start(self) -> None:
        """Start the watchdog daemon thread."""
        self._stopped.clear()
        self._last_heartbeat = time.monotonic()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="acp-watchdog",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Watchdog started (stale_threshold=%.0fs, check_interval=%.0fs)",
            self._stale_threshold_s,
            self._check_interval_s,
        )

    def stop(self) -> None:
        """Stop the watchdog thread gracefully."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _watch_loop(self) -> None:
        """Runs in a daemon thread.  Checks heartbeat staleness."""
        while not self._stopped.wait(timeout=self._check_interval_s):
            with self._lock:
                age = time.monotonic() - self._last_heartbeat

            if age > self._stale_threshold_s:
                self._on_stuck(age)
                # Don't loop — we're about to die.
                return

    def _on_stuck(self, age: float) -> None:
        """Called when heartbeat is stale: capture diagnostics, then kill."""
        pid = os.getpid()
        log.error(
            "WATCHDOG: Event loop unresponsive for %.0fs (threshold %.0fs). "
            "PID %d — capturing diagnostics and terminating.",
            age,
            self._stale_threshold_s,
            pid,
        )

        # 1. Dump Python tracebacks for all threads via faulthandler.
        #    Write to the same fault_file used by SIGUSR1 handler so
        #    capture-stuck-acp.sh can find it too.
        try:
            if self._fault_file and not self._fault_file.closed:
                self._fault_file.write(
                    f"\n=== WATCHDOG AUTO-CAPTURE (stale {age:.0f}s) "
                    f"pid={pid} ===\n"
                )
                self._fault_file.flush()
                faulthandler.dump_traceback(file=self._fault_file, all_threads=True)
                self._fault_file.flush()
                log.error(
                    "WATCHDOG: Traceback written to %s", self._fault_file.name
                )
            else:
                # Fallback: dump to a new file
                fallback = self._make_fallback_fault_file(pid)
                if fallback:
                    fallback.write(
                        f"\n=== WATCHDOG AUTO-CAPTURE (stale {age:.0f}s) "
                        f"pid={pid} ===\n"
                    )
                    fallback.flush()
                    faulthandler.dump_traceback(file=fallback, all_threads=True)
                    fallback.flush()
                    fallback.close()
                    log.error("WATCHDOG: Traceback written to fallback file")
        except Exception:
            log.exception("WATCHDOG: Failed to dump traceback")

        # 2. Kill self. SIGKILL because if the event loop is stuck,
        #    SIGTERM/SIGINT may not be handled.
        log.error("WATCHDOG: Sending SIGKILL to self (PID %d)", pid)
        os.kill(pid, signal.SIGKILL)

    @staticmethod
    def _make_fallback_fault_file(pid: int) -> TextIOWrapper | None:
        """Create a fallback fault file if the primary one is unavailable."""
        try:
            fault_dir = (
                Path(
                    os.environ.get(
                        "XDG_STATE_HOME", Path.home() / ".local" / "state"
                    )
                )
                / "pykoclaw"
            )
            fault_dir.mkdir(parents=True, exist_ok=True)
            return (fault_dir / f"watchdog-{pid}.txt").open("w")
        except Exception:
            log.exception("WATCHDOG: Cannot create fallback fault file")
            return None
