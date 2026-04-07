"""
Thread-safe shared state between the bot thread and the Textual UI.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BotState:
    """Shared state written by the bot thread, read by the UI."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Lifecycle
    phase: str = "initializing"
    cycle_count: int = 0
    error: str | None = None

    # Scan results (last cycle)
    last_scan_results: list[Any] = field(default_factory=list)
    last_scan_total: int = 0

    # Risk decisions (last cycle)
    last_risk_decisions: list[dict[str, Any]] = field(default_factory=list)
    approved_count: int = 0
    blocked_count: int = 0

    # Balance
    balance: float = 0.0

    # Log ring buffer
    log_lines: deque = field(default_factory=lambda: deque(maxlen=300))

    # --- Thread-safe setters ---

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def get_phase(self) -> str:
        with self._lock:
            return self.phase

    def increment_cycle(self) -> int:
        with self._lock:
            self.cycle_count += 1
            return self.cycle_count

    def set_balance(self, balance: float) -> None:
        with self._lock:
            self.balance = balance

    def get_balance(self) -> float:
        with self._lock:
            return self.balance

    def set_scan_results(self, candidates: list[Any], total_scanned: int = 0) -> None:
        with self._lock:
            self.last_scan_results = list(candidates)
            self.last_scan_total = total_scanned

    def get_scan_results(self) -> tuple[list[Any], int]:
        with self._lock:
            return list(self.last_scan_results), self.last_scan_total

    def set_risk_decisions(
        self, decisions: list[dict[str, Any]], approved: int, blocked: int
    ) -> None:
        with self._lock:
            self.last_risk_decisions = list(decisions)
            self.approved_count = approved
            self.blocked_count = blocked

    def get_risk_decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.last_risk_decisions)

    def set_error(self, error: str | None) -> None:
        with self._lock:
            self.error = error

    def get_error(self) -> str | None:
        with self._lock:
            return self.error

    def get_log_lines(self, last_n: int = 100) -> list[str]:
        with self._lock:
            items = list(self.log_lines)
            return items[-last_n:]

    def append_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)


class TUILogHandler(logging.Handler):
    """Logging handler that captures records into BotState for UI display."""

    def __init__(self, state: BotState) -> None:
        super().__init__()
        self._state = state

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._state.append_log(msg)
        except Exception:
            pass
