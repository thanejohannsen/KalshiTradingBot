"""
Shared utility helpers — logging setup, timing, data classes.
"""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any


_DEFER_PM_CONSOLE = False
_DEFER_PM_LOCK = threading.Lock()
_DEFER_PM_BUFFER: list[str] = []


def set_position_monitor_console_deferred(active: bool) -> None:
    global _DEFER_PM_CONSOLE
    with _DEFER_PM_LOCK:
        _DEFER_PM_CONSOLE = active


def drain_position_monitor_console_buffer() -> list[str]:
    with _DEFER_PM_LOCK:
        items = list(_DEFER_PM_BUFFER)
        _DEFER_PM_BUFFER.clear()
        return items


class _ConsoleNoiseFilter(logging.Filter):
    """Hide noisy scan/research logs from terminal only."""

    BLOCKED_PREFIXES = (
        "kalshi_bot.scan_agent",
        "kalshi_bot.research_agent",
        "kalshi_bot.news",
        "kalshi_bot.twitter",
        "kalshi_bot.scraper",
        "kalshi_bot.sentiment",
        "kalshi_bot.claude",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Optionally defer position-monitor console logs while research progress
        # is actively rendering to avoid breaking the one-line progress bar.
        if record.name.startswith("kalshi_bot.position_monitor"):
            with _DEFER_PM_LOCK:
                if _DEFER_PM_CONSOLE:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    msg = record.getMessage()
                    _DEFER_PM_BUFFER.append(
                        f"{ts} | {record.levelname:<8s} | {record.name} | {msg}"
                    )
                    return False

        if any(record.name.startswith(p) for p in self.BLOCKED_PREFIXES):
            return False

        # Hide market-fetch chatter in terminal; keep warnings/errors visible.
        if record.name.startswith("kalshi_bot.kalshi_client") and record.levelno < logging.WARNING:
            return False

        return True


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    quiet_research_console: bool = True,
) -> logging.Logger:
    """Configure structured logging to console and optional file."""
    logger = logging.getLogger("kalshi_bot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    if quiet_research_console:
        console.addFilter(_ConsoleNoiseFilter())
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Shared data classes used across agents ──────────────────────────


@dataclass
class MarketData:
    """Normalized snapshot of a single Kalshi market."""
    ticker: str
    title: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_24h: float
    liquidity: float
    open_interest: float
    last_price: float
    close_time: str | None = None       # ISO-8601
    status: str = "open"
    subtitle: str = ""
    category: str = ""
    event_ticker: str = ""

    @property
    def mid_price(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return 0.0

    @property
    def implied_probability(self) -> float:
        """Market-implied probability of YES (from mid-price in dollars)."""
        return self.mid_price


@dataclass
class CandidateMarket:
    """A market that passed the scan filter, annotated with anomaly signals."""
    market: MarketData
    signals: list[str] = field(default_factory=list)   # e.g. ["volume_spike", "wide_spread"]
    baseline_price: float | None = None
    baseline_volume: float | None = None


@dataclass
class DataPoint:
    """One piece of text from an external data source."""
    text: str
    source: str          # "twitter", "reddit", "news", "scraper"
    timestamp: str = ""  # ISO-8601 or empty
    engagement: float = 0.0  # likes, score, shares, etc.
    url: str = ""


@dataclass
class ResearchResult:
    """Aggregated research output for a single market."""
    ticker: str
    sentiment_score: float          # -1 to +1
    narrative_summary: str
    narrative_confidence: str       # "low", "medium", "high"
    data_point_count: int
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    data_points: list[DataPoint] = field(default_factory=list)
    sportsbook_prob: float | None = None  # vig-adjusted sportsbook consensus (SPORTS only)


@dataclass
class PredictionResult:
    """Output of the Prediction Agent."""
    ticker: str
    true_probability: float
    market_probability: float
    edge: float
    confidence: str          # "low", "medium", "high"
    reasoning: str = ""
    quality_score: float = 0.0
    model_votes: list = field(default_factory=list)  # [{model, probability, weight}, ...]


@dataclass
class TradeDecision:
    """Output of the Risk Agent."""
    approved: bool
    ticker: str = ""
    side: str = ""               # "yes" or "no"
    action: str = "buy"
    size_dollars: float = 0.0
    size_contracts: int = 0
    order_type: str = "limit"    # "limit" or "market"
    limit_price: float = 0.0
    reasoning: str = ""
    prediction: PredictionResult | None = None
    research: ResearchResult | None = None
    signals: list[str] = field(default_factory=list)
    category: str = ""
    spread_at_entry: float = 0.0
