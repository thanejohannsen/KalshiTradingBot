"""
Configuration module — loads all settings from .env with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_float(key: str, default: float = 0.0) -> float:
    raw = _env(key)
    return float(raw) if raw else default


def _env_int(key: str, default: int = 0) -> int:
    raw = _env(key)
    return int(raw) if raw else default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # --- Execution mode ---
    mode: str = field(default_factory=lambda: _env("MODE", "paper"))

    # --- Kalshi API ---
    kalshi_api_key: str = field(default_factory=lambda: _env("KALSHI_API_KEY"))
    kalshi_private_key_path: str = field(default_factory=lambda: _env("KALSHI_PRIVATE_KEY_PATH"))
    kalshi_api_base: str = field(
        default_factory=lambda: _env(
            "KALSHI_API_BASE",
            "https://api.elections.kalshi.com/trade-api/v2",
        )
    )

    # --- Anthropic / Claude (optional) ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))

    # --- Reddit ---
    reddit_client_id: str = field(default_factory=lambda: _env("REDDIT_CLIENT_ID"))
    reddit_client_secret: str = field(default_factory=lambda: _env("REDDIT_CLIENT_SECRET"))
    reddit_user_agent: str = field(
        default_factory=lambda: _env("REDDIT_USER_AGENT", "KalshiBot/1.0")
    )

    # --- NewsAPI ---
    newsapi_key: str = field(default_factory=lambda: _env("NEWSAPI_KEY"))

    # --- The Odds API (sportsbook line comparison) ---
    odds_api_key: str = field(default_factory=lambda: _env("ODDS_API_KEY"))

    # --- Scan agent ---
    max_markets: int = field(default_factory=lambda: _env_int("MAX_MARKETS", 5000))
    scan_interval_seconds: int = field(
        default_factory=lambda: _env_int("SCAN_INTERVAL_SECONDS", 300)
    )
    min_volume_24h: int = field(default_factory=lambda: _env_int("MIN_VOLUME_24H", 50))
    min_liquidity_dollars: float = field(
        default_factory=lambda: _env_float("MIN_LIQUIDITY_DOLLARS", 100.0)
    )
    max_time_to_resolution_days: int = field(
        default_factory=lambda: _env_int("MAX_TIME_TO_RESOLUTION_DAYS", 45)
    )
    min_time_to_resolution_hours: int = field(
        default_factory=lambda: _env_int("MIN_TIME_TO_RESOLUTION_HOURS", 1)
    )
    price_move_threshold_pct: float = field(
        default_factory=lambda: _env_float("PRICE_MOVE_THRESHOLD_PCT", 7.0)
    )
    volume_spike_multiplier: float = field(
        default_factory=lambda: _env_float("VOLUME_SPIKE_MULTIPLIER", 2.0)
    )
    wide_spread_cents: int = field(default_factory=lambda: _env_int("WIDE_SPREAD_CENTS", 4))
    divergence_cents: int = field(default_factory=lambda: _env_int("DIVERGENCE_CENTS", 5))

    # --- Prediction agent ---
    min_edge_threshold: float = field(
        default_factory=lambda: _env_float("MIN_EDGE_THRESHOLD", 0.01)
    )

    # --- Risk agent ---
    bankroll: float = field(default_factory=lambda: _env_float("BANKROLL", 1000.0))
    max_exposure_pct: float = field(
        default_factory=lambda: _env_float("MAX_EXPOSURE_PCT", 0.05)
    )
    kelly_fraction: float = field(
        default_factory=lambda: _env_float("KELLY_FRACTION", 0.25)
    )
    max_open_trades: int = field(
        default_factory=lambda: _env_int("MAX_OPEN_TRADES", 0)  # 0 = unlimited
    )
    min_trade_dollars: float = field(
        default_factory=lambda: _env_float("MIN_TRADE_DOLLARS", 1.0)
    )
    buy_order_adjust_seconds: int = field(
        default_factory=lambda: _env_int("BUY_ORDER_ADJUST_SECONDS", 60)
    )
    exit_order_adjust_seconds: int = field(
        default_factory=lambda: _env_int("EXIT_ORDER_ADJUST_SECONDS", 5)
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: _env("LOG_FILE", "kalshi_bot.log"))
    quiet_research_console: bool = field(
        default_factory=lambda: _env_bool("QUIET_RESEARCH_CONSOLE", True)
    )

    # --- Paths ---
    db_path: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "kalshi_bot.db")
    )

    @property
    def is_paper(self) -> bool:
        return self.mode.lower() == "paper"

    @property
    def is_live(self) -> bool:
        return self.mode.lower() == "live"

    @property
    def claude_enabled(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def reddit_enabled(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret)

    @property
    def newsapi_enabled(self) -> bool:
        return bool(self.newsapi_key)

    @property
    def odds_api_enabled(self) -> bool:
        return bool(self.odds_api_key)
