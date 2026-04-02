"""
Category Strategy Profiles — E component of the B+C+E system.

Each Kalshi category has a distinct strategy profile that controls:
  - Which ensemble sub-models are active and their base weights
  - How much sentiment can shift the probability estimate
  - Per-category Kelly fraction baseline
  - How much to trust the market price
  - Minimum edge threshold before considering a trade
  - Which scan signals to prefer or ignore
  - Scan priority multiplier (boosted by StrategyEvolutionAgent for winners)

Code defaults are defined here. The StrategyEvolutionAgent writes JSON overrides
into the heuristics DB under key "category_profile_<CATEGORY>", which
CategoryProfileLoader merges on top of the code defaults at runtime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from database import Database

logger = logging.getLogger("kalshi_bot.category_profiles")

# Canonical category names (uppercased).
# The loader normalises incoming category strings to these.
KNOWN_CATEGORIES = [
    "SPORTS",
    "POLITICS",
    "WEATHER",
    "ECONOMICS",
    "CRYPTO",
    "ENTERTAINMENT",
    "FINANCIALS",
    "SCIENCE",
    "DEFAULT",
]

# Map common Kalshi ticker prefixes → canonical category
_TICKER_PREFIX_MAP: dict[str, str] = {
    # Crypto
    "KXBT": "CRYPTO",
    "KXETH": "CRYPTO",
    "KXDOG": "CRYPTO",
    "KXSOL": "CRYPTO",
    # Sports
    "KXNFL": "SPORTS",
    "KXNBA": "SPORTS",
    "KXMLB": "SPORTS",
    "KXNHL": "SPORTS",
    "KXMMA": "SPORTS",
    "KXNCAA": "SPORTS",
    "KXSOC": "SPORTS",
    "KXATP": "SPORTS",   # ATP tennis (all events)
    "KXWTA": "SPORTS",   # WTA tennis
    "KXPGA": "SPORTS",   # PGA golf
    "KXFOME": "SPORTS",  # French Open men's (KXFOMEN)
    "KXFOWO": "SPORTS",  # French Open women's (KXFOWOM)
    "KXWIMB": "SPORTS",  # Wimbledon
    "KXUSOP": "SPORTS",  # US Open tennis
    "KXAUSO": "SPORTS",  # Australian Open
    # Weather
    "KXNA":  "WEATHER",
    "KXHU":  "WEATHER",
    "KXEQ":  "WEATHER",
    # Economics / Fed
    "KXFED": "ECONOMICS",
    "KXCPI": "ECONOMICS",
    "KXGDP": "ECONOMICS",
    "KXUNR": "ECONOMICS",
    "KXRET": "ECONOMICS",
    # Politics / Elections
    "KXPRE": "POLITICS",
    "KXSEN": "POLITICS",
    "KXELE": "POLITICS",
    # Entertainment
    "KXOSC": "ENTERTAINMENT",
    "KXMOV": "ENTERTAINMENT",
    "KXTVS": "ENTERTAINMENT",
    # Financials / Stocks
    "KXSP5": "FINANCIALS",
    "KXNAS": "FINANCIALS",
    "KXDOW": "FINANCIALS",
    "KXGOL": "FINANCIALS",
    "KXOIL": "FINANCIALS",
}


@dataclass
class CategoryProfile:
    """
    Strategy parameters for a single market category.

    active_models: dict mapping sub-model class name → base weight.
                   Weights are normalised to sum to 1.0 by the ensemble.
    """
    category: str

    # Ensemble model base weights (normalised at runtime)
    active_models: dict[str, float] = field(default_factory=dict)

    # Probability estimation
    sentiment_weight: float = 0.10    # max fraction sentiment can shift estimate
    market_trust: float = 0.50        # 0 = ignore market, 1 = fully trust market

    # Risk / sizing
    kelly_fraction: float = 0.25      # fractional-Kelly multiplier
    edge_threshold: float = 0.01      # minimum |edge| to consider trading (1%)

    # Scan signal preferences
    preferred_signals: list[str] = field(default_factory=list)
    ignored_signals: list[str] = field(default_factory=list)

    # Scan priority multiplier — boosted by StrategyEvolutionAgent for winners
    scan_priority_multiplier: float = 1.0

    # Set by StrategyEvolutionAgent when win_rate is consistently low
    is_underperforming: bool = False


# ---------------------------------------------------------------------------
# Code defaults — one profile per category
# ---------------------------------------------------------------------------
# Design rationale for each category:
#
#  SPORTS   — Kalshi sports lines are set by retail/fan bettors with strong
#             underdog bias.  Sportsbook lines (DraftKings, FanDuel, etc.) are
#             sharp-money calibrated and the dominant signal.  Sentiment is noisy.
#
#  POLITICS — heavy news/media coverage; sentiment can be genuinely
#             informative but polls/momentum matter more.
#
#  WEATHER  — scientific models dominate; social sentiment is near-useless.
#             Market anchor model should dominate.
#
#  ECONOMICS — calendar-driven (Fed dates, CPI release).  Analyst consensus
#              and momentum around release dates are most predictive.
#
#  CRYPTO   — fast-moving; volume and momentum dominate over sentiment.
#             High volatility → lower Kelly.
#
#  ENTERTAINMENT — social sentiment (Oscar buzz, chart positions) is highly
#                  predictive.  Use consensus + sentiment heavily.
#
#  FINANCIALS — correlated to live prices; momentum model most relevant.
#
#  DEFAULT  — balanced weights for any unmapped category.

CODE_DEFAULT_PROFILES: dict[str, CategoryProfile] = {
    "SPORTS": CategoryProfile(
        category="SPORTS",
        # Substantial inefficiency (2.23pp gap) — fan/underdog bias is
        # strong.  Sportsbook lines remain the dominant signal.
        active_models={
            "SportsbookModel":   0.40,  # primary signal — sharp-money consensus
            "MarketAnchorModel": 0.25,  # respect Kalshi price partially
            "MomentumModel":     0.20,  # price drift toward true value
            "VolumeModel":       0.10,  # volume confirms direction
            "SentimentModel":    0.05,  # fan sentiment is noisy — minimal weight
        },
        sentiment_weight=0.06,
        market_trust=0.45,          # ↓ less trust — retail fan bias
        kelly_fraction=0.22,        # ↑ slightly — bias is exploitable
        edge_threshold=0.012,       # ↓ lower bar — edges are real
        preferred_signals=["volume_spike", "price_move"],
        ignored_signals=[],
    ),
    "POLITICS": CategoryProfile(
        category="POLITICS",
        # Moderate inefficiency (1.02pp gap) — tribalism introduces some
        # bias but the market is heavily watched.
        active_models={
            "MarketAnchorModel": 0.35,
            "SentimentModel":    0.25,
            "ConsensusModel":    0.25,
            "MomentumModel":     0.15,
        },
        sentiment_weight=0.12,
        market_trust=0.55,
        kelly_fraction=0.18,        # ↓ moderate — edges are smaller
        edge_threshold=0.018,       # ↑ slightly — many perceived edges are tribal noise
        preferred_signals=["price_move", "volume_spike"],
        ignored_signals=[],
    ),
    "WEATHER": CategoryProfile(
        category="WEATHER",
        active_models={
            "MarketAnchorModel": 0.55,
            "MomentumModel":     0.30,
            "VolumeModel":       0.15,
        },
        sentiment_weight=0.03,
        market_trust=0.70,
        kelly_fraction=0.15,
        edge_threshold=0.020,
        preferred_signals=["price_move"],
        ignored_signals=["wide_spread"],
    ),
    "ECONOMICS": CategoryProfile(
        category="ECONOMICS",
        active_models={
            "MarketAnchorModel": 0.35,
            "ConsensusModel":    0.30,
            "SentimentModel":    0.20,
            "MomentumModel":     0.15,
        },
        sentiment_weight=0.10,
        market_trust=0.55,
        kelly_fraction=0.22,
        edge_threshold=0.015,
        preferred_signals=["volume_spike", "divergence"],
        ignored_signals=[],
    ),
    "CRYPTO": CategoryProfile(
        category="CRYPTO",
        active_models={
            "MomentumModel":     0.35,
            "VolumeModel":       0.30,
            "MarketAnchorModel": 0.25,
            "SentimentModel":    0.10,
        },
        sentiment_weight=0.08,
        market_trust=0.40,
        kelly_fraction=0.15,
        edge_threshold=0.025,
        preferred_signals=["volume_spike", "price_move", "divergence"],
        ignored_signals=[],
    ),
    "ENTERTAINMENT": CategoryProfile(
        category="ENTERTAINMENT",
        # Most inefficient category (4.79pp maker-taker gap) — retail
        # emotional engagement creates massive mispricing.  Lean into
        # sentiment which IS informative here, and size up.
        active_models={
            "MarketAnchorModel": 0.30,
            "SentimentModel":    0.35,  # ↑ sentiment is the edge here
            "ConsensusModel":    0.25,
            "VolumeModel":       0.10,
        },
        sentiment_weight=0.18,      # ↑ sentiment highly predictive
        market_trust=0.40,          # ↓ market is inefficient — trust it less
        kelly_fraction=0.25,        # ↑ higher Kelly — bigger exploitable bias
        edge_threshold=0.008,       # ↓ lower bar — edges are real here
        preferred_signals=["volume_spike"],
        ignored_signals=[],
    ),
    "FINANCIALS": CategoryProfile(
        category="FINANCIALS",
        # Near-efficient category (0.17pp gap) — professional traders
        # dominate.  High edge threshold since perceived edges are likely
        # noise.  Small Kelly since we're competing against quants.
        active_models={
            "MomentumModel":     0.40,
            "MarketAnchorModel": 0.35,  # ↑ trust market more — it's efficient
            "VolumeModel":       0.15,
            "SentimentModel":    0.10,
        },
        sentiment_weight=0.05,      # ↓ sentiment is noise here
        market_trust=0.70,          # ↑ efficient market — trust it heavily
        kelly_fraction=0.10,        # ↓ small size — edges are thin/illusory
        edge_threshold=0.035,       # ↑ high bar — most "edges" are noise
        preferred_signals=["price_move", "volume_spike", "divergence"],
        ignored_signals=[],
    ),
    "SCIENCE": CategoryProfile(
        category="SCIENCE",
        active_models={
            "MarketAnchorModel": 0.40,
            "ConsensusModel":    0.30,
            "SentimentModel":    0.20,
            "MomentumModel":     0.10,
        },
        sentiment_weight=0.08,
        market_trust=0.60,
        kelly_fraction=0.20,
        edge_threshold=0.015,
        preferred_signals=[],
        ignored_signals=[],
    ),
    "DEFAULT": CategoryProfile(
        category="DEFAULT",
        active_models={
            "MarketAnchorModel": 0.30,
            "SentimentModel":    0.20,
            "MomentumModel":     0.20,
            "VolumeModel":       0.15,
            "ConsensusModel":    0.15,
        },
        sentiment_weight=0.10,
        market_trust=0.50,
        kelly_fraction=0.25,
        edge_threshold=0.010,
        preferred_signals=[],
        ignored_signals=[],
    ),
}


class CategoryProfileLoader:
    """
    Returns the effective CategoryProfile for a market.

    Resolution order:
      1. Parse the category from market.category (normalised to uppercase)
         or fall back to ticker-prefix lookup.
      2. Load the code default from CODE_DEFAULT_PROFILES (or DEFAULT).
      3. Apply any JSON override stored in the heuristics DB under
         key "category_profile_<CATEGORY>".  Only recognised scalar
         fields are applied; the rest are ignored.
    """

    def __init__(self, db: "Database") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_profile(self, category: str) -> CategoryProfile:
        canonical = self._normalise(category)
        base = CODE_DEFAULT_PROFILES.get(canonical, CODE_DEFAULT_PROFILES["DEFAULT"])

        # Clone so we never mutate the class-level defaults
        profile = CategoryProfile(
            category=base.category,
            active_models=dict(base.active_models),
            sentiment_weight=base.sentiment_weight,
            market_trust=base.market_trust,
            kelly_fraction=base.kelly_fraction,
            edge_threshold=base.edge_threshold,
            preferred_signals=list(base.preferred_signals),
            ignored_signals=list(base.ignored_signals),
            scan_priority_multiplier=base.scan_priority_multiplier,
            is_underperforming=base.is_underperforming,
        )

        # Apply DB overrides if any exist
        raw = self._db.get_heuristic(f"category_profile_{canonical}")
        if raw:
            try:
                overrides: dict = json.loads(raw)
                _apply_overrides(profile, overrides)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        return profile

    def category_from_market(self, category_field: str, ticker: str) -> str:
        """Return canonical category string for a market."""
        if category_field:
            return self._normalise(category_field)
        return self._normalise(ticker[:6] if ticker else "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Kalshi API returns full human-readable category strings like
    # "Climate and Weather" or "Sports".  Map these to canonical names.
    _KALSHI_CATEGORY_MAP: dict[str, str] = {
        # Sports
        "SPORTS":               "SPORTS",
        "SPORT":                "SPORTS",
        # Weather
        "WEATHER":              "WEATHER",
        "CLIMATE AND WEATHER":  "WEATHER",
        "CLIMATE":              "WEATHER",
        # Politics
        "POLITICS":             "POLITICS",
        "POLITICAL":            "POLITICS",
        "ELECTIONS":            "POLITICS",
        "ELECTION":             "POLITICS",
        "GOVERNMENT":           "POLITICS",
        # Economics
        "ECONOMICS":            "ECONOMICS",
        "ECONOMY":              "ECONOMICS",
        "ECONOMIC INDICATORS":  "ECONOMICS",
        "FEDERAL RESERVE":      "ECONOMICS",
        "FINANCIALS":           "FINANCIALS",
        "FINANCE":              "FINANCIALS",
        "FINANCIAL MARKETS":    "FINANCIALS",
        "STOCKS":               "FINANCIALS",
        "COMPANIES":            "FINANCIALS",
        "COMPANY":              "FINANCIALS",
        # Crypto
        "CRYPTO":               "CRYPTO",
        "CRYPTOCURRENCY":       "CRYPTO",
        "CRYPTOCURRENCIES":     "CRYPTO",
        # Entertainment
        "ENTERTAINMENT":        "ENTERTAINMENT",
        "CULTURE":              "ENTERTAINMENT",
        "POP CULTURE":          "ENTERTAINMENT",
        "AWARDS":               "ENTERTAINMENT",
        "MUSIC":                "ENTERTAINMENT",
        "MOVIES":               "ENTERTAINMENT",
        "TV":                   "ENTERTAINMENT",
        # Science
        "SCIENCE":              "SCIENCE",
        "TECHNOLOGY":           "SCIENCE",
        "TECH":                 "SCIENCE",
        "SPACE":                "SCIENCE",
        "HEALTH":               "SCIENCE",
        # Misc — treat as DEFAULT
        "MENTIONS":             "DEFAULT",
        "SOCIAL MEDIA":         "DEFAULT",
        "NEWS":                 "DEFAULT",
    }

    @staticmethod
    def _normalise(raw: str) -> str:
        """Map raw category/prefix string to a canonical KNOWN_CATEGORIES key."""
        if not raw:
            return "DEFAULT"
        upper = raw.strip().upper()

        # Direct match against canonical names
        if upper in CODE_DEFAULT_PROFILES:
            return upper

        # Match against Kalshi's human-readable API category strings
        mapped = CategoryProfileLoader._KALSHI_CATEGORY_MAP.get(upper)
        if mapped:
            return mapped

        # Partial word match — handles slight variations like "Sports & Gaming"
        for kalshi_str, canonical in CategoryProfileLoader._KALSHI_CATEGORY_MAP.items():
            if kalshi_str in upper or upper in kalshi_str:
                return canonical

        # Ticker-prefix lookup (try longest prefix first)
        for length in (7, 6, 5, 4):
            prefix = upper[:length]
            if prefix in _TICKER_PREFIX_MAP:
                return _TICKER_PREFIX_MAP[prefix]

        return "DEFAULT"


def _apply_overrides(profile: CategoryProfile, overrides: dict) -> None:
    """Safely merge scalar DB overrides into a profile object."""
    float_fields = {
        "sentiment_weight", "market_trust", "kelly_fraction",
        "edge_threshold", "scan_priority_multiplier",
    }
    bool_fields = {"is_underperforming"}
    dict_fields = {"active_models"}

    for key, val in overrides.items():
        if key in float_fields:
            try:
                setattr(profile, key, float(val))
            except (TypeError, ValueError):
                pass
        elif key in bool_fields:
            setattr(profile, key, bool(val))
        elif key in dict_fields:
            if isinstance(val, dict):
                setattr(profile, key, val)
