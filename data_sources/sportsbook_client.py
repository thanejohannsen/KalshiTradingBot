"""
Sportsbook Odds Client — fetches moneyline odds from The Odds API and converts
them to vig-adjusted implied probabilities for comparison against Kalshi lines.

The core insight: Kalshi sports markets are priced by retail bettors with fan
bias.  Sportsbooks are priced by sharp money and sophisticated models.  When
a sportsbook has a team at 70% and Kalshi has them at 55%, that 15-point gap
is exactly the edge the bot should exploit.

API: https://the-odds-api.com  (free tier = 500 requests/month)
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("kalshi_bot.sportsbook")

_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# Cache per sport key: (monotonic_timestamp, data)
# Free tier = 500 req/month (~16/day). With 4-hour TTL and ~2-3 active sports,
# we use roughly 2 × 6 = 12 requests/day = 360/month — safely within the limit.
_CACHE_TTL_SECONDS = 14400  # 4 hours

# Minimum word overlap score to accept a team match
_MIN_MATCH_SCORE = 0.5

# Direct mapping: Kalshi ticker prefix → The Odds API sport key(s).
# Values can be a single string or a list (tried in order, stops on first match).
# The Odds API organises tennis by individual tournament, so KXATP gets a list
# of the four Grand Slams plus key Masters events.
_TICKER_PREFIX_TO_SPORT: dict[str, str | list[str]] = {
    "KXNFL":    "americanfootball_nfl",
    "KXNCAA":   "americanfootball_ncaaf",
    "KXNBA":    "basketball_nba",
    "KXNCAAB":  "basketball_ncaab",
    "KXMLB":    "baseball_mlb",
    "KXNHL":    "icehockey_nhl",
    "KXMMA":    "mma_mixed_martial_arts",
    "KXSOC":    "soccer_usa_mls",
    "KXPGA":    "golf_pga_championship",
    "KXWWE":    "mma_mixed_martial_arts",
    # ATP tennis — KXATP covers ALL ATP events (tour matches + grand slams).
    # Odds API is tournament-specific so we try all in calendar order.
    "KXATP": [
        "tennis_atp_australian_open",
        "tennis_atp_french_open",
        "tennis_atp_wimbledon",
        "tennis_atp_us_open",
        "tennis_atp_miami_open",
        "tennis_atp_madrid_open",
        "tennis_atp_rome",
        "tennis_atp_canadian_open",
        "tennis_atp_cincinnati",
        "tennis_atp_vienna",
        "tennis_atp_paris",
    ],
    # WTA tennis
    "KXWTA": [
        "tennis_wta_australian_open",
        "tennis_wta_french_open",
        "tennis_wta_wimbledon",
        "tennis_wta_us_open",
        "tennis_wta_miami_open",
        "tennis_wta_madrid_open",
    ],
    # Kalshi uses separate prefixes for individual Grand Slam tournaments.
    # French Open men's/women's
    "KXFOMEN": ["tennis_atp_french_open"],
    "KXFOWOM": ["tennis_wta_french_open"],
    # Wimbledon — add more prefixes here as you discover them from Kalshi
    "KXWIMB":  ["tennis_atp_wimbledon", "tennis_wta_wimbledon"],
    # US Open
    "KXUSOP":  ["tennis_atp_us_open", "tennis_wta_us_open"],
    # Australian Open
    "KXAUSO":  ["tennis_atp_australian_open", "tennis_wta_australian_open"],
}

# Fallback scan order when ticker prefix doesn't match — only used if we can't
# determine the sport from the ticker. Ordered by Kalshi popularity.
_FALLBACK_SPORT_KEYS = [
    "basketball_nba",
    "americanfootball_nfl",
    "baseball_mlb",
    "icehockey_nhl",
    "mma_mixed_martial_arts",
    "soccer_usa_mls",
    "americanfootball_ncaaf",
    "basketball_ncaab",
]

# Words to strip from Kalshi market titles before team-name matching
_STOP_WORDS = frozenset(
    "will the win beat cover spread vs against who wins in to a an by of "
    "super bowl nfl nba mlb nhl mma ncaa series game match championship "
    "playoff playoffs tournament open masters us pga regular season title "
    "finals conference division wild card".split()
)


class SportsbookClient:
    """
    Fetches sportsbook consensus implied probabilities for sporting events and
    matches them to Kalshi markets via ticker-directed sport key lookups.

    Matching strategy (in order):
      1. Parse the Kalshi ticker prefix (e.g. KXNFL) → map directly to the
         correct Odds API sport key (e.g. americanfootball_nfl).  One API
         call, no guessing.
      2. Extract team token from the ticker segment (e.g. KXNFL-CHIEFS-WIN
         → "chiefs") and use it as the primary match signal.
      3. Also parse team tokens from the market title as a supplementary signal.
      4. If the ticker prefix is unknown, fall back to scanning the most
         common sport keys in popularity order (stops on first match).
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._quota_exceeded = False  # set True on 401/429 — stops all further requests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_implied_prob(self, market_title: str, ticker: str = "") -> float | None:
        """
        Look up the sportsbook consensus implied probability for a Kalshi market.

        Args:
            market_title: Human-readable title, e.g. "Will the Chiefs win Super Bowl?"
            ticker:       Kalshi ticker, e.g. "KXNFL-CHIEFS-WIN-2025" (used to
                          identify sport and extract team name hint).

        Returns vig-adjusted float in [0, 1], or None if no match found.
        """
        if not self._api_key or self._quota_exceeded:
            return None

        # Derive team query: ticker segment is most reliable, title is fallback
        ticker_team = self._team_from_ticker(ticker)
        title_query = self._title_to_query(market_title)
        # Combine: ticker team first (higher precision), then title tokens
        combined_query = f"{ticker_team} {title_query}".strip() if ticker_team else title_query
        if not combined_query:
            return None

        # Identify the sport key(s) from ticker prefix — avoids scanning all sports
        sport_keys = self._sport_keys_from_ticker(ticker)

        search_list = sport_keys if sport_keys else _FALLBACK_SPORT_KEYS
        label = "targeted" if sport_keys else "fallback"

        for sk in search_list:
            if self._quota_exceeded:
                return None
            events = self._fetch_odds(sk)
            if not events:
                continue
            prob = self._find_team_prob(combined_query, events)
            if prob is not None:
                logger.debug(
                    "Sportsbook [%s %s]: '%s' → %.1f%%",
                    sk, label, combined_query, prob * 100,
                )
                return prob

        logger.debug("Sportsbook: no match found for '%s'", market_title[:60])
        return None

    # ------------------------------------------------------------------
    # Internal: odds fetching with caching
    # ------------------------------------------------------------------
    def _fetch_odds(self, sport_key: str) -> list[dict] | None:
        if self._quota_exceeded:
            return None

        now = time.monotonic()
        cached = self._cache.get(sport_key)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

        url = (
            f"{_BASE_URL}/{sport_key}/odds/"
            f"?apiKey={self._api_key}"
            f"&regions=us"
            f"&markets=h2h"
            f"&oddsFormat=decimal"
        )
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data: list[dict] = json.loads(resp.read())
            self._cache[sport_key] = (now, data)
            logger.debug("Fetched %d events for %s", len(data), sport_key)
            return data

        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                self._quota_exceeded = True
                logger.warning(
                    "⚠️  ODDS API QUOTA EXCEEDED (HTTP 401) — sportsbook comparisons "
                    "disabled for this session. Upgrade at https://the-odds-api.com "
                    "or wait until your monthly quota resets."
                )
            elif exc.code == 429:
                self._quota_exceeded = True
                logger.warning(
                    "⚠️  ODDS API RATE LIMITED (HTTP 429) — sportsbook comparisons "
                    "disabled for this session. Monthly quota may be exhausted."
                )
            elif exc.code == 422:
                # Usually means the sport key doesn't exist / no active events
                logger.debug("Odds API: sport key '%s' has no active events", sport_key)
            else:
                logger.warning("Odds API HTTP error [%s]: %d %s", sport_key, exc.code, exc.reason)
            return None

        except Exception as exc:
            logger.debug("Odds API fetch failed [%s]: %s", sport_key, exc)
            return None

    # ------------------------------------------------------------------
    # Internal: ticker parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _sport_keys_from_ticker(ticker: str) -> list[str]:
        """
        Map a Kalshi ticker prefix to a list of Odds API sport keys to try.
        Returns empty list if the prefix is not recognised.

        Examples:
            "KXNFL-CHIEFS-WIN-2025" → ["americanfootball_nfl"]
            "KXNBA-BOS-CHAMP"       → ["basketball_nba"]
            "KXATP-DJOKOVIC-FRN25"  → ["tennis_atp_french_open", "tennis_atp_wimbledon", ...]
        """
        if not ticker:
            return []
        upper = ticker.upper()
        # Try longest prefix first so KXPGATOUR matches before KXPGA, etc.
        for length in (7, 6, 5, 4):
            prefix = upper[:length]
            if prefix in _TICKER_PREFIX_TO_SPORT:
                value = _TICKER_PREFIX_TO_SPORT[prefix]
                return value if isinstance(value, list) else [value]
        return []

    @staticmethod
    def _team_from_ticker(ticker: str) -> str:
        """
        Extract the team/player name segment from a Kalshi ticker.

        Kalshi tickers follow the pattern: PREFIX-TEAM-DESCRIPTOR-YEAR
        e.g. "KXNFL-CHIEFS-WIN-2025"        → "chiefs"
             "KXNBA-BOS-CHAMPION"           → "bos"
             "KXMLB-YANKEES-WS"             → "yankees"
             "KXFOMEN-26"                   → ""  (no useful team segment)
             "KXATPMATCH-26MAR30MOLCLA"     → ""  (date blob — skip, use title instead)

        Returns empty string when the segment is a date, year, or ambiguous blob
        so the caller falls back to title-based matching.
        """
        if not ticker:
            return ""
        parts = ticker.upper().split("-")
        if len(parts) < 2:
            return ""

        segment = parts[1]

        # Skip if the segment is purely a year (e.g. "26", "2026")
        if segment.isdigit():
            return ""

        # Skip if the segment starts with a digit — likely a date+player blob
        # e.g. "26MAR30MOLCLA" from KXATPMATCH-26MAR30MOLCLA
        if segment and segment[0].isdigit():
            return ""

        # Skip very short segments (2 chars or less) — not useful for matching
        if len(segment) <= 2:
            return ""

        return segment.lower()

    # ------------------------------------------------------------------
    # Internal: title parsing and team matching
    # ------------------------------------------------------------------
    @staticmethod
    def _title_to_query(title: str) -> str:
        """Strip noise words from a Kalshi market title, leaving team/player name tokens."""
        cleaned = title.lower()
        # Remove punctuation
        cleaned = re.sub(r"[?!.,\-–]", " ", cleaned)
        # Remove stop words
        tokens = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) >= 3]
        return " ".join(tokens)

    def _find_team_prob(self, query: str, events: list[dict]) -> float | None:
        """
        Scan all events for the team that best matches the query string.
        Returns vig-adjusted consensus implied probability for that team.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        best_score = 0.0
        best_team: str | None = None
        best_event: dict | None = None

        for event in events:
            for team in (event.get("home_team", ""), event.get("away_team", "")):
                if not team:
                    continue
                score = self._match_score(team.lower(), query_words, query_lower)
                if score > best_score:
                    best_score = score
                    best_team = team
                    best_event = event

        if best_score < _MIN_MATCH_SCORE or best_team is None or best_event is None:
            return None

        return self._team_consensus_prob(best_team, best_event)

    @staticmethod
    def _match_score(team_lower: str, query_words: set[str], query_lower: str) -> float:
        """
        Score how well a team name matches the query.
        Word overlap is primary; substring match is a secondary signal.
        """
        team_words = set(team_lower.split())
        overlap = len(query_words & team_words)
        if overlap:
            return float(overlap)
        # Partial credit: significant substring match (e.g. "chiefs" in title)
        for w in team_words:
            if len(w) >= 4 and w in query_lower:
                return 0.5
        return 0.0

    @staticmethod
    def _team_consensus_prob(team: str, event: dict) -> float | None:
        """
        Compute vig-adjusted implied win probability averaged across all
        available bookmakers that list this team.
        """
        probs: list[float] = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes: list[dict] = market.get("outcomes", [])
                # Compute raw implied probs (decimal odds → 1/odds)
                raw: dict[str, float] = {}
                for outcome in outcomes:
                    price = outcome.get("price", 0.0)
                    if price > 1.0:
                        raw[outcome.get("name", "")] = 1.0 / price

                if not raw:
                    continue

                # Vig-remove: normalise so probabilities sum to 1
                total = sum(raw.values())
                if total <= 0:
                    continue
                adjusted = {k: v / total for k, v in raw.items()}

                # Find our team (case-insensitive)
                for name, prob in adjusted.items():
                    if name.lower() == team.lower():
                        probs.append(prob)
                        break

        if not probs:
            return None

        return round(sum(probs) / len(probs), 4)
