"""
SCAN AGENT — filters the full Kalshi market list down to actionable candidates.

V2: Bias-scoring approach (replaces anomaly-signal gating).

Instead of requiring anomaly signals (wide spread, volume spike, etc.)
this version scores every market that passes hard filters on how
*exploitable* its microstructure biases are likely to be, based on:

  1. Category inefficiency  — Entertainment/Media/Sports have the widest
                              maker-taker gaps (research: 2-7 pp)
  2. Price zone             — 70-92% YES (scalp favorites) and 25-45%
                              (fade retail YES bias) score highest
  3. Market quality         — tight spread, both sides quoted, decent volume
  4. Stability              — price NOT deviating wildly = bias persists

Old anomaly signals (volume_spike, price_move, etc.) are now optional
boosters — they add to the score but are no longer required.

A market becomes a candidate if its bias score exceeds a threshold,
even with zero anomaly signals.  This means a stable, liquid NBA game
at 78% passes the scan simply because the category bias + price zone
make it worth analyzing.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from models.category_profiles import CategoryProfileLoader
from utils import CandidateMarket, MarketData

logger = logging.getLogger("kalshi_bot.scan_agent")

# Lazy-loaded calibration module
_calibration = None

def _get_calibration():
    global _calibration
    if _calibration is None:
        try:
            from models import calibration
            _calibration = calibration if calibration.is_available() else False
        except ImportError:
            _calibration = False
    return _calibration if _calibration else None

# Minimum bias score for a market to become a candidate.
# Roughly: category_exploitability (0-4) + price_zone (0-3) + market_quality (0-3)
# A score of 3 means "decent category + good price zone + tradeable quality"
MIN_BIAS_SCORE = 3.0

# ── Category exploitability scores ─────────────────────────────────
# Based on maker-taker gap research (Becker 2026, 72M Kalshi trades):
#   Media:         7.28 pp gap → 4.0
#   Entertainment: 4.79 pp gap → 3.5
#   Sports:        2.23 pp gap → 2.5
#   Politics:      1.02 pp gap → 1.5
#   Economics:     ~0.5 pp est → 1.0
#   Crypto:        ~0.5 pp est → 1.0
#   Finance:       0.17 pp gap → 0.5  (near-efficient, MMs dominate)
#   Science/Other: unknown     → 1.0
_CATEGORY_EXPLOITABILITY: dict[str, float] = {
    "ENTERTAINMENT": 3.5,
    "SPORTS":        2.5,
    "POLITICS":      1.5,
    "ECONOMICS":     1.0,
    "CRYPTO":        1.0,
    "FINANCIALS":    0.5,
    "SCIENCE":       1.0,
    "WEATHER":       0.5,   # model-driven, low retail bias
    "DEFAULT":       1.0,
}


class ScanAgent:
    def __init__(self, cfg: Config, db: Database, kalshi: KalshiAPIClient) -> None:
        self._cfg = cfg
        self._db = db
        self._kalshi = kalshi
        self._profile_loader = CategoryProfileLoader(db)

    def _load_learned_context(self) -> None:
        """Load per-category avoidances and scan priorities from DB."""
        all_h = self._db.get_all_heuristics()
        self._avoided_categories: set[str] = set()
        self._category_scan_priority: dict[str, float] = {}

        for k, v in all_h.items():
            if k.startswith("avoid_category_") and v == "true":
                raw_cat = k.replace("avoid_category_", "")
                self._avoided_categories.add(raw_cat)
                self._avoided_categories.add(CategoryProfileLoader._normalise(raw_cat))
            elif k.startswith("category_scan_priority_"):
                cat = k.replace("category_scan_priority_", "")
                try:
                    self._category_scan_priority[cat] = float(v)
                except (ValueError, TypeError):
                    pass

        if self._avoided_categories:
            logger.info("LEARNED: avoiding categories %s", self._avoided_categories)

    # ------------------------------------------------------------------
    # Main scan
    # ------------------------------------------------------------------
    def scan(self) -> list[CandidateMarket]:
        """
        Fetch all active markets, apply hard filters, score for bias
        exploitability, return shortlist ranked by score.
        """
        logger.info("=== SCAN AGENT: starting market scan ===")
        self._load_learned_context()

        markets = self._kalshi.get_active_markets(max_markets=self._cfg.max_markets)
        logger.info("Total active markets: %d", len(markets))

        candidates: list[CandidateMarket] = []
        filter_counts = {
            "avoided_category": 0,
            "low_volume": 0,
            "low_liquidity": 0,
            "time_filter": 0,
            "longshot_block": 0,
            "low_score": 0,
            "zero_price": 0,
        }

        for mkt in markets:
            # Skip markets with zero price data (not yet active)
            if mkt.last_price == 0 and mkt.yes_bid == 0 and mkt.yes_ask == 0:
                filter_counts["zero_price"] += 1
                self._record_snapshot(mkt)
                continue

            # Skip avoided categories (normalised both ways)
            category = self._market_category(mkt)
            canonical = CategoryProfileLoader._normalise(category)
            if category in self._avoided_categories or canonical in self._avoided_categories:
                filter_counts["avoided_category"] += 1
                self._record_snapshot(mkt)
                continue

            # Hard filters
            filter_reason = self._check_hard_filters(mkt)
            if filter_reason:
                filter_counts[filter_reason] = filter_counts.get(filter_reason, 0) + 1
                self._record_snapshot(mkt)
                continue

            # Record snapshot BEFORE scoring (so baselines are available)
            self._record_snapshot(mkt)

            # Score for bias exploitability
            score, signals = self._compute_bias_score(mkt, canonical)

            if score < MIN_BIAS_SCORE:
                filter_counts["low_score"] += 1
                continue

            baseline = self._db.get_baseline(mkt.ticker, hours=24)
            candidates.append(
                CandidateMarket(
                    market=mkt,
                    signals=signals,
                    baseline_price=baseline["avg_price"] if baseline else None,
                    baseline_volume=baseline["avg_volume"] if baseline else None,
                )
            )

        # Log filter breakdown
        passed_hard = (
            len(markets)
            - filter_counts["zero_price"]
            - filter_counts["avoided_category"]
            - filter_counts["low_volume"]
            - filter_counts["low_liquidity"]
            - filter_counts["time_filter"]
            - filter_counts["longshot_block"]
        )
        logger.info(
            "Scan filter breakdown: %d total → %d zero_price, %d low_vol, "
            "%d low_liq, %d time, %d avoided_cat, %d longshot, "
            "%d passed_hard → %d low_score, %d CANDIDATES",
            len(markets),
            filter_counts["zero_price"],
            filter_counts["low_volume"],
            filter_counts["low_liquidity"],
            filter_counts["time_filter"],
            filter_counts["avoided_category"],
            filter_counts["longshot_block"],
            passed_hard,
            filter_counts["low_score"],
            len(candidates),
        )

        if not candidates and passed_hard > 0:
            logger.info("No candidates. Sampling markets that passed hard filters:")
            sample_count = 0
            for mkt in markets:
                if mkt.last_price == 0 and mkt.yes_bid == 0 and mkt.yes_ask == 0:
                    continue
                if not self._passes_hard_filters(mkt):
                    continue
                cat = CategoryProfileLoader._normalise(self._market_category(mkt))
                sc, _ = self._compute_bias_score(mkt, cat)
                logger.info(
                    "  sample: %s | price=%.2f bid=%.2f ask=%.2f spread=%.1fc | "
                    "vol=%.0f | cat=%s | score=%.1f (need %.1f)",
                    mkt.ticker[:40], mkt.last_price, mkt.yes_bid, mkt.yes_ask,
                    mkt.spread * 100, mkt.volume_24h, cat, sc, MIN_BIAS_SCORE,
                )
                sample_count += 1
                if sample_count >= 8:
                    break

        for c in candidates:
            cat = CategoryProfileLoader._normalise(self._market_category(c.market))
            sc, _ = self._compute_bias_score(c.market, cat)
            logger.info(
                "  → %s | score=%.1f | signals=%s | price=%.2f | vol=%.0f | cat=%s",
                c.market.ticker, sc, c.signals,
                c.market.last_price, c.market.volume_24h, cat,
            )

        # Rank by bias score (higher = more exploitable)
        candidates.sort(
            key=lambda c: self._compute_bias_score(
                c.market,
                CategoryProfileLoader._normalise(self._market_category(c.market)),
            )[0],
            reverse=True,
        )

        return candidates

    # ------------------------------------------------------------------
    # Hard filters (must pass ALL)
    # ------------------------------------------------------------------
    @staticmethod
    def _market_category(mkt: MarketData) -> str:
        return mkt.category if mkt.category else (mkt.ticker[:7] if mkt.ticker else "")

    def _check_hard_filters(self, mkt: MarketData) -> str | None:
        """Return filter reason string if blocked, or None if passed."""
        if mkt.volume_24h < self._cfg.min_volume_24h:
            return "low_volume"

        if mkt.yes_bid <= 0 and mkt.yes_ask <= 0:
            return "low_liquidity"

        if mkt.close_time:
            try:
                close_dt = datetime.fromisoformat(mkt.close_time.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hours_left = (close_dt - now).total_seconds() / 3600.0

                if hours_left < self._cfg.min_time_to_resolution_hours:
                    return "time_filter"
                if hours_left > self._cfg.max_time_to_resolution_days * 24:
                    return "time_filter"
            except (ValueError, TypeError):
                pass

        # Hard longshot block: contracts below 15¢ on either side
        mid = mkt.last_price or mkt.yes_ask or 0
        if 0 < mid < 0.15:
            return "longshot_block"
        if mid > 0.85:
            return "longshot_block"

        return None

    def _passes_hard_filters(self, mkt: MarketData) -> bool:
        return self._check_hard_filters(mkt) is None

    # ------------------------------------------------------------------
    # Bias scoring (replaces anomaly signal gating)
    # ------------------------------------------------------------------
    def _compute_bias_score(
        self, mkt: MarketData, canonical_category: str,
    ) -> tuple[float, list[str]]:
        """
        Score a market for exploitable bias (0-10+ scale).

        Components:
          1. Category exploitability (0-4)  — how inefficient is this category?
          2. Price zone (0-3)               — is the price in a biased range?
          3. Market quality (0-3)           — spread, quoting, volume
          4. Stability bonus (0-1)          — stable price = persistent bias
          5. Anomaly boosters (0-2)         — optional: volume spike, price move

        Returns (score, signal_list) where signal_list explains what scored.
        """
        score = 0.0
        signals: list[str] = []
        mid = mkt.last_price or ((mkt.yes_bid + mkt.yes_ask) / 2 if mkt.yes_bid and mkt.yes_ask else 0)
        spread_cents = mkt.spread * 100

        # ── 1. Category exploitability (0-4) ───────────────────────
        cat_score = _CATEGORY_EXPLOITABILITY.get(canonical_category, 1.0)

        # Apply learned scan priority (StrategyEvolution can boost/demote)
        priority = self._category_scan_priority.get(canonical_category, 1.0)
        cat_score *= max(0.5, min(1.5, priority))

        score += cat_score
        if cat_score >= 2.0:
            signals.append(f"high_bias_category({canonical_category})")

        # ── 2. Price zone (0-3) ────────────────────────────────────
        # Research: YES at 70-92% is the scalp zone (backing favorites).
        # 25-45% is the "fade retail YES bias" zone (bet NO against
        # retail optimism pushing YES contracts above fair value).
        # 45-70% is the efficient zone — harder edges, lower score.
        pz_score = 0.0
        if 0.70 <= mid <= 0.92:
            # Scalp favorite zone — highest +EV for YES bets
            # Peak at ~80%, tapering toward edges
            pz_score = 3.0 - abs(mid - 0.80) * 8.0
            pz_score = max(1.5, min(3.0, pz_score))
            signals.append(f"scalp_zone({mid:.2f})")
        elif 0.25 <= mid < 0.45:
            # Fade retail YES bias zone — good for NO bets
            pz_score = 2.0
            signals.append(f"fade_retail_zone({mid:.2f})")
        elif 0.45 <= mid < 0.70:
            # Mid-range: edges exist but smaller
            pz_score = 1.0
        elif 0.15 <= mid < 0.25:
            # Low probability — small score, marginal
            pz_score = 0.5
        # else: 0 (extremes already blocked by hard filter)
        score += pz_score

        # ── 3. Market quality (0-3) ────────────────────────────────
        mq_score = 0.0

        # Both sides quoted
        if mkt.yes_bid > 0 and mkt.yes_ask > 0:
            mq_score += 1.0

        # Tight spread (< 6¢ = good, < 3¢ = excellent)
        if 0 < spread_cents <= 3:
            mq_score += 1.0
        elif 0 < spread_cents <= 6:
            mq_score += 0.5

        # Volume sweet spot (5k-50k preferred — liquid but not MM-dominated)
        if 5_000 <= mkt.volume_24h < 50_000:
            mq_score += 1.0
            signals.append(f"sweet_volume({mkt.volume_24h:.0f})")
        elif mkt.volume_24h >= 50_000:
            mq_score += 0.5
        elif mkt.volume_24h >= 500:
            mq_score += 0.5
        # else: low volume, no points

        score += mq_score

        # ── 4. Stability bonus (0-1) ──────────────────────────────
        # If price hasn't moved much from baseline, the bias is
        # persistent — not being corrected by news/information.
        baseline = self._db.get_baseline(mkt.ticker, hours=24)
        if baseline and baseline["avg_price"] and baseline["avg_price"] > 0:
            drift_pct = abs(mkt.last_price - baseline["avg_price"]) / baseline["avg_price"] * 100
            if drift_pct < 5.0:
                score += 1.0
                signals.append("stable_price")
            elif drift_pct < 10.0:
                score += 0.5

        # ── 5. Calibration model edge (0-3) — ML-driven signal ────
        # If the calibration model is available, its predicted edge is
        # the single most informative signal.  A 5pp+ calibration edge
        # means this price zone × category historically misprices.
        cal = _get_calibration()
        if cal:
            cal_edge = cal.get_calibration_edge(
                market_price=mid,
                category=canonical_category,
                spread=mkt.spread,
                volume_24h=mkt.volume_24h,
                open_interest=mkt.open_interest,
                yes_bid=mkt.yes_bid,
                yes_ask=mkt.yes_ask,
                no_bid=mkt.no_bid,
                no_ask=mkt.no_ask,
            )
            if cal_edge is not None:
                abs_edge = abs(cal_edge)
                if abs_edge >= 0.08:
                    score += 3.0
                    signals.append(f"cal_edge({cal_edge:+.3f})")
                elif abs_edge >= 0.05:
                    score += 2.0
                    signals.append(f"cal_edge({cal_edge:+.3f})")
                elif abs_edge >= 0.02:
                    score += 1.0
                    signals.append(f"cal_edge({cal_edge:+.3f})")

        # ── 6. Anomaly boosters (0-2) — optional extras ───────────
        # These are the old signals, now worth only bonus points.
        if baseline and baseline["avg_volume"] and baseline["avg_volume"] > 0:
            volume_ratio = mkt.volume_24h / baseline["avg_volume"]
            if volume_ratio >= self._cfg.volume_spike_multiplier:
                score += 0.5
                signals.append(f"volume_spike({volume_ratio:.1f}x)")

        if baseline and baseline["avg_price"] and baseline["avg_price"] > 0:
            price_change_pct = abs(
                (mkt.last_price - baseline["avg_price"]) / baseline["avg_price"]
            ) * 100
            if price_change_pct >= self._cfg.price_move_threshold_pct:
                score += 0.5
                signals.append(f"price_move({price_change_pct:.1f}%)")

        # New market with decent activity (no baseline)
        if not baseline and mkt.volume_24h > self._cfg.min_volume_24h * 2:
            score += 0.5
            signals.append("new_market")

        return score, signals

    # ------------------------------------------------------------------
    # Snapshot recording
    # ------------------------------------------------------------------
    def _record_snapshot(self, mkt: MarketData) -> None:
        self._db.save_snapshot(
            {
                "ticker": mkt.ticker,
                "title": mkt.title,
                "yes_bid": mkt.yes_bid,
                "yes_ask": mkt.yes_ask,
                "no_bid": mkt.no_bid,
                "no_ask": mkt.no_ask,
                "volume_24h": mkt.volume_24h,
                "liquidity": mkt.liquidity,
                "open_interest": mkt.open_interest,
                "last_price": mkt.last_price,
            }
        )

    # ------------------------------------------------------------------
    # Cold-start bootstrap
    # ------------------------------------------------------------------
    def bootstrap_baselines(self, top_n: int = 300) -> None:
        """
        On cold start, fetch the top markets by volume/liquidity, grab
        candlestick history for each, and seed the snapshot table so the
        first real scan already has baselines for price stability checks.
        """
        logger.info("=== BOOTSTRAP: seeding baselines for top %d markets ===", top_n)

        markets = self._kalshi.get_active_markets(max_markets=self._cfg.max_markets)
        logger.info("Bootstrap: fetched %d markets, selecting top %d", len(markets), top_n)

        markets.sort(key=lambda m: m.volume_24h * max(m.liquidity, 1), reverse=True)
        top_markets = markets[:top_n]

        seeded = 0
        failed = 0
        for i, mkt in enumerate(top_markets):
            if self._kalshi._shutdown_check():
                logger.info("Bootstrap: shutdown requested, stopping at %d/%d", i, top_n)
                break

            series_ticker = self._derive_series_ticker(mkt.event_ticker)
            if not series_ticker:
                self._record_snapshot(mkt)
                seeded += 1
                continue

            candles = self._kalshi.get_market_history(
                ticker=mkt.ticker,
                series_ticker=series_ticker,
            )

            if candles:
                self._candles_to_snapshots(mkt, candles)
                seeded += 1
            else:
                self._record_snapshot(mkt)
                seeded += 1
                failed += 1

            time.sleep(0.3)

            if (i + 1) % 50 == 0:
                logger.info("Bootstrap progress: %d/%d markets processed", i + 1, top_n)

        logger.info(
            "Bootstrap complete: seeded %d markets (%d without candles, used current price)",
            seeded, failed,
        )

    @staticmethod
    def _derive_series_ticker(event_ticker: str) -> str | None:
        if not event_ticker:
            return None
        parts = event_ticker.split("-")
        series_parts: list[str] = []
        for part in parts:
            if re.match(r"^\d", part):
                break
            series_parts.append(part)
        return "-".join(series_parts) if series_parts else None

    def _candles_to_snapshots(self, mkt: MarketData, candles: list[dict]) -> None:
        """Convert candlestick data into synthetic snapshots in the DB."""
        now = datetime.now(timezone.utc)

        for candle in candles:
            ts = candle.get("end_period_ts") or candle.get("ts") or candle.get("timestamp")
            if ts:
                candle_time = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            else:
                continue

            if (now - candle_time).total_seconds() > 86400:
                continue

            price_data = candle.get("price", candle)
            yes_bid_data = candle.get("yes_bid", {})
            yes_ask_data = candle.get("yes_ask", {})
            no_bid_data = candle.get("no_bid", {})
            no_ask_data = candle.get("no_ask", {})

            last_price = self._candle_price(price_data, "close")
            yes_bid = self._candle_price(yes_bid_data, "close")
            yes_ask = self._candle_price(yes_ask_data, "close")
            no_bid = self._candle_price(no_bid_data, "close")
            no_ask = self._candle_price(no_ask_data, "close")

            self._db.save_snapshot(
                {
                    "ticker": mkt.ticker,
                    "title": mkt.title,
                    "yes_bid": yes_bid or mkt.yes_bid,
                    "yes_ask": yes_ask or mkt.yes_ask,
                    "no_bid": no_bid or mkt.no_bid,
                    "no_ask": no_ask or mkt.no_ask,
                    "volume_24h": mkt.volume_24h,
                    "liquidity": mkt.liquidity,
                    "open_interest": candle.get("open_interest", mkt.open_interest),
                    "last_price": last_price or mkt.last_price,
                },
                captured_at=candle_time.isoformat(),
            )

    @staticmethod
    def _candle_price(data: dict | int | float | None, key: str = "close") -> float | None:
        if data is None:
            return None
        if isinstance(data, (int, float)):
            val = float(data)
            return val / 100.0 if val > 1.0 else val
        if isinstance(data, dict):
            val = data.get(key) or data.get("close") or data.get("price")
            if val is not None:
                val = float(val)
                return val / 100.0 if val > 1.0 else val
        return None
