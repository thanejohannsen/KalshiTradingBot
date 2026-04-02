"""
SCAN AGENT — filters the full Kalshi market list down to actionable candidates.

Starts with lax thresholds and tightens/loosens them over time based on
learned heuristics from the postmortem agent.  Every signal has an
effectiveness score tracked in the database:

  signal_wins_<name>   — how many winning trades originated from this signal
  signal_losses_<name> — how many losing trades originated from this signal
  signal_adj_<name>    — a learned threshold adjustment (+stricter / -looser)

The scan agent loads these at the start of each cycle and adjusts the
threshold used for each signal accordingly.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone, timedelta

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from utils import CandidateMarket, MarketData

logger = logging.getLogger("kalshi_bot.scan_agent")

# Signals that can be suppressed if their loss rate gets too high
LEARNABLE_SIGNALS = [
    "wide_spread",
    "price_move",
    "volume_spike",
    "divergence",
    "new_high_activity",
]


class ScanAgent:
    def __init__(self, cfg: Config, db: Database, kalshi: KalshiAPIClient) -> None:
        self._cfg = cfg
        self._db = db
        self._kalshi = kalshi
        # Learned adjustments — loaded fresh each scan
        self._signal_adjustments: dict[str, float] = {}
        self._suppressed_signals: set[str] = set()

    def _load_learned_thresholds(self) -> None:
        """
        Read heuristics from the DB and compute per-signal adjustments.

        For each signal we track wins/losses.  If a signal's loss rate
        exceeds 75 % over ≥ 4 trades we suppress it entirely.  Otherwise
        we tighten the threshold proportionally to loss rate:
           adjustment = loss_rate * 0.5   (0 = no change, 0.5 = 50 % stricter)
        """
        self._signal_adjustments = {}
        self._suppressed_signals = set()

        for sig in LEARNABLE_SIGNALS:
            wins_raw = self._db.get_heuristic(f"signal_wins_{sig}")
            losses_raw = self._db.get_heuristic(f"signal_losses_{sig}")
            wins = int(wins_raw) if wins_raw and wins_raw.isdigit() else 0
            losses = int(losses_raw) if losses_raw and losses_raw.isdigit() else 0
            total = wins + losses

            if total < 2:
                # Not enough data — keep default (lax) thresholds
                self._signal_adjustments[sig] = 0.0
                continue

            loss_rate = losses / total

            if total >= 4 and loss_rate >= 0.75:
                # This signal is hurting us — suppress it
                self._suppressed_signals.add(sig)
                logger.info(
                    "LEARNED: suppressing signal '%s' (win=%d loss=%d rate=%.0f%%)",
                    sig, wins, losses, loss_rate * 100,
                )
            else:
                # Tighten threshold proportionally
                adj = loss_rate * 0.5
                self._signal_adjustments[sig] = adj
                if adj > 0.05:
                    logger.info(
                        "LEARNED: tightening '%s' by %.0f%% (win=%d loss=%d)",
                        sig, adj * 100, wins, losses,
                    )

        # Also load per-category avoidances and scan priorities
        all_h = self._db.get_all_heuristics()
        self._avoided_categories: set[str] = set()
        self._category_scan_priority: dict[str, float] = {}

        for k, v in all_h.items():
            if k.startswith("avoid_category_") and v == "true":
                cat = k.replace("avoid_category_", "")
                self._avoided_categories.add(cat)
            elif k.startswith("category_scan_priority_"):
                cat = k.replace("category_scan_priority_", "")
                try:
                    self._category_scan_priority[cat] = float(v)
                except (ValueError, TypeError):
                    pass

        if self._avoided_categories:
            logger.info("LEARNED: avoiding categories %s", self._avoided_categories)
        if self._category_scan_priority:
            logger.info("LEARNED: category scan priorities %s", self._category_scan_priority)

        # Load category profitability bias from trade history
        self._category_bias: dict[str, float] = {}
        cat_stats = self._db.get_category_stats()
        for cat, stats in cat_stats.items():
            wins = stats.get("wins", 0) or 0
            losses = stats.get("losses", 0) or 0
            total = wins + losses
            pnl = stats.get("total_pnl", 0) or 0
            if total < 2:
                continue  # Not enough data
            win_rate = wins / total
            # Bias formula: base 1.0, boosted by win rate and P&L direction
            # Win rate > 60% → boost, < 40% → penalize
            # Positive P&L → additional boost
            rate_factor = (win_rate - 0.5) * 2.0  # -1.0 to +1.0
            pnl_factor = 0.1 if pnl > 0 else -0.1 if pnl < 0 else 0
            bias = 1.0 + rate_factor * 0.5 + pnl_factor  # range ~0.1 to 1.6
            bias = max(0.1, min(bias, 2.0))  # clamp
            self._category_bias[cat] = bias
            if abs(bias - 1.0) > 0.05:
                logger.info(
                    "LEARNED: category '%s' bias=%.2f (%dW/%dL, $%.2f P&L)",
                    cat, bias, wins, losses, pnl,
                )

    # ------------------------------------------------------------------
    # Effective thresholds (base from config + learned adjustment)
    # ------------------------------------------------------------------
    def _effective_threshold(self, signal_name: str, base_value: float) -> float:
        """Return the threshold after applying any learned tightening."""
        adj = self._signal_adjustments.get(signal_name, 0.0)
        # Tightening = make the threshold harder to pass
        # For "greater than" signals: increase threshold
        return base_value * (1.0 + adj)

    def scan(self) -> list[CandidateMarket]:
        """
        Fetch all active markets, apply filters, return shortlist.
        Also records snapshots for future baseline comparison.
        """
        logger.info("=== SCAN AGENT: starting market scan ===")
        self._load_learned_thresholds()

        markets = self._kalshi.get_active_markets(max_markets=self._cfg.max_markets)
        logger.info("Total active markets: %d", len(markets))

        candidates: list[CandidateMarket] = []

        # Diagnostic counters
        filter_counts = {
            "avoided_category": 0,
            "low_volume": 0,
            "low_liquidity": 0,
            "time_filter": 0,
            "no_signals": 0,
            "zero_price": 0,
        }

        for mkt in markets:
            # Skip markets with zero price data (not yet active)
            if mkt.last_price == 0 and mkt.yes_bid == 0 and mkt.yes_ask == 0:
                filter_counts["zero_price"] += 1
                # Still record snapshot for future baselines
                self._record_snapshot(mkt)
                continue

            # Skip categories the bot has learned to avoid
            category = self._market_category(mkt)
            if category in self._avoided_categories:
                filter_counts["avoided_category"] += 1
                self._record_snapshot(mkt)
                continue

            # Apply hard filters (with tracking)
            filter_reason = self._check_hard_filters(mkt)
            if filter_reason:
                filter_counts[filter_reason] += 1
                self._record_snapshot(mkt)
                continue

            # Check signals BEFORE saving snapshot so first-run baselines
            # don't mask the current data point
            signals = self._detect_signals(mkt)

            # NOW save the snapshot (after signal detection)
            self._record_snapshot(mkt)

            if not signals:
                filter_counts["no_signals"] += 1
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
        passed_hard = len(markets) - filter_counts["zero_price"] - filter_counts["avoided_category"] - filter_counts["low_volume"] - filter_counts["low_liquidity"] - filter_counts["time_filter"]
        logger.info(
            "Scan filter breakdown: %d total → %d zero_price, %d low_vol, "
            "%d low_liq, %d time, %d avoided_cat, %d passed_hard → %d no_signals, "
            "%d CANDIDATES",
            len(markets),
            filter_counts["zero_price"],
            filter_counts["low_volume"],
            filter_counts["low_liquidity"],
            filter_counts["time_filter"],
            filter_counts["avoided_category"],
            passed_hard,
            filter_counts["no_signals"],
            len(candidates),
        )

        if not candidates and passed_hard > 0:
            # Log some samples of markets that passed hard filters but had no signals
            logger.info("No candidates found. Sampling markets that passed hard filters:")
            sample_count = 0
            for mkt in markets:
                if mkt.last_price == 0 and mkt.yes_bid == 0 and mkt.yes_ask == 0:
                    continue
                if not self._passes_hard_filters(mkt):
                    continue
                baseline = self._db.get_baseline(mkt.ticker, hours=24)
                logger.info(
                    "  sample: %s price=%.2f bid=%.2f ask=%.2f vol=%.0f liq=%.0f "
                    "spread=%.1fc baseline=%s",
                    mkt.ticker[:40],
                    mkt.last_price,
                    mkt.yes_bid,
                    mkt.yes_ask,
                    mkt.volume_24h,
                    mkt.liquidity,
                    mkt.spread * 100,
                    f"price={baseline['avg_price']:.2f}" if baseline else "NONE",
                )
                sample_count += 1
                if sample_count >= 5:
                    break

        for c in candidates:
            cat = self._market_category(c.market)
            bias = self._category_bias.get(cat, 1.0)
            bias_tag = f" [bias={bias:.2f}]" if bias != 1.0 else ""
            logger.info(
                "  → %s | signals=%s | price=%.2f | vol=%.0f | cat=%s%s",
                c.market.ticker,
                c.signals,
                c.market.last_price,
                c.market.volume_24h,
                cat,
                bias_tag,
            )

        # Rank candidates: signals × category_bias × scan_priority × volume_regime
        # category_bias    — learned from win/loss history per category
        # scan_priority    — set by StrategyEvolutionAgent (winners boosted, losers demoted)
        # volume_regime    — mid-tier volume preferred (MMs haven't arb'd the edge)
        candidates.sort(
            key=lambda c: len(c.signals)
                * self._category_bias.get(self._market_category(c.market), 1.0)
                * self._category_scan_priority.get(self._market_category(c.market), 1.0)
                * self._volume_regime_score(c.market.volume_24h),
            reverse=True,
        )

        return candidates

    # ------------------------------------------------------------------
    # Hard filters (must pass ALL)
    # ------------------------------------------------------------------
    @staticmethod
    def _market_category(mkt: MarketData) -> str:
        """Return the best category label for a market."""
        return mkt.category if mkt.category else (mkt.ticker[:4] if mkt.ticker else "")

    def _check_hard_filters(self, mkt: MarketData) -> str | None:
        """Return filter reason string if blocked, or None if passed."""
        if mkt.volume_24h < self._cfg.min_volume_24h:
            return "low_volume"

        # Liquidity check: since liquidity_dollars is often 0 in the API,
        # check for active bid/ask instead (they must both exist and be > 0)
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

        return None

    def _passes_hard_filters(self, mkt: MarketData) -> bool:
        return self._check_hard_filters(mkt) is None

    @staticmethod
    def _volume_regime_score(volume_24h: float) -> float:
        """
        Prefer mid-tier volume markets where MM arbitrage hasn't squeezed
        out all taker edge.  Research shows post-2024 professional MMs
        extract value most efficiently in high-volume markets.

          >= 100k  → 0.60  (MM-dominated, thin edges)
          50k-100k → 0.80  (competitive)
          5k-50k   → 1.20  (sweet spot — liquid enough, not yet arb'd)
          1k-5k    → 1.00  (some edge but fills are harder)
          < 1k     → 0.70  (too thin to trade reliably)
        """
        if volume_24h >= 100_000:
            return 0.60
        if volume_24h >= 50_000:
            return 0.80
        if volume_24h >= 5_000:
            return 1.20
        if volume_24h >= 1_000:
            return 1.00
        return 0.70

    # ------------------------------------------------------------------
    # Anomaly signal detection (need at least one)
    # ------------------------------------------------------------------
    def _detect_signals(self, mkt: MarketData) -> list[str]:
        signals: list[str] = []

        baseline = self._db.get_baseline(mkt.ticker, hours=24)

        # 1. Wide bid-ask spread (learned: may tighten or suppress)
        if "wide_spread" not in self._suppressed_signals:
            threshold = self._effective_threshold(
                "wide_spread", self._cfg.wide_spread_cents
            )
            spread_cents = mkt.spread * 100
            if spread_cents >= threshold:
                signals.append(f"wide_spread({spread_cents:.0f}c)")

        # 2. Price movement vs baseline
        if "price_move" not in self._suppressed_signals:
            threshold = self._effective_threshold(
                "price_move", self._cfg.price_move_threshold_pct
            )
            if baseline and baseline["avg_price"] and baseline["avg_price"] > 0:
                price_change_pct = abs(
                    (mkt.last_price - baseline["avg_price"]) / baseline["avg_price"]
                ) * 100
                if price_change_pct >= threshold:
                    signals.append(f"price_move({price_change_pct:.1f}%)")

        # 3. Volume spike vs baseline
        if "volume_spike" not in self._suppressed_signals:
            threshold = self._effective_threshold(
                "volume_spike", self._cfg.volume_spike_multiplier
            )
            if baseline and baseline["avg_volume"] and baseline["avg_volume"] > 0:
                volume_ratio = mkt.volume_24h / baseline["avg_volume"]
                if volume_ratio >= threshold:
                    signals.append(f"volume_spike({volume_ratio:.1f}x)")

        # 4. Price divergence from rolling average
        if "divergence" not in self._suppressed_signals:
            threshold_cents = self._effective_threshold(
                "divergence", self._cfg.divergence_cents
            )
            if baseline and baseline["avg_price"] is not None:
                divergence_cents = abs(mkt.last_price - baseline["avg_price"]) * 100
                if divergence_cents >= threshold_cents:
                    signals.append(f"divergence({divergence_cents:.0f}c)")

        # 5. New high-activity market (no baseline yet)
        if "new_high_activity" not in self._suppressed_signals:
            threshold = self._effective_threshold(
                "new_high_activity", self._cfg.min_volume_24h * 2
            )
            if not baseline and mkt.volume_24h > threshold:
                signals.append("new_high_activity")

        return signals

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
        first real scan already has baselines for price_move, volume_spike,
        and divergence signals.
        """
        logger.info("=== BOOTSTRAP: seeding baselines for top %d markets ===", top_n)

        markets = self._kalshi.get_active_markets(max_markets=self._cfg.max_markets)
        logger.info("Bootstrap: fetched %d markets, selecting top %d", len(markets), top_n)

        # Rank by volume * liquidity so we pick the most active/liquid markets
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
                # Can't get candlesticks — save a single current snapshot instead
                self._record_snapshot(mkt)
                seeded += 1
                continue

            candles = self._kalshi.get_market_history(
                ticker=mkt.ticker,
                series_ticker=series_ticker,
                period="1h",
            )

            if candles:
                self._candles_to_snapshots(mkt, candles)
                seeded += 1
            else:
                # Fallback: save one snapshot from current market data
                self._record_snapshot(mkt)
                seeded += 1
                failed += 1

            # Throttle to avoid rate limits (0.3s between calls)
            time.sleep(0.3)

            if (i + 1) % 50 == 0:
                logger.info("Bootstrap progress: %d/%d markets processed", i + 1, top_n)

        logger.info(
            "Bootstrap complete: seeded %d markets (%d without candles, used current price)",
            seeded, failed,
        )

    @staticmethod
    def _derive_series_ticker(event_ticker: str) -> str | None:
        """
        Best-effort extraction of series ticker from event ticker.
        Event tickers are typically SERIES-DATEPART, e.g. 'KXBTCD-24DEC31'.
        Series ticker is the prefix before the date segment.
        """
        if not event_ticker:
            return None
        # Split on '-' and take everything before the first segment that
        # looks like a date (starts with digits)
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
            # Extract timestamp — Kalshi uses 'end_period_ts' (unix seconds)
            ts = candle.get("end_period_ts") or candle.get("ts") or candle.get("timestamp")
            if ts:
                candle_time = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            else:
                # Skip candles without timestamps
                continue

            # Only keep candles from the last 24 hours
            if (now - candle_time).total_seconds() > 86400:
                continue

            # Extract price data — candles may nest OHLC under 'price', or be flat
            price_data = candle.get("price", candle)
            yes_bid_data = candle.get("yes_bid", {})
            yes_ask_data = candle.get("yes_ask", {})
            no_bid_data = candle.get("no_bid", {})
            no_ask_data = candle.get("no_ask", {})

            # Use 'close' values from candle, falling back to flat fields
            last_price = self._candle_price(price_data, "close")
            yes_bid = self._candle_price(yes_bid_data, "close")
            yes_ask = self._candle_price(yes_ask_data, "close")
            no_bid = self._candle_price(no_bid_data, "close")
            no_ask = self._candle_price(no_ask_data, "close")

            # Volume from candle represents that period, but for baseline
            # comparison we use the market's current 24h volume
            volume = candle.get("volume", mkt.volume_24h)

            self._db.save_snapshot(
                {
                    "ticker": mkt.ticker,
                    "title": mkt.title,
                    "yes_bid": yes_bid or mkt.yes_bid,
                    "yes_ask": yes_ask or mkt.yes_ask,
                    "no_bid": no_bid or mkt.no_bid,
                    "no_ask": no_ask or mkt.no_ask,
                    "volume_24h": mkt.volume_24h,  # Use market's 24h volume, not candle period volume
                    "liquidity": mkt.liquidity,
                    "open_interest": candle.get("open_interest", mkt.open_interest),
                    "last_price": last_price or mkt.last_price,
                },
                captured_at=candle_time.isoformat(),
            )

    @staticmethod
    def _candle_price(data: dict | int | float | None, key: str = "close") -> float | None:
        """Extract a price from candle data which may be nested OHLC or a flat value."""
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
