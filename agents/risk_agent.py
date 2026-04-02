"""
RISK AGENT — determines position sizing using fractional Kelly criterion,
enforces exposure limits, and gates trades that are too risky.

Also computes smart limit prices using orderbook data to minimize slippage.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from models.category_profiles import CategoryProfileLoader
from utils import CandidateMarket, PredictionResult, ResearchResult, TradeDecision

logger = logging.getLogger("kalshi_bot.risk_agent")


class RiskAgent:
    def __init__(self, cfg: Config, db: Database, kalshi: KalshiAPIClient | None = None) -> None:
        self._cfg = cfg
        self._db = db
        self._kalshi = kalshi
        self._profile_loader = CategoryProfileLoader(db)

    def evaluate(
        self,
        prediction: PredictionResult,
        research: ResearchResult,
        candidate: CandidateMarket,
        bankroll: float | None = None,
    ) -> TradeDecision:
        """
        Given a prediction with positive edge, decide:
          - Whether to trade at all
          - Position size (fractional Kelly, adjusted by quality score)
          - Side (yes/no) and order type

        Returns a TradeDecision (approved or not).
        """
        ticker = prediction.ticker
        edge = prediction.edge
        true_prob = prediction.true_probability
        market_prob = prediction.market_probability
        confidence = prediction.confidence
        quality_score = prediction.quality_score
        liquidity = candidate.market.liquidity
        current_bankroll = bankroll or self._cfg.bankroll

        logger.debug("Risk eval: %s | edge=%+.3f | quality=%.0f | bankroll=%.2f", ticker, edge, quality_score, current_bankroll)

        reasons: list[str] = []

        # ── Determine side ──────────────────────────────────────────
        if edge > 0:
            side = "yes"
            entry_price = candidate.market.yes_ask
            p = true_prob
            odds = (1.0 / entry_price) - 1.0 if entry_price > 0 else 0
        else:
            side = "no"
            entry_price = candidate.market.no_ask if candidate.market.no_ask else (1.0 - candidate.market.yes_bid)
            p = 1.0 - true_prob
            odds = (1.0 / entry_price) - 1.0 if entry_price > 0 else 0

        # ── Kelly criterion ─────────────────────────────────────────
        if odds <= 0:
            reasons.append("odds <= 0")
            return self._reject(ticker, reasons, prediction, research)

        kelly = p - (1.0 - p) / odds
        if kelly <= 0:
            # Allow through if quality score is very high despite negative Kelly
            if quality_score >= 60:
                kelly = 0.01  # minimum Kelly for quality override
                reasons.append(f"Kelly negative but quality={quality_score:.0f} overrides")
            else:
                reasons.append(f"Kelly fraction negative ({kelly:.4f})")
                return self._reject(ticker, reasons, prediction, research)

        # Apply fractional Kelly — use per-category base, then apply learned adjustment
        category = candidate.market.category or candidate.market.ticker[:4]
        kelly_fraction = self._get_learned_kelly_fraction(category)
        # Scale Kelly by quality score: quality 50 = 1.0x, quality 100 = 1.5x, quality 30 = 0.6x
        quality_multiplier = max(0.5, min(1.5, quality_score / 50.0))
        # Scale Kelly by time to resolution: short-term = slight boost, long-term = reduction
        time_multiplier = self._time_decay_kelly_multiplier(candidate.market.close_time)
        fraction = kelly * kelly_fraction * quality_multiplier * time_multiplier
        raw_size = fraction * current_bankroll

        # ── Cap at max exposure ─────────────────────────────────────
        max_size = self._cfg.max_exposure_pct * current_bankroll
        size_dollars = min(raw_size, max_size)

        # ── Minimum order check ─────────────────────────────────────
        if size_dollars < self._cfg.min_trade_dollars:
            reasons.append(f"Position size too small (${size_dollars:.2f} < ${self._cfg.min_trade_dollars:.2f})")
            return self._reject(ticker, reasons, prediction, research)

        # ── Liquidity check ─────────────────────────────────────────
        if liquidity > 0 and size_dollars > liquidity * 0.10:
            old_size = size_dollars
            size_dollars = liquidity * 0.10
            reasons.append(
                f"Liquidity-capped: ${old_size:.2f} → ${size_dollars:.2f}"
            )
            if size_dollars < self._cfg.min_trade_dollars:
                reasons.append(f"Size below ${self._cfg.min_trade_dollars:.2f} after liquidity cap")
                return self._reject(ticker, reasons, prediction, research)

        # ── Longshot YES penalty ───────────────────────────────────
        # Research: YES contracts below 15¢ have -41% EV.  Retail takers
        # disproportionately buy YES at longshot prices.  We reduce sizing
        # when buying YES on longshots; NO on longshots is the +EV side.
        mkt_prob = candidate.market.implied_probability
        if side == "yes" and mkt_prob < 0.15:
            extremity = (0.15 - mkt_prob) / 0.15
            penalty = 1.0 - extremity * 0.50  # up to 50% size cut
            old_size = size_dollars
            size_dollars *= penalty
            if abs(old_size - size_dollars) >= 0.25:
                reasons.append(
                    f"Longshot-YES penalty: ${old_size:.2f} → ${size_dollars:.2f} "
                    f"(prob={mkt_prob:.2f})"
                )
            if size_dollars < self._cfg.min_trade_dollars:
                reasons.append("Size below min after longshot-YES penalty")
                return self._reject(ticker, reasons, prediction, research)

        # ── Open interest floor ─────────────────────────────────────
        # Don't let a single trade exceed 5% of open interest — oversizing
        # a thin market moves the price against us and signals our intent.
        open_interest = candidate.market.open_interest
        if open_interest > 0 and size_dollars > open_interest * 0.05:
            old_size = size_dollars
            size_dollars = min(size_dollars, open_interest * 0.05)
            if abs(old_size - size_dollars) >= 0.50:
                reasons.append(
                    f"OI-floored: ${old_size:.2f} → ${size_dollars:.2f} "
                    f"(OI=${open_interest:.0f})"
                )
            if size_dollars < self._cfg.min_trade_dollars:
                reasons.append(f"Size below ${self._cfg.min_trade_dollars:.2f} after OI floor")
                return self._reject(ticker, reasons, prediction, research)

        # ── Correlated position guard ───────────────────────────────
        # If we already have open trades on the same event (same event_ticker),
        # cap total event exposure at 2× the per-trade max to prevent correlated
        # losses from doubling our drawdown on a single outcome.
        event_ticker = candidate.market.event_ticker
        if event_ticker:
            event_exposure = self._get_event_exposure(event_ticker)
            max_event_exposure = self._cfg.max_exposure_pct * current_bankroll * 2.0
            if event_exposure + size_dollars > max_event_exposure:
                allowed = max(0.0, max_event_exposure - event_exposure)
                if allowed < self._cfg.min_trade_dollars:
                    reasons.append(
                        f"Event exposure limit reached for {event_ticker} "
                        f"(existing=${event_exposure:.2f}, cap=${max_event_exposure:.2f})"
                    )
                    return self._reject(ticker, reasons, prediction, research)
                old_size = size_dollars
                size_dollars = allowed
                reasons.append(
                    f"Event-capped [{event_ticker}]: ${old_size:.2f} → ${size_dollars:.2f} "
                    f"(existing event exposure=${event_exposure:.2f})"
                )

        # ── Max open trades check ───────────────────────────────────
        if self._cfg.max_open_trades > 0:
            open_trades = self._db.get_open_trades()
            if len(open_trades) >= self._cfg.max_open_trades:
                reasons.append(f"Max open trades reached ({self._cfg.max_open_trades})")
                return self._reject(ticker, reasons, prediction, research)

        # ── Duplicate position check ────────────────────────────────
        if self._db.has_open_position(ticker):
            reasons.append("Already have open position in this market")
            return self._reject(ticker, reasons, prediction, research)

        # ── Confidence gate (relaxed: quality score is primary filter) ─
        # Low confidence + low quality = reject
        if confidence == "low" and quality_score < 50:
            reasons.append(f"Confidence low and quality={quality_score:.0f}")
            return self._reject(ticker, reasons, prediction, research)

        # ── Calculate contracts ─────────────────────────────────────
        contracts = max(1, int(size_dollars / entry_price)) if entry_price > 0 else 0

        # Prefer limit orders to avoid slippage
        order_type = "limit"
        limit_price = self._compute_smart_limit(
            ticker, side, entry_price, candidate.market
        )

        # ── Spread at entry ─────────────────────────────────────────
        if side == "yes":
            spread_at_entry = max(0.0, (candidate.market.yes_ask or 0) - (candidate.market.yes_bid or 0))
        else:
            spread_at_entry = max(0.0, (candidate.market.no_ask or 0) - (candidate.market.no_bid or 0))

        reasoning = (
            f"Kelly={kelly:.4f}, fractional={fraction:.4f}, "
            f"raw=${raw_size:.2f}, capped=${size_dollars:.2f}, "
            f"contracts={contracts}, side={side}, "
            f"limit_price=${limit_price:.2f}, quality={quality_score:.0f}, "
            f"spread=${spread_at_entry:.3f}"
        )
        if reasons:
            reasoning += f" | notes: {'; '.join(reasons)}"

        logger.debug(
            "  APPROVED: %s %s | $%.2f (%d contracts) @ $%.2f | Kelly=%.4f | quality=%.0f",
            side.upper(),
            ticker,
            size_dollars,
            contracts,
            limit_price,
            kelly,
            quality_score,
        )

        return TradeDecision(
            approved=True,
            ticker=ticker,
            side=side,
            action="buy",
            size_dollars=round(size_dollars, 2),
            size_contracts=contracts,
            order_type=order_type,
            limit_price=round(limit_price, 4),
            reasoning=reasoning,
            prediction=prediction,
            research=research,
            signals=candidate.signals,
            category=candidate.market.category or candidate.market.ticker[:4],
            spread_at_entry=round(spread_at_entry, 4),
        )

    # ------------------------------------------------------------------
    # Rejection helper
    # ------------------------------------------------------------------
    @staticmethod
    def _reject(
        ticker: str,
        reasons: list[str],
        prediction: PredictionResult,
        research: ResearchResult,
    ) -> TradeDecision:
        reason_str = "; ".join(reasons)
        logger.debug("  BLOCKED: %s — %s", ticker, reason_str)
        return TradeDecision(
            approved=False,
            ticker=ticker,
            reasoning=reason_str,
            prediction=prediction,
            research=research,
        )

    # ------------------------------------------------------------------
    # Learned Kelly fraction
    # ------------------------------------------------------------------
    def _get_learned_kelly_fraction(self, category: str = "") -> float:
        """
        Return Kelly fraction for a category.

        Priority:
          1. Category profile base (domain-specific default)
          2. Global learned adjustment from postmortem DB (loss-streak reduction)
          3. Config fallback
        """
        # Category-specific base from profile
        if category:
            profile = self._profile_loader.get_profile(category)
            base = profile.kelly_fraction
        else:
            base = self._cfg.kelly_fraction

        # Apply global learned multiplier if postmortem has been adjusting it
        raw = self._db.get_heuristic("learned_kelly_fraction")
        if raw:
            try:
                learned = max(0.05, min(0.50, float(raw)))
                # Use the more conservative of profile base vs learned value
                return min(base, learned)
            except (ValueError, TypeError):
                pass
        return base

    # ------------------------------------------------------------------
    # Smart limit pricing
    # ------------------------------------------------------------------
    def _compute_smart_limit(
        self,
        ticker: str,
        side: str,
        ask_price: float,
        market: Any,
    ) -> float:
        """
        Compute a limit price that sits inside the spread — closer to
        the bid than the ask — to get better fills.

        Strategy:
          1. Get the orderbook to find best bid and ask
          2. Place our limit at bid + spread * aggression_factor
          3. aggression_factor starts at 0.3 (closer to bid) and
             increases toward 1.0 (at the ask) as our fill rate drops
          4. Never exceed the ask price
          5. Round to the nearest valid tick (1 cent)
        """
        bid_price = market.yes_bid if side == "yes" else market.no_bid

        # If no bid, just use the ask
        if not bid_price or bid_price <= 0:
            return ask_price

        spread = ask_price - bid_price
        if spread <= 0.01:
            # Spread is already 1 cent — no room to improve
            return ask_price

        # Aggression scales with how thin the market is.
        # Low-volume markets need aggressive pricing to get fills at all.
        volume = getattr(market, "volume_24h", 0) or 0
        aggression = self._get_learned_aggression(volume)

        # Place inside the spread
        raw_price = bid_price + spread * aggression

        # Round to nearest cent
        limit_price = round(raw_price, 2)

        # Clamp: never go below bid+1¢ or above ask
        limit_price = max(bid_price + 0.01, min(limit_price, ask_price))

        if abs(limit_price - ask_price) >= 0.01:
            logger.info(
                "  Smart limit: bid=$%.2f ask=$%.2f → limit=$%.2f "
                "(saving $%.2f/contract, aggression=%.2f)",
                bid_price, ask_price, limit_price,
                ask_price - limit_price, aggression,
            )

        return limit_price

    @staticmethod
    def _time_decay_kelly_multiplier(close_time: str | None) -> float:
        """
        Scale Kelly fraction based on days to resolution.

        Short-term trades resolve quickly so capital isn't tied up long;
        long-term trades carry more uncertainty and should be sized smaller.

          < 1 day    → 1.20x (quick resolution, less overnight risk)
          1-3 days   → 1.10x
          3-7 days   → 1.00x (baseline)
          7-14 days  → 0.90x
          14+ days   → 0.80x
        """
        if not close_time:
            return 1.0
        try:
            ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            days = max(0.0, (ct - datetime.now(timezone.utc)).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 1.0

        if days < 1.0:
            return 1.20
        if days < 3.0:
            return 1.10
        if days < 7.0:
            return 1.00
        if days < 14.0:
            return 0.90
        return 0.80

    def _get_event_exposure(self, event_ticker: str) -> float:
        """Sum size_dollars of all open trades sharing the same event_ticker."""
        try:
            open_trades = self._db.get_open_trades()
            return sum(
                t.get("size_dollars", 0) or 0
                for t in open_trades
                if t.get("ticker", "").startswith(event_ticker)
            )
        except Exception:
            return 0.0

    def _get_learned_aggression(self, volume_24h: float = 0) -> float:
        """
        Return aggression factor between 0.5 (moderate) and 1.0 (at the ask).

        Base is volume-driven — thin markets need aggressive pricing to fill:
          volume >= 50k  → 0.65  (liquid, can afford to be patient)
          volume >= 10k  → 0.75
          volume >= 1k   → 0.85
          volume < 1k    → 0.95  (thin — go right to the ask)

        A learned DB override (from postmortem fill-rate analysis) is then
        applied as a delta on top, clamped to [0.5, 1.0].
        """
        if volume_24h >= 50_000:
            base = 0.65
        elif volume_24h >= 10_000:
            base = 0.75
        elif volume_24h >= 1_000:
            base = 0.85
        else:
            base = 0.95

        raw = self._db.get_heuristic("order_aggression")
        if raw:
            try:
                learned = float(raw)
                # Treat the stored value as a target; blend toward it
                base = (base + learned) / 2.0
            except (ValueError, TypeError):
                pass

        return max(0.5, min(1.0, base))
