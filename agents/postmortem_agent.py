"""
POSTMORTEM AGENT — ML-enhanced trade analysis.

Runs after every resolved trade (wins AND losses).  Uses historical
calibration data from the trained LightGBM model to evaluate:

  1. Was the edge real?  Compare our predicted probability to what the
     calibration model says the true probability was.  If our edge was
     in the same direction as the calibration model's edge, the thesis
     was sound — we just got unlucky.  If it disagreed, we had a bad read.

  2. Did the CalibrationModel outperform?  Track per-model accuracy so
     the ensemble can learn which models to trust.

  3. Was the trade in a historically profitable zone?  Price zone × category
     has known calibration curves — was this trade in a +EV or -EV zone?

  4. Closing Line Value (CLV) — did we get a better price than the market's
     closing consensus?  Positive CLV = real edge, even if we lost.

GUARDRAILS: The old postmortem agent poisoned heuristics by making
unbounded adjustments after loss streaks.  This version has hard floors
and ceilings on every heuristic, and requires more data before acting.

Heuristic adjustment bounds:
  - kelly_fraction:      [0.10, 0.30]  — never go below 10% or above 30% Kelly
  - loss_streak effect:  max 3 reductions before floor
  - category avoidance:  requires 5+ losses AND negative CLV
  - sentiment discount:  capped at 0.30, requires 3+ misleading instances
"""

from __future__ import annotations

import json
import logging
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.claude_client import ClaudeClient

from config import Config
from database import Database
from models.category_profiles import CategoryProfileLoader

logger = logging.getLogger("kalshi_bot.postmortem_agent")

# Hard bounds on heuristic adjustments — prevents runaway reductions
KELLY_FLOOR = 0.10
KELLY_CEILING = 0.30
KELLY_WIN_RECOVERY = 0.05          # +5% absolute on a win
KELLY_LOSS_REDUCTION = 0.02        # -2% absolute per streak step (5+ losses)
MAX_SENTIMENT_DISCOUNT = 0.30
MIN_LOSSES_FOR_AVOIDANCE = 5       # require 5 losses before flagging category
MIN_LOSSES_FOR_SENTIMENT_DISC = 3  # require 3 misleading sentiment trades


class PostmortemAgent:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        claude: "ClaudeClient | None" = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._claude = claude
        # Lazy import — calibration model may not be available
        self._calibration = None

    def _get_calibration(self):
        if self._calibration is None:
            try:
                from models import calibration
                self._calibration = calibration
            except ImportError:
                self._calibration = False  # sentinel: don't retry
        return self._calibration if self._calibration else None

    # ------------------------------------------------------------------
    # Main entry point — called for LOSING trades
    # ------------------------------------------------------------------
    def analyze(self, trade: dict[str, Any]) -> dict[str, Any]:
        """
        Run a postmortem on a losing trade.

        Uses the calibration model to determine whether the edge was real
        (unlucky) or phantom (bad read), and adjusts heuristics accordingly.
        """
        trade_id = trade["id"]
        ticker = trade["ticker"]

        logger.info("=== POSTMORTEM: Trade #%d (%s) ===", trade_id, ticker)

        predicted_prob = trade.get("predicted_prob") or 0.5
        market_prob = trade.get("market_prob") or 0.5
        edge = trade.get("edge") or 0.0
        sentiment_score = trade.get("sentiment_score") or 0.0
        side = trade.get("side", "yes")
        size_dollars = trade.get("size_dollars", 0)
        confidence = trade.get("confidence", "unknown")
        category = trade.get("category") or trade.get("ticker", "")[:6]
        canonical_cat = CategoryProfileLoader._normalise(category)

        failure_reasons: list[str] = []

        # ── 1. Calibration model analysis ──────────────────────────────
        cal = self._get_calibration()
        cal_prob = None
        cal_edge = None
        edge_agreement = None

        if cal and cal.is_available():
            cal_prob = cal.predict_true_probability(
                market_price=market_prob,
                category=canonical_cat,
                spread=trade.get("spread_at_entry", 0) or 0,
                volume_24h=0,  # not stored per-trade
                open_interest=0,
            )
            if cal_prob is not None:
                cal_edge = cal_prob - market_prob
                # Does our edge agree with the calibration model's edge?
                if edge != 0 and cal_edge != 0:
                    edge_agreement = (edge > 0) == (cal_edge > 0)

                if edge_agreement:
                    failure_reasons.append(
                        f"Calibration model agreed with thesis (cal_edge={cal_edge:+.3f}) "
                        f"— likely unlucky, not a bad read"
                    )
                elif edge_agreement is False:
                    failure_reasons.append(
                        f"Calibration model DISAGREED (our_edge={edge:+.3f}, "
                        f"cal_edge={cal_edge:+.3f}) — edge was likely phantom"
                    )

                # Check if trade was in a known -EV price zone
                bucket_stats = cal.get_price_bucket_stats(market_prob)
                if bucket_stats:
                    market_error = bucket_stats.get("market_error", 0)
                    if market_error > 0.03:
                        failure_reasons.append(
                            f"Price zone {market_prob:.0%} has {market_error:.1%} "
                            f"average calibration error — high-mispricing zone"
                        )

        # ── 2. Probability estimate accuracy ───────────────────────────
        if side == "yes":
            prob_estimate_off = predicted_prob  # lost YES → true was ~0
        else:
            prob_estimate_off = 1.0 - predicted_prob  # lost NO → true was ~1

        if prob_estimate_off > 0.20:
            failure_reasons.append(
                f"Probability estimate off by {prob_estimate_off:.1%}"
            )

        # ── 3. Sentiment misleading? ───────────────────────────────────
        sentiment_wrong = False
        if (side == "yes" and sentiment_score > 0.1) or (
            side == "no" and sentiment_score < -0.1
        ):
            sentiment_wrong = True
            failure_reasons.append(
                f"Sentiment ({sentiment_score:+.3f}) agreed with position "
                f"but outcome was opposite"
            )

        # ── 4. Timing check ───────────────────────────────────────────
        timing_incorrect = False
        opened_at = trade.get("opened_at")
        resolved_at = trade.get("resolved_at")
        if opened_at and resolved_at:
            from datetime import datetime
            try:
                t_open = datetime.fromisoformat(opened_at)
                t_resolve = datetime.fromisoformat(resolved_at)
                hours_held = (t_resolve - t_open).total_seconds() / 3600
                if hours_held < 1:
                    timing_incorrect = True
                    failure_reasons.append(
                        f"Market resolved very quickly ({hours_held:.1f}h) — "
                        f"possibly entered too late"
                    )
            except (ValueError, TypeError):
                pass

        # ── 5. Sizing check ────────────────────────────────────���──────
        sizing_aggressive = False
        bankroll = self._cfg.bankroll
        if bankroll > 0 and size_dollars > bankroll * 0.05 and confidence != "high":
            sizing_aggressive = True
            failure_reasons.append(
                f"Position size ${size_dollars:.2f} was {size_dollars/bankroll:.1%} "
                f"of bankroll on {confidence}-confidence trade"
            )

        if not failure_reasons:
            failure_reasons.append("No clear single cause — possible bad luck")

        failure_reason = "; ".join(failure_reasons)

        # ── Build pattern to avoid ─────────────────────────────────────
        pattern = self._extract_pattern(failure_reasons, trade, edge_agreement)

        # ── Store postmortem ───────────────────────────────────────────
        pm = {
            "trade_id": trade_id,
            "prob_estimate_off": prob_estimate_off,
            "sentiment_wrong": sentiment_wrong,
            "timing_incorrect": timing_incorrect,
            "sizing_aggressive": sizing_aggressive,
            "failure_reason": failure_reason,
            "corrected_insight": failure_reason,
            "pattern_to_avoid": pattern,
            "calibration_prob": cal_prob,
            "calibration_edge": cal_edge,
            "edge_agreement": edge_agreement,
        }
        pm_id = self._db.insert_postmortem(pm)

        # ── Update heuristics (with guardrails) ───────────────────────
        self._update_heuristics(trade, pm)

        logger.info(
            "Postmortem #%d stored | reasons: %s | cal_edge=%s | agreement=%s",
            pm_id, failure_reason,
            f"{cal_edge:+.3f}" if cal_edge is not None else "n/a",
            edge_agreement,
        )
        return pm

    # ------------------------------------------------------------------
    # Outcome tracking — called for ALL resolved trades (wins + losses)
    # ------------------------------------------------------------------
    def record_outcome(self, trade: dict[str, Any]) -> None:
        """Record win/loss for category profitability tracking."""
        raw_category = trade.get("category") or trade.get("ticker", "")[:6]
        category = CategoryProfileLoader._normalise(raw_category)
        status = trade.get("status", "")
        pnl = trade.get("pnl", 0) or 0

        # Track wins per category
        if status in ("won", "exited_profit"):
            win_key = f"cat_wins_{category}"
            current = self._db.get_heuristic(win_key)
            count = int(current) + 1 if current and current.isdigit() else 1
            self._db.set_heuristic(win_key, str(count))

            # Reset loss streak on any win
            self._db.set_heuristic("loss_streak", "0")

            # Recover Kelly fraction toward ceiling (bounded)
            raw_kelly = self._db.get_heuristic("learned_kelly_fraction")
            if raw_kelly:
                try:
                    current_kelly = float(raw_kelly)
                    new_kelly = min(KELLY_CEILING, current_kelly + KELLY_WIN_RECOVERY)
                    self._db.set_heuristic("learned_kelly_fraction", f"{new_kelly:.3f}")
                except (ValueError, TypeError):
                    pass

            # Track per-signal wins
            signals_raw = trade.get("signals")
            if signals_raw:
                try:
                    signal_names = json.loads(signals_raw)
                    for sig_full in signal_names:
                        sig_base = sig_full.split("(")[0]
                        win_sig_key = f"signal_wins_{sig_base}"
                        raw = self._db.get_heuristic(win_sig_key)
                        sig_count = int(raw) + 1 if raw and raw.isdigit() else 1
                        self._db.set_heuristic(win_sig_key, str(sig_count))
                except (json.JSONDecodeError, TypeError):
                    pass

            # Remove category avoidance on a win (evidence against avoidance)
            avoid_key = f"avoid_category_{category}"
            if self._db.get_heuristic(avoid_key) == "true":
                self._db.set_heuristic(avoid_key, "false")
                logger.info("Category '%s' avoidance cleared — won a trade", category)

        # Track cumulative P&L per category
        pnl_key = f"cat_pnl_{category}"
        current_pnl = self._db.get_heuristic(pnl_key)
        try:
            cumulative = float(current_pnl) + pnl if current_pnl else pnl
        except (ValueError, TypeError):
            cumulative = pnl
        self._db.set_heuristic(pnl_key, f"{cumulative:.4f}")

        # ── Closing Line Value (CLV) ────────────────────────────────
        closing_prob = trade.get("closing_prob")
        entry_price = trade.get("entry_price")
        side = trade.get("side", "yes")
        if closing_prob is not None and entry_price is not None:
            try:
                if side == "yes":
                    clv = closing_prob - entry_price
                else:
                    closing_no = 1.0 - closing_prob
                    clv = closing_no - entry_price

                clv_key = f"cat_clv_{category}"
                raw_clv = self._db.get_heuristic(clv_key)
                try:
                    cum_clv = float(raw_clv) + clv if raw_clv else clv
                except (ValueError, TypeError):
                    cum_clv = clv
                self._db.set_heuristic(clv_key, f"{cum_clv:.4f}")

                # Track CLV count for averaging
                clv_count_key = f"cat_clv_count_{category}"
                raw_count = self._db.get_heuristic(clv_count_key)
                clv_count = int(raw_count) + 1 if raw_count and raw_count.isdigit() else 1
                self._db.set_heuristic(clv_count_key, str(clv_count))

                logger.info(
                    "CLV: %s %s entry=%.2f close=%.2f CLV=%+.4f (cat_avg=%+.4f over %d)",
                    status, trade.get("ticker", ""), entry_price, closing_prob,
                    clv, cum_clv / clv_count, clv_count,
                )
            except (TypeError, ValueError):
                pass

        logger.info(
            "Outcome recorded: %s category=%s pnl=$%.2f (cumulative=$%.2f)",
            status, category, pnl, cumulative,
        )

        # Update per-model win/loss for ensemble weight learning
        if status in ("won", "lost"):
            self._update_model_performance(trade)

        # ── Calibration model accuracy tracking ─────────────────────
        if status in ("won", "lost"):
            self._track_calibration_accuracy(trade)

    # ------------------------------------------------------------------
    # Per-model performance tracking
    # ------------------------------------------------------------------
    def _update_model_performance(self, trade: dict[str, Any]) -> None:
        """
        Increment model_wins_<name> or model_losses_<name> for each sub-model.
        """
        trade_id = trade.get("id")
        status = trade.get("status", "")
        side = trade.get("side", "yes")
        won = (status == "won")

        if not trade_id:
            return

        model_preds = self._db.get_model_predictions_for_trade(trade_id)
        if not model_preds:
            return

        for mp in model_preds:
            model_name = mp.get("model_name", "")
            model_prob = mp.get("probability", 0.5)
            if not model_name:
                continue

            if side == "yes":
                model_correct = (model_prob > 0.5) == won
            else:
                model_correct = (model_prob < 0.5) == won

            key = f"model_wins_{model_name}" if model_correct else f"model_losses_{model_name}"
            raw = self._db.get_heuristic(key)
            count = int(raw) + 1 if raw and raw.isdigit() else 1
            self._db.set_heuristic(key, str(count))

        logger.debug("Model performance updated for trade #%d (%s)", trade_id, status)

    def _track_calibration_accuracy(self, trade: dict[str, Any]) -> None:
        """Track whether the calibration model's edge prediction was correct."""
        cal = self._get_calibration()
        if not cal or not cal.is_available():
            return

        market_prob = trade.get("market_prob") or 0.5
        status = trade.get("status", "")
        side = trade.get("side", "yes")
        category = CategoryProfileLoader._normalise(
            trade.get("category") or trade.get("ticker", "")[:6]
        )

        cal_prob = cal.predict_true_probability(
            market_price=market_prob,
            category=category,
        )
        if cal_prob is None:
            return

        cal_edge = cal_prob - market_prob
        won = (status == "won")

        # Was the calibration model on the right side?
        if side == "yes":
            cal_correct = (cal_edge > 0) == won
        else:
            cal_correct = (cal_edge < 0) == won

        key = "cal_model_correct" if cal_correct else "cal_model_incorrect"
        raw = self._db.get_heuristic(key)
        count = int(raw) + 1 if raw and raw.isdigit() else 1
        self._db.set_heuristic(key, str(count))

        # Per-category calibration accuracy
        cat_key = f"cal_correct_{category}" if cal_correct else f"cal_incorrect_{category}"
        raw_cat = self._db.get_heuristic(cat_key)
        cat_count = int(raw_cat) + 1 if raw_cat and raw_cat.isdigit() else 1
        self._db.set_heuristic(cat_key, str(cat_count))

    # ------------------------------------------------------------------
    # Heuristic updates (BOUNDED)
    # ------------------------------------------------------------------
    def _update_heuristics(self, trade: dict[str, Any], pm: dict[str, Any]) -> None:
        """Store learned patterns — all adjustments are bounded."""
        category = trade.get("category") or trade.get("ticker", "")[:7]
        canonical = CategoryProfileLoader._normalise(category)

        # ── Track loss count by category ────────────────────────────
        loss_key = f"losses_{canonical}"
        current = self._db.get_heuristic(loss_key)
        count = int(current) + 1 if current and current.isdigit() else 1
        self._db.set_heuristic(loss_key, str(count))

        # Only flag for avoidance if losses AND negative cumulative CLV
        # This prevents flagging categories where we're getting good prices
        # but just having bad luck on outcomes.
        if count >= MIN_LOSSES_FOR_AVOIDANCE:
            clv_raw = self._db.get_heuristic(f"cat_clv_{canonical}")
            clv_count_raw = self._db.get_heuristic(f"cat_clv_count_{canonical}")
            try:
                cum_clv = float(clv_raw) if clv_raw else 0
                clv_count = int(clv_count_raw) if clv_count_raw and clv_count_raw.isdigit() else 0
                avg_clv = cum_clv / clv_count if clv_count > 0 else 0
            except (ValueError, TypeError):
                avg_clv = 0

            if avg_clv < -0.02:  # negative CLV = consistently bad prices
                self._db.set_heuristic(f"avoid_category_{canonical}", "true")
                logger.info(
                    "Category '%s' flagged for avoidance (%d losses, avg_CLV=%+.3f)",
                    canonical, count, avg_clv,
                )
            else:
                logger.info(
                    "Category '%s' has %d losses but CLV=%+.3f — NOT avoiding (CLV is positive)",
                    canonical, count, avg_clv,
                )

        # ── Track per-signal losses ────────────────────────────────
        signals_raw = trade.get("signals")
        if signals_raw:
            try:
                signal_names = json.loads(signals_raw)
                for sig_full in signal_names:
                    sig_base = sig_full.split("(")[0]
                    loss_sig_key = f"signal_losses_{sig_base}"
                    raw = self._db.get_heuristic(loss_sig_key)
                    sig_count = int(raw) + 1 if raw and raw.isdigit() else 1
                    self._db.set_heuristic(loss_sig_key, str(sig_count))
            except (json.JSONDecodeError, TypeError):
                pass

        # ── If sentiment was wrong, discount it (bounded) ──────────
        if pm.get("sentiment_wrong"):
            disc_key = f"sentiment_discount_{canonical}"
            current_disc_raw = self._db.get_heuristic(disc_key)

            # Count how many times sentiment has been misleading
            mislead_key = f"sentiment_misleading_count_{canonical}"
            mislead_raw = self._db.get_heuristic(mislead_key)
            mislead_count = int(mislead_raw) + 1 if mislead_raw and mislead_raw.isdigit() else 1
            self._db.set_heuristic(mislead_key, str(mislead_count))

            if mislead_count >= MIN_LOSSES_FOR_SENTIMENT_DISC:
                discount = float(current_disc_raw) + 0.05 if current_disc_raw else 0.05
                discount = min(discount, MAX_SENTIMENT_DISCOUNT)
                self._db.set_heuristic(disc_key, str(discount))
                logger.info(
                    "Sentiment discount for '%s': %.2f (misleading %d times)",
                    canonical, discount, mislead_count,
                )

        # ── Loss streak tracking (BOUNDED Kelly reduction) ─────────
        streak_raw = self._db.get_heuristic("loss_streak")
        streak = int(streak_raw) + 1 if streak_raw and streak_raw.isdigit() else 1
        self._db.set_heuristic("loss_streak", str(streak))

        if streak >= 5:
            raw_kelly = self._db.get_heuristic("learned_kelly_fraction")
            current_kelly = float(raw_kelly) if raw_kelly else self._cfg.kelly_fraction
            # Reduce by fixed amount, with hard floor
            new_kelly = max(KELLY_FLOOR, current_kelly - KELLY_LOSS_REDUCTION)
            self._db.set_heuristic("learned_kelly_fraction", f"{new_kelly:.3f}")
            if new_kelly < current_kelly:
                logger.info(
                    "Loss streak=%d: Kelly fraction %.3f → %.3f (floor=%.3f)",
                    streak, current_kelly, new_kelly, KELLY_FLOOR,
                )
            else:
                logger.info(
                    "Loss streak=%d: Kelly already at floor %.3f — no further reduction",
                    streak, KELLY_FLOOR,
                )

        # ── Store pattern for lookup ───────────────────────────────
        pattern = pm.get("pattern_to_avoid", "")
        if pattern:
            self._db.set_heuristic(f"pattern_{trade['id']}", pattern)

        logger.info("Heuristics updated for category '%s'", canonical)

    # ------------------------------------------------------------------
    # Pattern extraction (enhanced with calibration awareness)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_pattern(
        reasons: list[str],
        _trade: dict[str, Any],
        edge_agreement: bool | None,
    ) -> str:
        patterns: list[str] = []
        if edge_agreement is False:
            patterns.append("calibration_disagreed")
        if any("Sentiment" in r for r in reasons):
            patterns.append("sentiment_misleading")
        if any("quickly" in r for r in reasons):
            patterns.append("late_entry")
        if any("Probability" in r for r in reasons):
            patterns.append("overconfident_estimate")
        if any("Position size" in r for r in reasons):
            patterns.append("oversized_position")
        if edge_agreement is True:
            patterns.append("unlucky_loss")  # calibration agreed with thesis
        return ",".join(patterns) if patterns else "unknown"

    # ------------------------------------------------------------------
    # Exit regret analysis
    # ------------------------------------------------------------------
    def analyze_exit_regret(self, trade: dict[str, Any], market_result: Any) -> None:
        """
        After a market settles, check if an early exit was a good decision.
        Updates regret counters and adjusts trailing stop percentage.
        """
        trade_id = trade["id"]
        ticker = trade["ticker"]

        exit_raw = self._db.get_heuristic(f"last_exit_{trade_id}")
        if not exit_raw:
            return

        try:
            exit_data = json.loads(exit_raw)
        except (json.JSONDecodeError, TypeError):
            return

        reason = exit_data.get("reason", "")
        exit_pnl = exit_data.get("pnl", 0)
        entry_price = exit_data.get("entry_price", 0)
        side = exit_data.get("side", "yes")

        result_price = market_result.last_price if hasattr(market_result, "last_price") else 0
        contracts = trade.get("size_contracts", 1)

        if side == "yes":
            if result_price >= 0.90:
                hold_pnl = (1.0 - entry_price) * contracts
            else:
                hold_pnl = -entry_price * contracts
        else:
            if result_price <= 0.10:
                hold_pnl = (1.0 - entry_price) * contracts
            else:
                hold_pnl = -entry_price * contracts

        is_regret = hold_pnl > exit_pnl

        if is_regret:
            missed = hold_pnl - exit_pnl
            logger.info(
                "EXIT REGRET: %s | exit_pnl=$%.2f, hold_pnl=$%.2f | missed $%.2f (%s)",
                ticker, exit_pnl, hold_pnl, missed, reason,
            )
        else:
            saved = exit_pnl - hold_pnl
            logger.info(
                "EXIT GOOD: %s | exit_pnl=$%.2f, hold_pnl=$%.2f | saved $%.2f (%s)",
                ticker, exit_pnl, hold_pnl, saved, reason,
            )

        # Map reason to regret/analysed counter keys
        if reason in ("trailing_stop", "take_profit", "time_decay_tp"):
            regret_key = "exit_tp_regret_count"
            analysed_key = "exit_tp_analysed_count"
        elif reason in ("time_decay_loss", "catastrophic_loss"):
            regret_key = "exit_td_loss_regret_count"
            analysed_key = "exit_td_loss_analysed_count"
        elif reason in ("edge_reversed",):
            regret_key = "exit_edge_regret_count"
            analysed_key = "exit_edge_analysed_count"
        else:
            regret_key = "exit_other_regret_count"
            analysed_key = "exit_other_analysed_count"

        raw_a = self._db.get_heuristic(analysed_key)
        analysed_count = int(raw_a) + 1 if raw_a and raw_a.isdigit() else 1
        self._db.set_heuristic(analysed_key, str(analysed_count))

        if is_regret:
            raw = self._db.get_heuristic(regret_key)
            regret_count = int(raw) + 1 if raw and raw.isdigit() else 1
            self._db.set_heuristic(regret_key, str(regret_count))

        logger.info(
            "Exit regret analysis: %s | reason=%s | exit=$%.2f hold=$%.2f | regret=%s | analysed=%d",
            ticker, reason, exit_pnl, hold_pnl,
            "YES" if is_regret else "no", analysed_count,
        )

        # ── Trailing stop learning ──────────────────────────────────
        # Adjust trail_pct based on whether trailing stop exits are regretted.
        # Only tune when we have enough trailing stop data points.
        if reason == "trailing_stop":
            self._adjust_trail_pct(exit_data, is_regret)

        self._db.set_heuristic(f"last_exit_{trade_id}", "")

    # ------------------------------------------------------------------
    # Trailing stop tuning
    # ------------------------------------------------------------------
    _TRAIL_PCT_FLOOR = 0.10
    _TRAIL_PCT_CEILING = 0.40
    _TRAIL_STEP = 0.02          # adjust by 2% per learning step (was 0.5% — too slow)
    _TRAIL_MIN_ANALYSED = 3     # require 3 settled trailing-stop exits before tuning

    def _adjust_trail_pct(self, exit_data: dict, is_regret: bool) -> None:
        """Nudge trail_pct up or down based on regret analysis.

        - If we regret exiting (holding would have been better), the trail
          was too tight → widen it (increase trail_pct).
        - If the exit was good (we saved money vs holding), the trail is
          working → tighten it slightly (decrease trail_pct) to capture
          more gains on future trades.

        Requires at least _TRAIL_MIN_ANALYSED settled trailing-stop exits
        before making any adjustment.
        """
        # Count how many trailing stop exits have been analysed
        ts_analysed_raw = self._db.get_heuristic("exit_trailing_stop_analysed")
        ts_analysed = int(ts_analysed_raw) + 1 if ts_analysed_raw and ts_analysed_raw.isdigit() else 1
        self._db.set_heuristic("exit_trailing_stop_analysed", str(ts_analysed))

        ts_regret_raw = self._db.get_heuristic("exit_trailing_stop_regret")
        ts_regret = int(ts_regret_raw) if ts_regret_raw and ts_regret_raw.isdigit() else 0
        if is_regret:
            ts_regret += 1
            self._db.set_heuristic("exit_trailing_stop_regret", str(ts_regret))

        if ts_analysed < self._TRAIL_MIN_ANALYSED:
            logger.info(
                "Trail learning: waiting for more data (%d/%d analysed)",
                ts_analysed, self._TRAIL_MIN_ANALYSED,
            )
            return

        # Current trail_pct
        trail_raw = self._db.get_heuristic("trail_pct")
        try:
            current_trail = float(trail_raw) if trail_raw else 0.08
        except (ValueError, TypeError):
            current_trail = 0.08

        regret_rate = ts_regret / ts_analysed
        old_trail = current_trail

        if regret_rate > 0.50:
            # Exiting too early more than half the time → widen trail
            current_trail = min(self._TRAIL_PCT_CEILING, current_trail + self._TRAIL_STEP)
        elif regret_rate < 0.25 and ts_analysed >= 5:
            # Exits are well-timed → tighten trail to capture more gains
            current_trail = max(self._TRAIL_PCT_FLOOR, current_trail - self._TRAIL_STEP)

        if current_trail != old_trail:
            self._db.set_heuristic("trail_pct", f"{current_trail:.4f}")
            logger.info(
                "LEARNED: trail_pct %.1f%% → %.1f%% (regret_rate=%.0f%%, %d analysed, %d regretted)",
                old_trail * 100, current_trail * 100,
                regret_rate * 100, ts_analysed, ts_regret,
            )
        else:
            logger.info(
                "Trail learning: trail_pct=%.1f%% unchanged (regret_rate=%.0f%%, %d analysed)",
                current_trail * 100, regret_rate * 100, ts_analysed,
            )
