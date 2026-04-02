"""
POSTMORTEM AGENT — runs after a trade resolves as a LOSS.

Analyzes what went wrong and stores learnings in the heuristics table
so future predictions can avoid repeating the same mistakes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import Config
from database import Database
from models.category_profiles import CategoryProfileLoader
from models.claude_client import ClaudeClient

logger = logging.getLogger("kalshi_bot.postmortem_agent")


class PostmortemAgent:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        claude: ClaudeClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._claude = claude

    def analyze(self, trade: dict[str, Any]) -> dict[str, Any]:
        """
        Run a postmortem on a losing trade.

        Checks:
          1. Was the probability estimate significantly off?
          2. Was sentiment directionally wrong?
          3. Was timing incorrect (resolved much faster/slower)?
          4. Was position sizing too aggressive?

        Stores findings in the postmortems table and updates heuristics.
        """
        trade_id = trade["id"]
        ticker = trade["ticker"]

        logger.info("=== POSTMORTEM: Trade #%d (%s) ===", trade_id, ticker)

        # ── Analysis flags ──────────────────────────────────────────
        prob_estimate_off = 0.0
        sentiment_wrong = False
        timing_incorrect = False
        sizing_aggressive = False
        failure_reasons: list[str] = []

        predicted_prob = trade.get("predicted_prob") or 0.5
        market_prob = trade.get("market_prob") or 0.5
        edge = trade.get("edge") or 0.0
        sentiment_score = trade.get("sentiment_score") or 0.0
        side = trade.get("side", "yes")
        size_dollars = trade.get("size_dollars", 0)
        confidence = trade.get("confidence", "unknown")

        # 1. Probability estimate accuracy
        # If we predicted YES (edge > 0) but lost, true prob was likely
        # much lower than estimated
        if side == "yes":
            # We bet YES and lost → event didn't happen → true prob was ~0
            prob_estimate_off = predicted_prob - 0.0
        else:
            # We bet NO and lost → event happened → true prob was ~1
            prob_estimate_off = (1.0 - predicted_prob) - 0.0

        if abs(prob_estimate_off) > 0.10:
            failure_reasons.append(
                f"Probability estimate off by {prob_estimate_off:.1%}"
            )

        # 2. Sentiment misleading?
        if (side == "yes" and sentiment_score > 0) or (
            side == "no" and sentiment_score < 0
        ):
            sentiment_wrong = True
            failure_reasons.append(
                f"Sentiment ({sentiment_score:+.3f}) agreed with position "
                f"but outcome was opposite"
            )

        # 3. Timing check (if resolution timestamps available)
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

        # 4. Sizing check
        bankroll = self._cfg.bankroll
        if bankroll > 0 and size_dollars > bankroll * 0.03 and confidence != "high":
            sizing_aggressive = True
            failure_reasons.append(
                f"Position size ${size_dollars:.2f} was {size_dollars/bankroll:.1%} "
                f"of bankroll on {confidence}-confidence trade"
            )

        # ── Build failure reason summary ────────────────────────────
        if not failure_reasons:
            failure_reasons.append("No clear single cause identified")

        failure_reason = "; ".join(failure_reasons)

        # ── Optional Claude analysis ────────────────────────────────
        corrected_insight = ""
        if self._claude and self._claude.available:
            corrected_insight = self._claude.analyze_loss(trade)

        if not corrected_insight:
            corrected_insight = self._generate_heuristic_insight(
                failure_reasons, trade
            )

        # ── Pattern to avoid ────────────────────────────────────────
        pattern = self._extract_pattern(failure_reasons, trade)

        # ── Store postmortem ────────────────────────────────────────
        pm = {
            "trade_id": trade_id,
            "prob_estimate_off": prob_estimate_off,
            "sentiment_wrong": sentiment_wrong,
            "timing_incorrect": timing_incorrect,
            "sizing_aggressive": sizing_aggressive,
            "failure_reason": failure_reason,
            "corrected_insight": corrected_insight,
            "pattern_to_avoid": pattern,
        }
        pm_id = self._db.insert_postmortem(pm)

        # ── Update heuristics ───────────────────────────────────────
        self._update_heuristics(trade, pm)

        logger.info(
            "Postmortem #%d stored | reasons: %s",
            pm_id,
            failure_reason,
        )
        return pm

    # ------------------------------------------------------------------
    # Heuristic generation (non-Claude fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_heuristic_insight(
        reasons: list[str], trade: dict[str, Any]
    ) -> str:
        insights: list[str] = []
        for r in reasons:
            if "Probability estimate" in r:
                insights.append(
                    "Consider widening the edge threshold to require stronger signals"
                )
            if "Sentiment" in r:
                insights.append(
                    "Sentiment data was misleading — discount sentiment weight "
                    "for similar market categories"
                )
            if "quickly" in r:
                insights.append(
                    "Avoid entering markets that are very close to resolution "
                    "unless edge is extremely large"
                )
            if "Position size" in r:
                insights.append(
                    "Reduce position sizing for medium/low confidence trades"
                )
        return "; ".join(insights) if insights else "Review trading thesis manually"

    @staticmethod
    def _extract_pattern(reasons: list[str], trade: dict[str, Any]) -> str:
        """Extract a short pattern string for the heuristics table."""
        patterns: list[str] = []
        if any("Sentiment" in r for r in reasons):
            patterns.append("sentiment_misleading")
        if any("quickly" in r for r in reasons):
            patterns.append("late_entry")
        if any("Probability" in r for r in reasons):
            patterns.append("overconfident_estimate")
        if any("Position size" in r for r in reasons):
            patterns.append("oversized_position")
        return ",".join(patterns) if patterns else "unknown"

    # ------------------------------------------------------------------
    # Heuristic updates
    # ------------------------------------------------------------------
    def record_outcome(self, trade: dict[str, Any]) -> None:
        """Record win/loss for category profitability tracking.
        Called for ALL resolved trades (wins and losses)."""
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

            # Boost Kelly fraction back toward default on wins
            raw_kelly = self._db.get_heuristic("learned_kelly_fraction")
            if raw_kelly:
                current_kelly = float(raw_kelly)
                new_kelly = min(0.25, current_kelly * 1.10)  # recover 10%
                self._db.set_heuristic("learned_kelly_fraction", f"{new_kelly:.3f}")

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

        # Track cumulative P&L per category
        pnl_key = f"cat_pnl_{category}"
        current_pnl = self._db.get_heuristic(pnl_key)
        try:
            cumulative = float(current_pnl) + pnl if current_pnl else pnl
        except (ValueError, TypeError):
            cumulative = pnl
        self._db.set_heuristic(pnl_key, f"{cumulative:.4f}")

        # ── Closing Line Value (CLV) ────────────────────────────────
        # CLV = how much better (or worse) our entry was vs where the market
        # closed just before resolution.  Positive CLV means we got a better
        # price than the consensus closing line — a sign of real edge, not luck.
        closing_prob = trade.get("closing_prob")
        entry_price = trade.get("entry_price")
        side = trade.get("side", "yes")
        if closing_prob is not None and entry_price is not None:
            try:
                # For YES buys: lower entry price = better (we paid less than close)
                # For NO buys: entry_price is the no-side cost, compare to (1-closing_prob)
                if side == "yes":
                    clv = closing_prob - entry_price   # positive = we got a discount
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

                clv_direction = "+" if clv >= 0 else ""
                logger.info(
                    "CLV: %s %s entry=%.2f close=%.2f CLV=%s%.4f (cat_total=%.4f)",
                    status, trade.get("ticker", ""), entry_price, closing_prob,
                    clv_direction, clv, cum_clv,
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

    def _update_model_performance(self, trade: dict[str, Any]) -> None:
        """
        Increment model_wins_<name> or model_losses_<name> for each sub-model
        that contributed to this trade.  The ensemble uses these counts to
        adjust weights toward models that have been right more often.
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

            # A model is "correct" if its probability agreed with the outcome.
            # For a YES bet: prob > 0.5 and won, or prob <= 0.5 and lost.
            # For a NO bet: prob < 0.5 and won, or prob >= 0.5 and lost.
            if side == "yes":
                model_correct = (model_prob > 0.5) == won
            else:
                model_correct = (model_prob < 0.5) == won

            key = f"model_wins_{model_name}" if model_correct else f"model_losses_{model_name}"
            raw = self._db.get_heuristic(key)
            count = int(raw) + 1 if raw and raw.isdigit() else 1
            self._db.set_heuristic(key, str(count))

        logger.debug("Model performance updated for trade #%d (%s)", trade_id, status)

    def _update_heuristics(self, trade: dict[str, Any], pm: dict[str, Any]) -> None:
        """Store learned patterns so future agents (scan, prediction, risk) can adjust."""
        category = trade.get("category") or trade.get("ticker", "")[:4]

        # ── Track loss count by category ────────────────────────────
        loss_key = f"losses_{category}"
        current = self._db.get_heuristic(loss_key)
        count = int(current) + 1 if current and current.isdigit() else 1
        self._db.set_heuristic(loss_key, str(count))

        # If 3+ losses in this category, flag it for avoidance (lowered from 5)
        if count >= 3:
            self._db.set_heuristic(f"avoid_category_{category}", "true")
            logger.info("Category '%s' now flagged for avoidance (%d losses)", category, count)

        # ── Track per-signal win/loss for scan agent learning ───────
        signals_raw = trade.get("signals")
        signal_names: list[str] = []
        if signals_raw:
            try:
                signal_names = json.loads(signals_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        for sig_full in signal_names:
            # Strip the parameter part: "wide_spread(5c)" → "wide_spread"
            sig_base = sig_full.split("(")[0]
            loss_sig_key = f"signal_losses_{sig_base}"
            raw = self._db.get_heuristic(loss_sig_key)
            sig_count = int(raw) + 1 if raw and raw.isdigit() else 1
            self._db.set_heuristic(loss_sig_key, str(sig_count))

        # ── If sentiment was wrong, discount it for this market type ─
        if pm.get("sentiment_wrong"):
            disc_key = f"sentiment_discount_{category}"
            current_disc = self._db.get_heuristic(disc_key)
            discount = float(current_disc) + 0.05 if current_disc else 0.05
            self._db.set_heuristic(disc_key, str(min(discount, 0.5)))

        # ── Adjust market trust if probability was way off ───────────
        # The model trusts the market price as a starting point. If we're
        # consistently wrong, the market was right and we should trust it more.
        if pm.get("prob_estimate_off") and abs(pm["prob_estimate_off"]) > 0.15:
            raw_trust = self._db.get_heuristic("market_trust")
            current_trust = float(raw_trust) if raw_trust else 0.5
            # Increase trust in the market (our estimates are too far off)
            new_trust = min(0.85, current_trust + 0.03)
            self._db.set_heuristic("market_trust", f"{new_trust:.2f}")
            logger.info(
                "Market trust increased %.2f -> %.2f (estimate was off by %.1f%%)",
                current_trust, new_trust, pm["prob_estimate_off"] * 100,
            )

        # ── Track overall loss streak for aggression dampening ──────
        streak_raw = self._db.get_heuristic("loss_streak")
        streak = int(streak_raw) + 1 if streak_raw and streak_raw.isdigit() else 1
        self._db.set_heuristic("loss_streak", str(streak))
        # After 5 consecutive losses, reduce position sizing aggression
        if streak >= 5:
            raw_kelly = self._db.get_heuristic("learned_kelly_fraction")
            current_kelly = float(raw_kelly) if raw_kelly else 0.25
            new_kelly = max(0.10, current_kelly * 0.85)  # reduce by 15%
            self._db.set_heuristic("learned_kelly_fraction", f"{new_kelly:.3f}")
            logger.info(
                "Loss streak=%d: Kelly fraction reduced %.3f -> %.3f",
                streak, current_kelly, new_kelly,
            )

        # ── Store pattern for lookup ────────────────────────────────
        pattern = pm.get("pattern_to_avoid", "")
        if pattern:
            self._db.set_heuristic(
                f"pattern_{trade['id']}", pattern
            )

        logger.info("Heuristics updated for category '%s' (signals: %s)", category, signal_names)

    # ------------------------------------------------------------------
    # Exit regret analysis — evaluate early-exit decisions
    # ------------------------------------------------------------------
    def analyze_exit_regret(self, trade: dict[str, Any], market_result: Any) -> None:
        """
        After a market settles, check if any early exit on this trade was
        a good or bad decision.  Updates regret counters that the
        PositionMonitorAgent uses to tune its thresholds.

        Called for trades with status 'exited_profit' or 'exited_loss'.
        """
        trade_id = trade["id"]
        ticker = trade["ticker"]

        # Look up the stored exit data
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

        # Calculate what P&L would have been if we held to settlement
        result_price = market_result.last_price if hasattr(market_result, "last_price") else 0
        contracts = trade.get("size_contracts", 1)

        if side == "yes":
            if result_price >= 0.90:  # Settled YES
                hold_pnl = (1.0 - entry_price) * contracts
            else:  # Settled NO
                hold_pnl = -entry_price * contracts
        else:
            if result_price <= 0.10:  # Settled NO
                hold_pnl = (1.0 - entry_price) * contracts
            else:  # Settled YES
                hold_pnl = -entry_price * contracts

        # Was the exit a good decision?
        # Good exit: exit P&L > hold P&L (we avoided a worse outcome)
        # Bad exit (regret): hold P&L > exit P&L (we left money on the table)
        is_regret = hold_pnl > exit_pnl

        if is_regret:
            missed = hold_pnl - exit_pnl
            logger.info(
                "EXIT REGRET: %s | exit_pnl=$%.2f, hold_pnl=$%.2f | "
                "missed $%.2f by exiting (%s)",
                ticker, exit_pnl, hold_pnl, missed, reason,
            )
        else:
            saved = exit_pnl - hold_pnl
            logger.info(
                "EXIT GOOD: %s | exit_pnl=$%.2f, hold_pnl=$%.2f | "
                "saved $%.2f by exiting (%s)",
                ticker, exit_pnl, hold_pnl, saved, reason,
            )

        # Update regret counters based on exit type
        if reason in ("take_profit", "time_decay_tp"):
            regret_key  = "exit_tp_regret_count"
            analysed_key = "exit_tp_analysed_count"
        elif reason in ("stop_loss", "time_decay_stop"):
            regret_key  = "exit_sl_regret_count"
            analysed_key = "exit_sl_analysed_count"
        else:
            regret_key  = "exit_edge_regret_count"
            analysed_key = "exit_edge_analysed_count"

        # Always increment analysed count — this is the denominator that
        # position_monitor uses.  It only moves when a market actually settles,
        # so it can't be inflated by open/pending exits.
        raw_a = self._db.get_heuristic(analysed_key)
        analysed_count = int(raw_a) + 1 if raw_a and raw_a.isdigit() else 1
        self._db.set_heuristic(analysed_key, str(analysed_count))

        if is_regret:
            raw = self._db.get_heuristic(regret_key)
            regret_count = int(raw) + 1 if raw and raw.isdigit() else 1
            self._db.set_heuristic(regret_key, str(regret_count))

        logger.info(
            "Exit regret analysis complete: %s | reason=%s | exit_pnl=$%.2f hold_pnl=$%.2f | "
            "regret=%s | analysed=%d",
            ticker, reason, exit_pnl, hold_pnl,
            "YES" if is_regret else "no", analysed_count,
        )

        # Clean up the stored exit data
        self._db.set_heuristic(f"last_exit_{trade_id}", "")
