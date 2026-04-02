"""
POSITION MONITOR AGENT — watches open positions and exits early when warranted.

Runs every cycle after the execution agent's settlement check.  Evaluates
each open position for:

  1. TAKE PROFIT  — price moved in our favor past a dynamic threshold
  2. STOP LOSS    — price moved against us past a dynamic threshold
  3. EDGE GONE    — the original edge has evaporated or reversed
  4. TIME DECAY   — market is approaching expiry with shrinking edge

All thresholds are learned from past exits via heuristics:

  exit_tp_threshold     — take-profit trigger (default 40% of max possible gain)
  exit_sl_threshold     — stop-loss trigger   (default 50% of entry cost)
  exit_tp_wins          — how many take-profit exits were ultimately correct
  exit_tp_losses        — take-profit exits where holding would have been better
  exit_sl_wins          — stop-loss exits where the market did go against us
  exit_sl_losses        — stop-loss exits where the market would have recovered
  exit_edge_gone_count  — exits due to edge reversal
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from utils import MarketData

logger = logging.getLogger("kalshi_bot.position_monitor")

# Defaults — will be overridden by learned values
DEFAULT_TP_THRESHOLD = 0.40   # exit when unrealized gain >= 40% of max gain
DEFAULT_SL_THRESHOLD = 0.50   # exit when unrealized loss >= 50% of entry cost
DEFAULT_EDGE_MIN = 0.01       # exit if remaining edge drops below 1%
DEFAULT_TIME_DECAY_HOURS = 4  # more aggressive exits when < 4h to close


class PositionMonitorAgent:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        kalshi: KalshiAPIClient,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._kalshi = kalshi

        # Learned thresholds — loaded each cycle
        self._tp_threshold = DEFAULT_TP_THRESHOLD
        self._sl_threshold = DEFAULT_SL_THRESHOLD
        self._exit_aggression = 0.70

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def monitor_positions(
        self,
        report: bool = True,
        allow_exits: bool = True,
        log_holds: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Check all open trades for early exit opportunities.
        Returns list of trades that were exited early.
        """
        open_trades = self._db.get_open_trades()
        if not open_trades:
            if report:
                logger.info(
                    "Position monitor report: open=0 checked=0 held=0 exits=0 settled_skips=0 errors=0"
                )
            return []

        self._load_learned_thresholds(log_output=report)
        live_position_counts = self._get_live_position_counts()
        live_pending_entry_orders = self._get_live_pending_entry_orders()
        live_positions_total = sum(1 for contracts in live_position_counts.values() if contracts > 0)

        exited: list[dict[str, Any]] = []
        held = 0
        managed_positions = 0
        pending_entries = len(live_pending_entry_orders)
        settled_skips = 0
        errors = 0
        would_exit = 0
        exit_reason_counts: dict[str, int] = {}

        for trade in open_trades:
            ticker = trade["ticker"]
            try:
                monitor_trade, state = self._get_monitorable_trade(
                    trade,
                    live_position_counts,
                    live_pending_entry_orders,
                )
                if state == "pending_entry":
                    continue
                if monitor_trade is None:
                    continue

                managed_positions += 1
                market = self._kalshi.get_market(ticker)

                # Skip settled markets — execution agent handles those
                if market.status in ("settled", "closed", "finalized"):
                    settled_skips += 1
                    continue

                exit_decision = self._evaluate_exit(monitor_trade, market, log_holds=log_holds)
                if exit_decision:
                    reason, current_price = exit_decision
                    if allow_exits:
                        success = self._execute_exit(monitor_trade, market, reason, current_price)
                        if success:
                            monitor_trade["exit_reason"] = reason
                            exited.append(monitor_trade)
                            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
                        else:
                            held += 1
                    else:
                        would_exit += 1
                        held += 1
                else:
                    held += 1

            except Exception as exc:
                logger.debug("Could not check position %s: %s", ticker, exc)
                errors += 1

        if exited:
            logger.info("Exited %d position(s) early this cycle", len(exited))

        if report:
            unmanaged_positions = sum(1 for contracts in live_position_counts.values() if contracts > 0)
            unmanaged_tickers = sorted({ticker for ticker, _side in live_position_counts.keys() if live_position_counts[(ticker, _side)] > 0})
            logger.info(
                "Position monitor report: live_positions=%d pending_entries=%d checked=%d unmanaged=%d held=%d exits=%d would_exit=%d settled_skips=%d errors=%d",
                live_positions_total,
                pending_entries,
                managed_positions - settled_skips,
                unmanaged_positions,
                held,
                len(exited),
                would_exit,
                settled_skips,
                errors,
            )
            if unmanaged_tickers:
                logger.info("Position monitor unmanaged live positions: %s", ", ".join(unmanaged_tickers))
            if exit_reason_counts:
                reason_summary = ", ".join(
                    f"{k}={v}" for k, v in sorted(exit_reason_counts.items())
                )
                logger.info("Position monitor exits by reason: %s", reason_summary)

        return exited

    def _get_monitorable_trade(
        self,
        trade: dict[str, Any],
        live_position_counts: dict[tuple[str, str], int] | None = None,
        live_pending_entry_orders: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Return a monitorable trade record and state.

        For live entry orders, only fully or partially filled orders represent
        actual positions. Resting entry orders should not be counted as open
        positions in the position monitor.
        """
        if not self._cfg.is_live:
            return trade, "position"

        ticker = str(trade.get("ticker", ""))
        side = str(trade.get("side", ""))
        total_contracts = int(trade.get("size_contracts", 0) or 0)
        key = (ticker, side)

        available = 0
        if live_position_counts is not None:
            available = int(live_position_counts.get(key, 0) or 0)

        if available <= 0:
            order_id = str(trade.get("kalshi_order_id") or "")
            if live_pending_entry_orders is not None and order_id in live_pending_entry_orders:
                return None, "pending_entry"
            return None, "no_live_exposure"

        # Consume live position count so multiple DB rows cannot overcount one live position.
        monitor_contracts = min(total_contracts, available) if total_contracts > 0 else available
        if live_position_counts is not None:
            live_position_counts[key] = max(0, available - monitor_contracts)

        filled_trade = dict(trade)
        filled_trade["size_contracts"] = monitor_contracts
        return filled_trade, "position"

    def _get_live_position_counts(self) -> dict[tuple[str, str], int]:
        """Return live held contracts from Kalshi keyed by (ticker, side)."""
        if not self._cfg.is_live:
            return {}

        counts: dict[tuple[str, str], int] = {}
        try:
            positions = self._kalshi.get_positions()
        except Exception as exc:
            logger.debug("Could not fetch live positions: %s", exc)
            return counts

        for pos in positions:
            ticker = str(pos.get("ticker", ""))
            try:
                yes_count = int(pos.get("yes_count", 0) or 0)
            except Exception:
                yes_count = 0
            try:
                no_count = int(pos.get("no_count", 0) or 0)
            except Exception:
                no_count = 0

            if yes_count > 0:
                counts[(ticker, "yes")] = yes_count
            if no_count > 0:
                counts[(ticker, "no")] = no_count

        return counts

    def _get_live_pending_entry_orders(self) -> dict[str, dict[str, Any]]:
        """Return active live buy orders keyed by order_id."""
        if not self._cfg.is_live:
            return {}

        try:
            orders = self._kalshi.list_orders(only_active=True)
        except Exception as exc:
            logger.debug("Could not fetch live orders: %s", exc)
            return {}

        pending_orders: dict[str, dict[str, Any]] = {}
        for order in orders:
            if str(order.get("action", "")).lower() != "buy":
                continue
            order_id = str(order.get("order_id") or "")
            if not order_id:
                continue
            pending_orders[order_id] = order

        return pending_orders

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------
    def _evaluate_exit(
        self,
        trade: dict[str, Any],
        market: MarketData,
        log_holds: bool = True,
    ) -> tuple[str, float] | None:
        """
        Decide whether to exit a position early.
        Returns (reason, current_price) or None to hold.
        """
        ticker = trade["ticker"]
        side = trade["side"]
        entry_price = trade["entry_price"]
        contracts = trade.get("size_contracts", 1)
        spread_at_entry = float(trade.get("spread_at_entry") or 0.0)
        tp_progress: float | None = None
        sl_progress: float | None = None
        current_edge: float | None = None
        hours_left: float | None = None
        has_live_tp = bool(trade.get("tp_order_id"))
        has_live_sl = bool(trade.get("sl_order_id"))

        # Current value of our position
        if side == "yes":
            # We own YES contracts — they're worth the current yes_bid
            current_price = market.yes_bid if market.yes_bid > 0 else market.last_price
            # Max gain = $1 - entry, max loss = entry
            max_gain = (1.0 - entry_price) * contracts
            unrealized_pnl = (current_price - entry_price) * contracts
        else:
            # We own NO contracts — they're worth the current no_bid (= 1 - yes_ask)
            current_price = market.no_bid if market.no_bid > 0 else (1.0 - market.last_price)
            max_gain = (1.0 - entry_price) * contracts
            unrealized_pnl = (current_price - entry_price) * contracts

        # Spread-adjusted cost basis: the bid-ask spread is not a real loss —
        # we immediately "lose" it just by buying, so we exclude it from the
        # stop-loss reference price.  If spread_at_entry=0 (old trades), this
        # is identical to the original logic.
        sl_basis = max(0.01, entry_price - spread_at_entry)
        max_loss = sl_basis * contracts

        # Absolute trigger levels (in per-contract price terms)
        tp_trigger_price = entry_price + self._tp_threshold * (1.0 - entry_price)
        sl_trigger_price = sl_basis - self._sl_threshold * sl_basis
        tp_trigger_price = max(0.0, min(1.0, tp_trigger_price))
        sl_trigger_price = max(0.0, min(1.0, sl_trigger_price))

        # ── CHECK 1: TAKE PROFIT ────────────────────────────────────
        if max_gain > 0 and unrealized_pnl > 0:
            gain_pct = unrealized_pnl / max_gain
            tp_progress = gain_pct
            if gain_pct >= self._tp_threshold:
                if has_live_tp:
                    return None
                logger.info(
                    "  TP trigger: %s | pnl=$%.2f (%.0f%% of max $%.2f) | threshold=%.0f%%",
                    ticker, unrealized_pnl, gain_pct * 100,
                    max_gain, self._tp_threshold * 100,
                )
                return ("take_profit", current_price)

        # ── CHECK 2: STOP LOSS (spread-adjusted) ───────────────────
        # Measure loss from sl_basis (entry minus spread) so the bid-ask
        # spread alone does not trip the stop loss on thin markets.
        adjusted_pnl = (current_price - sl_basis) * contracts
        if max_loss > 0 and adjusted_pnl < 0:
            loss_pct = abs(adjusted_pnl) / max_loss
            sl_progress = loss_pct
            if loss_pct >= self._sl_threshold:
                if has_live_sl:
                    return None
                logger.info(
                    "  SL trigger: %s | loss=$%.2f (%.0f%% of adj-basis $%.2f) | "
                    "threshold=%.0f%% | spread_adj=$%.3f",
                    ticker, adjusted_pnl, loss_pct * 100,
                    max_loss, self._sl_threshold * 100, spread_at_entry,
                )
                return ("stop_loss", current_price)

        # ── CHECK 3: EDGE GONE ──────────────────────────────────────
        original_edge = trade.get("edge") or 0
        if original_edge != 0:
            # Recalculate current edge: how far is current price from our entry
            if side == "yes":
                # We bought YES at entry_price. Current mid = market mid.
                current_mid = market.mid_price
                current_edge = current_mid - entry_price
            else:
                current_mid = 1.0 - market.mid_price
                current_edge = current_mid - entry_price

            # Exit if edge has reversed (we're now on the wrong side)
            if original_edge > 0 and current_edge < -DEFAULT_EDGE_MIN:
                logger.info(
                    "  Edge reversed: %s | original_edge=%.3f → current=%.3f",
                    ticker, original_edge, current_edge,
                )
                return ("edge_reversed", current_price)

        # ── CHECK 4: TIME DECAY — tighten thresholds near expiry ───
        close_time_str = market.close_time
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(
                    close_time_str.replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                hours_left = (close_time - now).total_seconds() / 3600

                if 0 < hours_left < DEFAULT_TIME_DECAY_HOURS:
                    # Near expiry with a loss — cut it (also spread-adjusted)
                    if adjusted_pnl < 0:
                        loss_pct = abs(adjusted_pnl) / max_loss if max_loss > 0 else 0
                        # Tighter stop: 25% loss near expiry
                        if loss_pct >= 0.25:
                            logger.info(
                                "  Time decay exit: %s | %.1fh left | loss=$%.2f (%.0f%%)",
                                ticker, hours_left,
                                unrealized_pnl, loss_pct * 100,
                            )
                            return ("time_decay_stop", current_price)

                    # Near expiry with small gain — lock it in
                    if unrealized_pnl > 0 and max_gain > 0:
                        gain_pct = unrealized_pnl / max_gain
                        # Tighter TP: 20% gain near expiry
                        if gain_pct >= 0.20:
                            logger.info(
                                "  Time decay take-profit: %s | %.1fh left | gain=$%.2f (%.0f%%)",
                                ticker, hours_left,
                                unrealized_pnl, gain_pct * 100,
                            )
                            return ("time_decay_tp", current_price)
            except (ValueError, TypeError):
                pass

        # Explain hold decision so monitoring behavior is transparent.
        tp_msg = (
            f"TP progress {tp_progress * 100:.0f}%/{self._tp_threshold * 100:.0f}%"
            if tp_progress is not None
            else f"TP progress inactive (needs gain)/{self._tp_threshold * 100:.0f}%"
        )
        sl_msg = (
            f"SL progress {sl_progress * 100:.0f}%/{self._sl_threshold * 100:.0f}%"
            if sl_progress is not None
            else f"SL progress inactive (needs loss)/{self._sl_threshold * 100:.0f}%"
        )
        edge_msg = (
            f"edge {current_edge:+.3f}" if current_edge is not None else "edge n/a"
        )
        time_msg = (
            f"{hours_left:.1f}h to expiry" if hours_left is not None else "expiry unknown"
        )
        if log_holds:
            spread_suffix = f" spread=${spread_at_entry:.3f}" if spread_at_entry > 0 else ""
            logger.info(
                "  HOLD: %s | px=%.2f entry=%.2f basis=%.2f | tp_px>=%.2f sl_px<=%.2f | pnl=$%.2f | %s | %s | %s | %s%s",
                ticker,
                current_price,
                entry_price,
                sl_basis,
                tp_trigger_price,
                sl_trigger_price,
                unrealized_pnl,
                tp_msg,
                sl_msg,
                edge_msg,
                time_msg,
                spread_suffix,
            )

        return None  # Hold position

    # ------------------------------------------------------------------
    # Exit execution
    # ------------------------------------------------------------------
    def _execute_exit(
        self,
        trade: dict[str, Any],
        market: MarketData,
        reason: str,
        current_price: float,
    ) -> bool:
        """Sell the position. Returns True on success."""
        ticker = trade["ticker"]
        side = trade["side"]
        contracts = trade.get("size_contracts", 1)
        entry_price = trade["entry_price"]

        self._cancel_trade_brackets(trade)

        # Base quote used to seed the first exit limit order.
        if side == "yes":
            bid_price = market.yes_bid
            ask_price = market.yes_ask
        else:
            bid_price = market.no_bid
            ask_price = market.no_ask

        exit_price = current_price
        if bid_price and ask_price and ask_price > bid_price:
            spread = ask_price - bid_price
            raw_price = bid_price + spread * self._exit_aggression
            exit_price = round(max(bid_price, min(raw_price, ask_price)), 2)

        if self._cfg.is_live:
            ok, filled_price = self._place_and_chase_exit_order(
                ticker=ticker,
                side=side,
                contracts=contracts,
                initial_price=exit_price,
            )
            if not ok:
                logger.error("[LIVE] Exit did not fill in time for %s", ticker)
                return False
            exit_price = filled_price

        # Calculate realized P&L from early exit
        pnl = (exit_price - entry_price) * contracts

        logger.info(
            "EARLY EXIT: %s %s | reason=%s | entry=$%.2f → exit=$%.2f | pnl=$%.2f (%d contracts)",
            side.upper(), ticker, reason, entry_price, exit_price, pnl, contracts,
        )

        # Update trade record in DB
        status = "exited_profit" if pnl > 0 else "exited_loss"
        self._db.resolve_trade(trade["id"], status, round(pnl, 2))

        # Record exit data for learning
        self._record_exit(trade, reason, pnl, exit_price)

        return True

    def _cancel_trade_brackets(self, trade: dict[str, Any]) -> None:
        """Cancel resting TP/SL orders before forcing a manual exit."""
        cancelled_any = False
        for key in ("tp_order_id", "sl_order_id"):
            order_id = trade.get(key)
            if order_id:
                self._kalshi.cancel_order(str(order_id))
                cancelled_any = True
        if cancelled_any:
            self._db.clear_trade_bracket_orders(trade["id"])

    def _place_and_chase_exit_order(
        self,
        ticker: str,
        side: str,
        contracts: int,
        initial_price: float,
    ) -> tuple[bool, float]:
        """Place exit order and aggressively reprice every few seconds until filled."""
        remaining = max(1, int(contracts))
        price = initial_price
        max_reprices = 3
        timeout = max(3, int(self._cfg.exit_order_adjust_seconds))

        for attempt in range(max_reprices + 1):
            try:
                order_result = self._kalshi.place_order(
                    ticker=ticker,
                    side=side,
                    action="sell",
                    count=remaining,
                    price=price,
                    order_type="limit",
                )
                order_id = order_result.get("order_id")
                logger.info(
                    "[LIVE] Exit order placed: %s (order_id=%s, px=%.2f, remaining=%d)",
                    ticker,
                    order_id,
                    price,
                    remaining,
                )
            except Exception as exc:
                logger.error("[LIVE] Exit order failed for %s: %s", ticker, exc)
                return (False, price)

            if not order_id:
                return (False, price)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    status = self._kalshi.get_order_status(order_id)
                    order_status = str(status.get("status", "")).lower()
                    rem = status.get("remaining_count", remaining)
                    try:
                        rem_int = int(rem)
                    except Exception:
                        rem_int = remaining

                    if order_status in ("executed", "filled") or rem_int <= 0:
                        return (True, price)

                    if rem_int < remaining:
                        remaining = rem_int
                except Exception:
                    pass
                time.sleep(1)

            # Timed out on this resting exit order: cancel and reprice more aggressively.
            self._kalshi.cancel_order(order_id)

            if attempt >= max_reprices or remaining <= 0:
                break

            try:
                market = self._kalshi.get_market(ticker)
                bid = market.yes_bid if side == "yes" else market.no_bid
            except Exception:
                bid = 0.0

            # For sell exits, lower price toward bid to increase fill probability.
            next_price = round(max(bid if bid > 0 else 0.01, price - 0.01), 2)
            if next_price == price and bid > 0:
                next_price = round(bid, 2)

            logger.info(
                "[LIVE] Exit order stale after %ds — repricing %s %.2f -> %.2f (remaining=%d)",
                timeout,
                ticker,
                price,
                next_price,
                remaining,
            )
            price = next_price

        return (False, price)

    # ------------------------------------------------------------------
    # Learning — record exit outcomes for threshold tuning
    # ------------------------------------------------------------------
    def _record_exit(
        self,
        trade: dict[str, Any],
        reason: str,
        pnl: float,
        exit_price: float,
    ) -> None:
        """Store exit metadata so we can learn whether exits were good."""
        # Increment exit counter by reason
        count_key = f"exit_{reason}_count"
        raw = self._db.get_heuristic(count_key)
        count = int(raw) + 1 if raw and raw.isdigit() else 1
        self._db.set_heuristic(count_key, str(count))

        # Track cumulative P&L from early exits
        pnl_key = f"exit_{reason}_pnl"
        raw_pnl = self._db.get_heuristic(pnl_key)
        try:
            cum_pnl = float(raw_pnl) + pnl if raw_pnl else pnl
        except (ValueError, TypeError):
            cum_pnl = pnl
        self._db.set_heuristic(pnl_key, f"{cum_pnl:.4f}")

        # Store this specific exit for later postmortem comparison
        # The postmortem agent can check: "if we had held, would the outcome
        # have been better?"
        exit_data = {
            "trade_id": trade["id"],
            "ticker": trade["ticker"],
            "reason": reason,
            "entry_price": trade["entry_price"],
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "side": trade["side"],
        }
        self._db.set_heuristic(
            f"last_exit_{trade['id']}", json.dumps(exit_data)
        )

        logger.info(
            "Exit recorded: %s reason=%s pnl=$%.2f (cumulative %s=$%.2f, count=%d)",
            trade["ticker"], reason, pnl, reason, cum_pnl, count,
        )

    # ------------------------------------------------------------------
    # Threshold learning
    # ------------------------------------------------------------------
    def _load_learned_thresholds(self, log_output: bool = True) -> None:
        """
        Adjust TP/SL thresholds based on whether past exits were good decisions.

        Logic:
          - If take-profit exits are mostly profitable AND we're not leaving
            too much on the table → keep or tighten TP threshold
          - If take-profit exits often miss bigger gains → loosen TP threshold
          - If stop-loss exits prevent bigger losses → keep or tighten SL
          - If stop-loss exits keep cutting winners → loosen SL threshold
        """
        self._tp_threshold = DEFAULT_TP_THRESHOLD
        self._sl_threshold = DEFAULT_SL_THRESHOLD
        self._exit_aggression = 0.70

        # Separate aggression for exits: default more aggressive than entries
        exit_agg_raw = self._db.get_heuristic("exit_order_aggression")
        if exit_agg_raw:
            try:
                self._exit_aggression = max(0.60, min(0.98, float(exit_agg_raw)))
            except (ValueError, TypeError):
                pass

        # ── Take-profit threshold learning ──────────────────────────
        tp_count_raw = self._db.get_heuristic("exit_take_profit_count")
        tp_count = int(tp_count_raw) if tp_count_raw and tp_count_raw.isdigit() else 0
        tp_pnl_raw = self._db.get_heuristic("exit_take_profit_pnl")

        # Also check time-decay TP exits
        td_tp_count_raw = self._db.get_heuristic("exit_time_decay_tp_count")
        td_tp_count = int(td_tp_count_raw) if td_tp_count_raw and td_tp_count_raw.isdigit() else 0

        total_tp = tp_count + td_tp_count

        if total_tp >= 3:
            # Only adjust if we have enough COMPLETED analyses — not just exit counts.
            # exit_tp_analysed_count is incremented by postmortem.analyze_exit_regret()
            # once a market actually settles, so it can't be gamed by open markets.
            tp_analysed_raw = self._db.get_heuristic("exit_tp_analysed_count")
            tp_analysed = int(tp_analysed_raw) if tp_analysed_raw and tp_analysed_raw.isdigit() else 0

            tp_regret_raw = self._db.get_heuristic("exit_tp_regret_count")
            tp_regret = int(tp_regret_raw) if tp_regret_raw and tp_regret_raw.isdigit() else 0

            if tp_analysed > 0:
                regret_rate = tp_regret / tp_analysed
            else:
                regret_rate = None  # not enough data yet

            if regret_rate is not None and regret_rate > 0.60:
                # We're exiting too early too often — raise TP threshold
                self._tp_threshold = min(0.80, DEFAULT_TP_THRESHOLD + 0.10)
                if log_output:
                    logger.info(
                        "LEARNED: TP threshold raised to %.0f%% (regret_rate=%.0f%%, %d/%d analysed)",
                        self._tp_threshold * 100, regret_rate * 100, tp_analysed, total_tp,
                    )
            elif regret_rate is not None and regret_rate < 0.25 and tp_analysed >= 3:
                # Our exits are well-timed — can tighten slightly
                self._tp_threshold = max(0.25, DEFAULT_TP_THRESHOLD - 0.05)
                if log_output:
                    logger.info(
                        "LEARNED: TP threshold lowered to %.0f%% (regret_rate=%.0f%%, %d/%d analysed)",
                        self._tp_threshold * 100, regret_rate * 100, tp_analysed, total_tp,
                    )
            elif regret_rate is None and log_output:
                logger.info(
                    "TP threshold: holding at default %.0f%% — waiting for %d exits to settle "
                    "(%d/%d analysed so far)",
                    self._tp_threshold * 100, 3, tp_analysed, total_tp,
                )

        # ── Stop-loss threshold learning ────────────────────────────
        sl_count_raw = self._db.get_heuristic("exit_stop_loss_count")
        sl_count = int(sl_count_raw) if sl_count_raw and sl_count_raw.isdigit() else 0

        td_sl_count_raw = self._db.get_heuristic("exit_time_decay_stop_count")
        td_sl_count = int(td_sl_count_raw) if td_sl_count_raw and td_sl_count_raw.isdigit() else 0

        total_sl = sl_count + td_sl_count

        if total_sl >= 3:
            # Only adjust if we have enough COMPLETED analyses — not just exit counts.
            # exit_sl_analysed_count is incremented by postmortem.analyze_exit_regret()
            # once a market actually settles, so pending/unsettled exits don't count.
            sl_analysed_raw = self._db.get_heuristic("exit_sl_analysed_count")
            sl_analysed = int(sl_analysed_raw) if sl_analysed_raw and sl_analysed_raw.isdigit() else 0

            sl_regret_raw = self._db.get_heuristic("exit_sl_regret_count")
            sl_regret = int(sl_regret_raw) if sl_regret_raw and sl_regret_raw.isdigit() else 0

            if sl_analysed > 0:
                regret_rate = sl_regret / sl_analysed
            else:
                regret_rate = None  # not enough data yet

            if regret_rate is not None and regret_rate > 0.50:
                # Stopping out too early — give positions more room
                self._sl_threshold = min(0.75, DEFAULT_SL_THRESHOLD + 0.10)
                if log_output:
                    logger.info(
                        "LEARNED: SL threshold raised to %.0f%% (regret_rate=%.0f%%, %d/%d analysed)",
                        self._sl_threshold * 100, regret_rate * 100, sl_analysed, total_sl,
                    )
            elif regret_rate is not None and regret_rate < 0.20 and sl_analysed >= 3:
                # Our stops are saving us money — tighten them
                self._sl_threshold = max(0.30, DEFAULT_SL_THRESHOLD - 0.05)
                if log_output:
                    logger.info(
                        "LEARNED: SL threshold lowered to %.0f%% (regret_rate=%.0f%%, %d/%d analysed)",
                        self._sl_threshold * 100, regret_rate * 100, sl_analysed, total_sl,
                    )
            elif regret_rate is None and log_output:
                logger.info(
                    "SL threshold: holding at default %.0f%% — waiting for exits to settle "
                    "(%d/%d analysed so far)",
                    self._sl_threshold * 100, sl_analysed, total_sl,
                )

        if log_output and (total_tp > 0 or total_sl > 0):
            logger.info(
                "Exit thresholds: TP=%.0f%% (exits=%d) | SL=%.0f%% (exits=%d) | exit_aggr=%.2f",
                self._tp_threshold * 100, total_tp,
                self._sl_threshold * 100, total_sl,
                self._exit_aggression,
            )
