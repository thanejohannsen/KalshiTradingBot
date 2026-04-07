"""
POSITION MONITOR AGENT - watches open positions and exits early when warranted.

Runs every cycle after the execution agent's settlement check.  Evaluates
each open position for:

  1. TRAILING STOP     - tracks high-water mark; exits when price drops trail_pct
                         from peak while profitable (default 8%, learnable)
  2. CATASTROPHIC LOSS - exits if position lost >50% of entry cost
  3. EDGE GONE         - the original edge has reversed (near expiry only)
  4. TIME DECAY        - market is approaching expiry, lock in gains or cut losses

Kalshi has no stop/conditional order type.  Resting sell orders below the
bid fill instantly (they are just market sells).  All exit logic runs via
active polling and immediate sell orders.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from utils import MarketData

logger = logging.getLogger("kalshi_bot.position_monitor")

# Defaults — will be overridden by learned values
DEFAULT_TRAIL_PCT = 0.20      # trailing stop: exit when price drops 20% from peak
                              # binary markets swing 10-20% on noise; 8% was way too tight
DEFAULT_EDGE_MIN = 0.03       # exit if edge reversed by 3%+ (was 1% — too sensitive)
DEFAULT_TIME_DECAY_HOURS = 4  # more aggressive exits when < 4h to close
MIN_HOLD_MINUTES = 30         # don't exit on edge reversal until held at least 30 min
TRAIL_PCT_FLOOR = 0.10        # postmortem can't tighten trail below 10%
TRAIL_PCT_CEILING = 0.40      # postmortem can't widen trail above 40%
CATASTROPHIC_LOSS_PCT = 0.50  # exit if position lost >50% of entry cost
MIN_PROFIT_TO_TRAIL = 0.15    # trailing stop only activates when position is 15%+ above entry


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
        self._trail_pct = DEFAULT_TRAIL_PCT
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
            # Skip manually placed trades — bot must not touch them
            if trade.get("manual"):
                continue

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
        current_edge: float | None = None
        hours_left: float | None = None

        # Current value of our position
        if side == "yes":
            current_price = market.yes_bid if market.yes_bid > 0 else market.last_price
            max_gain = (1.0 - entry_price) * contracts
            unrealized_pnl = (current_price - entry_price) * contracts
        else:
            current_price = market.no_bid if market.no_bid > 0 else (1.0 - market.last_price)
            max_gain = (1.0 - entry_price) * contracts
            unrealized_pnl = (current_price - entry_price) * contracts

        # ── CHECK 1: TRAILING STOP ──────────────────────────────────
        # Track the high-water mark (peak price since entry).
        # Once the position has gained enough (MIN_PROFIT_TO_TRAIL above
        # entry), if price drops trail_pct from the peak, exit to lock
        # in gains.  The minimum profit gate prevents the trail from
        # triggering on tiny run-ups that are just noise in binary markets.
        high_water = float(trade.get("high_water_mark") or entry_price)
        if current_price > high_water:
            high_water = current_price
            self._db.update_high_water_mark(trade["id"], high_water)

        trail_drop = 0.0
        # Require the peak to have exceeded entry by MIN_PROFIT_TO_TRAIL
        # before the trailing stop is armed.  This prevents exits on tiny
        # run-ups (e.g., entry $0.62, peak $0.72, 8% trail → exit at $0.66
        # for a pathetic $0.04 profit per contract).
        profit_from_entry = (high_water - entry_price) / entry_price if entry_price > 0 else 0
        trail_armed = profit_from_entry >= MIN_PROFIT_TO_TRAIL

        if trail_armed and current_price > entry_price:
            # Position has had a meaningful run-up — check if price dropped from peak
            trail_drop = (high_water - current_price) / high_water
            if trail_drop >= self._trail_pct:
                profit_captured = current_price - entry_price
                potential = 1.0 - entry_price
                pct_captured = (profit_captured / potential * 100) if potential > 0 else 0
                logger.info(
                    "  TRAILING STOP: %s | peak=$%.2f → now=$%.2f (drop=%.1f%%, trail=%.1f%%) | "
                    "captured $%.2f/contract (%.0f%% of potential)",
                    ticker, high_water, current_price,
                    trail_drop * 100, self._trail_pct * 100,
                    profit_captured, pct_captured,
                )
                return ("trailing_stop", current_price)

        # ── CHECK 2: CATASTROPHIC LOSS ─────────────────────────────
        # If the position has lost more than 50% of its entry cost,
        # salvage remaining value rather than riding to zero.
        if unrealized_pnl < 0 and entry_price > 0:
            loss_pct = abs(unrealized_pnl) / (entry_price * contracts)
            if loss_pct >= CATASTROPHIC_LOSS_PCT:
                logger.info(
                    "  CATASTROPHIC LOSS: %s | entry=$%.2f → now=$%.2f | "
                    "loss=$%.2f (%.0f%% of entry) — cutting losses",
                    ticker, entry_price, current_price,
                    unrealized_pnl, loss_pct * 100,
                )
                return ("catastrophic_loss", current_price)

        # ── CHECK 3: EDGE GONE (near expiry only) ──────────────────
        # Only exit on edge reversal near close.  Far from close, normal
        # price noise on cheap contracts easily reverses the raw edge —
        # a 2c move on a 18c contract flips the sign.  That's noise,
        # not a thesis failure.
        #
        # Minimum hold time prevents entering and exiting in the same cycle.
        original_edge = trade.get("edge") or 0
        if original_edge != 0:
            # Enforce minimum hold time — don't edge-exit a brand-new position
            opened_at = trade.get("opened_at")
            held_long_enough = True
            if opened_at:
                try:
                    t_open = datetime.fromisoformat(
                        str(opened_at).replace("Z", "+00:00")
                    )
                    minutes_held = (datetime.now(timezone.utc) - t_open).total_seconds() / 60
                    held_long_enough = minutes_held >= MIN_HOLD_MINUTES
                except (ValueError, TypeError):
                    pass

            if held_long_enough:
                if side == "yes":
                    current_mid = market.mid_price
                    current_edge = current_mid - entry_price
                else:
                    current_mid = 1.0 - market.mid_price
                    current_edge = current_mid - entry_price

                # Scale the reversal threshold by time remaining:
                #   < 2h  : exit on reversal > 3% (converging but not hair-trigger)
                #   2-12h : exit on reversal > 60% of original edge
                #   > 12h : don't exit on edge reversal at all (too noisy)
                edge_exit_ok = False
                close_time_str_edge = market.close_time
                if close_time_str_edge:
                    try:
                        ct = datetime.fromisoformat(close_time_str_edge.replace("Z", "+00:00"))
                        hrs = (ct - datetime.now(timezone.utc)).total_seconds() / 3600
                        if hrs <= 2:
                            edge_exit_ok = current_edge < -DEFAULT_EDGE_MIN
                        elif hrs <= 12:
                            reversal_threshold = max(DEFAULT_EDGE_MIN, abs(original_edge) * 0.60)
                            edge_exit_ok = current_edge < -reversal_threshold
                    except (ValueError, TypeError):
                        pass

                if edge_exit_ok:
                    logger.info(
                        "  Edge reversed: %s | original_edge=%.3f -> current=%.3f",
                        ticker, original_edge, current_edge,
                    )
                    return ("edge_reversed", current_price)

        # ── CHECK 4: TIME DECAY — lock in gains OR cut losses near expiry
        close_time_str = market.close_time
        if close_time_str:
            try:
                close_time = datetime.fromisoformat(
                    close_time_str.replace("Z", "+00:00")
                )
                now = datetime.now(timezone.utc)
                hours_left = (close_time - now).total_seconds() / 3600

                if 0 < hours_left < DEFAULT_TIME_DECAY_HOURS:
                    # Near expiry with a gain — lock it in with tighter TP
                    if unrealized_pnl > 0 and max_gain > 0:
                        gain_pct = unrealized_pnl / max_gain
                        if gain_pct >= 0.20:
                            logger.info(
                                "  Time decay take-profit: %s | %.1fh left | gain=$%.2f (%.0f%%)",
                                ticker, hours_left,
                                unrealized_pnl, gain_pct * 100,
                            )
                            return ("time_decay_tp", current_price)

                    # Near expiry with a significant loss — cut it.
                    # With < 2h left, a position losing > 40% of entry cost
                    # is unlikely to recover.  With < 1h, cut at 25%.
                    # We can still recoup partial value by selling now rather
                    # than holding to resolution for a total loss.
                    if unrealized_pnl < 0 and entry_price > 0:
                        loss_pct = abs(unrealized_pnl) / (entry_price * contracts)
                        if hours_left < 1.0 and loss_pct >= 0.25:
                            logger.info(
                                "  Time decay loss-cut: %s | %.1fh left | "
                                "loss=$%.2f (%.0f%% of entry) — cutting near expiry",
                                ticker, hours_left,
                                unrealized_pnl, loss_pct * 100,
                            )
                            return ("time_decay_loss", current_price)
                        elif hours_left < 2.0 and loss_pct >= 0.40:
                            logger.info(
                                "  Time decay loss-cut: %s | %.1fh left | "
                                "loss=$%.2f (%.0f%% of entry) — cutting near expiry",
                                ticker, hours_left,
                                unrealized_pnl, loss_pct * 100,
                            )
                            return ("time_decay_loss", current_price)
            except (ValueError, TypeError):
                pass

        # Explain hold decision so monitoring behavior is transparent.
        if trail_armed and current_price > entry_price:
            trail_msg = f"trail ARMED drop={trail_drop * 100:.1f}%/{self._trail_pct * 100:.0f}%"
        elif profit_from_entry > 0:
            trail_msg = f"trail waiting (run-up={profit_from_entry * 100:.0f}%, need {MIN_PROFIT_TO_TRAIL * 100:.0f}%)"
        else:
            trail_msg = f"trail inactive (underwater) trail={self._trail_pct * 100:.0f}%"
        edge_msg = (
            f"edge {current_edge:+.3f}" if current_edge is not None else "edge n/a"
        )
        time_msg = (
            f"{hours_left:.1f}h to expiry" if hours_left is not None else "expiry unknown"
        )
        if log_holds:
            logger.info(
                "  HOLD: %s | px=%.2f entry=%.2f peak=%.2f | pnl=$%.2f | %s | %s | %s",
                ticker,
                current_price,
                entry_price,
                high_water,
                unrealized_pnl,
                trail_msg,
                edge_msg,
                time_msg,
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

        # SAFETY: Verify we actually hold these contracts on Kalshi before
        # attempting to sell. Prevents overselling after restarts or if
        # another sell path already closed the position.
        if self._cfg.is_live:
            held = self._kalshi.get_position_count(ticker, side)
            if held <= 0:
                logger.warning(
                    "SKIP EXIT: %s — Kalshi shows 0 %s contracts held. "
                    "Position may already be closed.",
                    ticker, side,
                )
                return False
            if held < contracts:
                logger.warning(
                    "EXIT CAPPED: %s — DB says x%d but Kalshi only has x%d. Using x%d.",
                    ticker, contracts, held, held,
                )
                contracts = held

        # Don't try to sell at extreme prices — just let the market settle.
        # Selling at 1-2c recovers almost nothing and often won't fill anyway.
        # Selling at 98-99c gives up the last 1-2c for no reason.
        if current_price <= 0.03 or current_price >= 0.97:
            logger.info(
                "  SKIP EXIT: %s | price=%.2f too extreme — letting market settle",
                ticker, current_price,
            )
            return False

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
            "high_water_mark": float(trade.get("high_water_mark") or exit_price),
            "trail_pct_used": self._trail_pct,
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
        """Load learned trailing-stop percentage and exit aggression.

        The trail_pct heuristic is adjusted by the postmortem agent based
        on regret analysis of past trailing-stop exits.
        """
        self._trail_pct = DEFAULT_TRAIL_PCT
        self._exit_aggression = 0.70

        # Separate aggression for exits: default more aggressive than entries
        exit_agg_raw = self._db.get_heuristic("exit_order_aggression")
        if exit_agg_raw:
            try:
                self._exit_aggression = max(0.60, min(0.98, float(exit_agg_raw)))
            except (ValueError, TypeError):
                pass

        # ── Trailing stop percentage learning ───────────────────────
        trail_raw = self._db.get_heuristic("trail_pct")
        if trail_raw:
            try:
                learned = float(trail_raw)
                self._trail_pct = max(TRAIL_PCT_FLOOR, min(TRAIL_PCT_CEILING, learned))
            except (ValueError, TypeError):
                pass

        if log_output:
            logger.info(
                "Exit thresholds: trail=%.1f%% | exit_aggr=%.2f",
                self._trail_pct * 100,
                self._exit_aggression,
            )
