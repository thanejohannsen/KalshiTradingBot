"""
EXECUTION AGENT — places orders (live) or simulates them (paper),
then monitors open positions for resolution.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from utils import TradeDecision

logger = logging.getLogger("kalshi_bot.execution_agent")

DEFAULT_TP_THRESHOLD = 0.40
DEFAULT_SL_THRESHOLD = 0.50


class ExecutionAgent:
    def __init__(self, cfg: Config, db: Database, kalshi: KalshiAPIClient) -> None:
        self._cfg = cfg
        self._db = db
        self._kalshi = kalshi
        self._counted_orders: set[str] = set()  # prevent double-counting fills

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------
    def execute(self, decision: TradeDecision) -> int | None:
        """
        Execute an approved trade decision.
        Returns the trade ID in the database, or None on failure.
        """
        if not decision.approved:
            logger.warning("execute() called on unapproved decision for %s", decision.ticker)
            return None

        pred = decision.prediction
        research = decision.research

        trade_record = {
            "ticker": decision.ticker,
            "side": decision.side,
            "action": decision.action,
            "entry_price": decision.limit_price,
            "size_dollars": decision.size_dollars,
            "size_contracts": decision.size_contracts,
            "order_type": decision.order_type,
            "thesis": decision.reasoning,
            "predicted_prob": pred.true_probability if pred else None,
            "market_prob": pred.market_probability if pred else None,
            "edge": pred.edge if pred else None,
            "sentiment_score": research.sentiment_score if research else None,
            "narrative": research.narrative_summary if research else None,
            "confidence": pred.confidence if pred else None,
            "signals": json.dumps(decision.signals) if decision.signals else None,
            "category": decision.category,
            "spread_at_entry": decision.spread_at_entry,
            "status": "open",
            "kalshi_order_id": None,
        }

        if self._cfg.is_live:
            return self._execute_live(decision, trade_record)
        else:
            return self._execute_paper(decision, trade_record)

    def _execute_paper(self, decision: TradeDecision, trade_record: dict) -> int:
        """Paper trading — just record the trade."""
        logger.info(
            "[PAPER] %s %s %s | $%.2f (%d contracts) @ $%.4f",
            decision.action.upper(),
            decision.side.upper(),
            decision.ticker,
            decision.size_dollars,
            decision.size_contracts,
            decision.limit_price,
        )
        trade_id = self._db.insert_trade(trade_record)
        self._save_model_votes(trade_id, decision)
        logger.info("[PAPER] Trade recorded with ID %d", trade_id)
        return trade_id

    def _execute_live(self, decision: TradeDecision, trade_record: dict) -> int | None:
        """Live trading — place a real order on Kalshi."""
        logger.info(
            "[LIVE] Placing %s %s %s | $%.2f (%d contracts) @ $%.4f",
            decision.action.upper(),
            decision.side.upper(),
            decision.ticker,
            decision.size_dollars,
            decision.size_contracts,
            decision.limit_price,
        )

        try:
            order_result = self._kalshi.place_order(
                ticker=decision.ticker,
                side=decision.side,
                action=decision.action,
                count=decision.size_contracts,
                price=decision.limit_price,
                order_type=decision.order_type,
            )

            order_id = order_result.get("order_id")
            trade_record["kalshi_order_id"] = order_id
            trade_record["status"] = "open"

            trade_id = self._db.insert_trade(trade_record)
            self._save_model_votes(trade_id, decision)
            logger.info(
                "[LIVE] Order placed: kalshi_id=%s, db_id=%d",
                order_id,
                trade_id,
            )
            return trade_id

        except Exception as exc:
            logger.error("[LIVE] Order placement failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------
    def import_untracked_positions(self) -> int:
        """
        Import any positions held on Kalshi that have no matching DB trade.
        Called at startup so the bot tracks everything it owns, even positions
        placed manually or before the DB was initialized.
        Returns count of trades imported.
        """
        if not self._cfg.is_live:
            return 0
        try:
            positions = self._kalshi.get_positions()
        except Exception as exc:
            logger.warning("Could not fetch positions for import: %s", exc)
            return 0

        imported = 0
        for pos in positions:
            ticker = pos.get("ticker", "")
            yes_count = pos.get("yes_count") or 0
            no_count  = pos.get("no_count") or 0
            if not ticker or (yes_count == 0 and no_count == 0):
                continue

            # Check if any open DB trade exists for this ticker
            if self._db.has_open_position(ticker):
                continue

            # Find the most recent buy order for this ticker to get entry price
            side = "yes" if yes_count > 0 else "no"
            contracts = yes_count if side == "yes" else no_count
            entry_price = 0.0
            try:
                orders = self._kalshi.list_orders(limit=200)
                buy_orders = [
                    o for o in orders
                    if o.get("ticker") == ticker
                    and o.get("side") == side
                    and o.get("action") == "buy"
                    and o.get("status") in ("executed", "filled")
                ]
                if buy_orders:
                    entry_price = buy_orders[-1].get("price") or 0.0
            except Exception:
                pass

            # Fall back to current market price if no order found
            if entry_price <= 0:
                try:
                    market = self._kalshi.get_market(ticker)
                    entry_price = market.yes_ask if side == "yes" else (market.no_ask or 0.0)
                except Exception:
                    entry_price = 0.01

            trade_record = {
                "ticker": ticker,
                "side": side,
                "action": "buy",
                "entry_price": entry_price,
                "size_dollars": round(contracts * entry_price, 2),
                "size_contracts": contracts,
                "order_type": "limit",
                "thesis": "Imported from Kalshi API at startup",
                "predicted_prob": None,
                "market_prob": None,
                "edge": None,
                "sentiment_score": None,
                "narrative": None,
                "confidence": None,
                "signals": None,
                "category": "",
                "status": "open",
                "kalshi_order_id": None,
                "entry_filled_at": datetime.now(timezone.utc).isoformat(),
            }
            trade_id = self._db.insert_trade(trade_record)
            logger.info(
                "Imported untracked position: trade #%d %s %s x%d @ $%.4f",
                trade_id, ticker, side, contracts, entry_price,
            )
            imported += 1

        return imported

    def reconcile_with_kalshi(self) -> int:
        """
        Use the Kalshi API as source of truth to clean up the DB.

        A DB trade marked 'open' should stay open only if:
          (a) Kalshi shows we currently hold contracts in that market, OR
          (b) Kalshi shows a pending (resting) order for that order_id.

        Anything else is cancelled — the order expired, was rejected, or
        was never filled and Kalshi already cleaned it up.

        Returns count of trades cancelled.
        """
        if not self._cfg.is_live:
            return 0

        open_trades = self._db.get_open_trades()
        if not open_trades:
            return 0

        try:
            # Tickers where we currently hold contracts
            positions = self._kalshi.get_positions()
            held_tickers: set[str] = {
                p["ticker"] for p in positions
                if (p.get("yes_count") or 0) > 0 or (p.get("no_count") or 0) > 0
            }

            # Order IDs that are still resting/pending on Kalshi
            active_orders = self._kalshi.list_orders(only_active=True)
            active_order_ids: set[str] = {
                o["order_id"] for o in active_orders if o.get("order_id")
            }
        except Exception as exc:
            logger.warning("Reconciliation skipped — could not fetch Kalshi state: %s", exc)
            return 0

        cancelled = 0
        for trade in open_trades:
            ticker = trade["ticker"]
            order_id = trade.get("kalshi_order_id") or ""
            trade_id = trade["id"]

            in_positions = ticker in held_tickers
            in_orders    = order_id in active_order_ids

            if not in_positions and not in_orders:
                self._db.cancel_trade(trade_id)
                logger.info(
                    "Reconcile: trade #%d %s not in Kalshi positions or orders — cancelled",
                    trade_id, ticker,
                )
                cancelled += 1

        if cancelled:
            logger.info("Reconciliation cancelled %d ghost trade(s)", cancelled)
        return cancelled

    def monitor_open_trades(self) -> list[dict[str, Any]]:
        """
        Check all open trades for resolution.
        Also checks order fill status and learns from fill rates.
        Returns list of resolved trades (newly updated).
        """
        # Always reconcile against Kalshi API first so the DB reflects reality
        self.reconcile_with_kalshi()

        open_trades = self._db.get_open_trades()
        if not open_trades:
            return []

        resolved: list[dict[str, Any]] = []

        for trade in open_trades:
            ticker = trade["ticker"]
            order_id = trade.get("kalshi_order_id")

            # Check order fill status for live orders
            status_info: dict[str, Any] | None = None
            if order_id and self._cfg.is_live:
                status_info = self._check_fill_status(trade, order_id)
                self._maybe_reprice_stale_buy_order(trade, status_info)

            bracket_resolved = self._sync_entry_and_brackets(trade, status_info)
            if bracket_resolved:
                resolved.append(bracket_resolved)
                continue

            try:
                market = self._kalshi.get_market(ticker)

                # Check if market has settled
                if market.status in ("settled", "closed", "finalized"):
                    pnl = self._calculate_pnl(trade, market)
                    status = "won" if pnl > 0 else "lost"

                    # Closing line = last snapshot price before settlement (for CLV)
                    closing_prob = self._db.get_latest_snapshot_price(ticker)
                    self._db.resolve_trade(trade["id"], status, pnl, closing_prob=closing_prob)
                    trade["status"] = status
                    trade["pnl"] = pnl
                    resolved.append(trade)

                    # Track signal wins for scan-agent learning
                    if status == "won":
                        self._record_signal_wins(trade)

                    logger.info(
                        "Trade %d resolved: %s | %s | P&L=$%.2f",
                        trade["id"],
                        ticker,
                        status.upper(),
                        pnl,
                    )

            except Exception as exc:
                logger.warning("Could not check market %s: %s", ticker, exc)

        return resolved

    def _sync_entry_and_brackets(
        self,
        trade: dict[str, Any],
        status_info: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Detect entry fills, place live stop protection, and resolve stop exits."""
        if not self._cfg.is_live or trade.get("action") != "buy":
            return None

        entry_status = str((status_info or {}).get("status", "")).lower()
        if not trade.get("entry_filled_at") and entry_status in ("executed", "filled"):
            self._db.mark_trade_entry_filled(trade["id"])
            trade["entry_filled_at"] = datetime.utcnow().isoformat()
            self._ensure_exit_brackets(trade)

        if trade.get("entry_filled_at") and not (trade.get("tp_order_id") or trade.get("sl_order_id")):
            self._ensure_exit_brackets(trade)

        if trade.get("tp_order_id") or trade.get("sl_order_id"):
            return self._reconcile_exit_brackets(trade)

        return None

    def _calculate_pnl(self, trade: dict, market: Any) -> float:
        """
        Calculate P&L for a resolved trade.

        In Kalshi binary markets:
          - YES contracts pay $1 if the event happens, $0 otherwise
          - NO contracts pay $1 if the event doesn't happen, $0 otherwise
        """
        entry_price = trade["entry_price"]
        contracts = trade["size_contracts"]
        side = trade["side"]

        # Determine outcome: did the market resolve YES or NO?
        result_price = market.last_price  # 1.0 for YES, 0.0 for NO after settlement

        if side == "yes":
            if result_price >= 0.90:  # Resolved YES
                pnl = (1.0 - entry_price) * contracts
            else:  # Resolved NO
                pnl = -entry_price * contracts
        else:  # side == "no"
            if result_price <= 0.10:  # Resolved NO
                pnl = (1.0 - entry_price) * contracts
            else:  # Resolved YES
                pnl = -entry_price * contracts

        return round(pnl, 2)

    def _record_signal_wins(self, trade: dict) -> None:
        """Increment signal_wins_<name> for each signal that contributed to a win."""
        signals_raw = trade.get("signals")
        if not signals_raw:
            return
        try:
            signal_names = json.loads(signals_raw)
        except (json.JSONDecodeError, TypeError):
            return
        for sig_full in signal_names:
            sig_base = sig_full.split("(")[0]
            key = f"signal_wins_{sig_base}"
            raw = self._db.get_heuristic(key)
            count = int(raw) + 1 if raw and raw.isdigit() else 1
            self._db.set_heuristic(key, str(count))

    # ------------------------------------------------------------------
    # Fill tracking & execution learning
    # ------------------------------------------------------------------
    def _check_fill_status(self, trade: dict, order_id: str) -> dict[str, Any] | None:
        """Check if a live order has filled and record fill data for learning."""
        if order_id in self._counted_orders:
            return None  # already counted this order
        try:
            status = self._kalshi.get_order_status(order_id)
            order_status = status.get("status", "")
            remaining = status.get("remaining_count", 0)

            if order_status in ("executed", "filled"):
                self._counted_orders.add(order_id)
                self._record_fill(trade, filled=True, partial=False)
            elif order_status in ("canceled", "cancelled", "expired"):
                self._counted_orders.add(order_id)
                self._record_fill(trade, filled=False, partial=False)
                logger.info(
                    "Order %s for %s did NOT fill (status=%s) — marking cancelled",
                    order_id, trade["ticker"], order_status,
                )
                self._db.cancel_trade(trade["id"])
            elif remaining > 0 and remaining < trade.get("size_contracts", 0):
                self._counted_orders.add(order_id)
                self._record_fill(trade, filled=True, partial=True)
            return status
        except Exception as exc:
            # 404 = order no longer exists on Kalshi (expired, cancelled server-side,
            # or cleaned up). Treat as cancelled so it stops polluting open trades.
            exc_str = str(exc)
            if "404" in exc_str or "not_found" in exc_str:
                self._counted_orders.add(order_id)
                self._db.cancel_trade(trade["id"])
                logger.info(
                    "Order %s for %s not found on Kalshi (404) — marking cancelled",
                    order_id, trade["ticker"],
                )
            else:
                logger.debug("Could not check fill status for %s: %s", order_id, exc)
            return None

    def _maybe_reprice_stale_buy_order(
        self,
        trade: dict[str, Any],
        status: dict[str, Any] | None,
    ) -> None:
        """Cancel/replace stale buy orders so they don't rest indefinitely."""
        if not status:
            return
        if trade.get("action") != "buy":
            return

        order_id = trade.get("kalshi_order_id")
        if not order_id:
            return

        order_status = str(status.get("status", "")).lower()
        if order_status in ("executed", "filled", "canceled", "expired"):
            return

        opened_at = trade.get("opened_at")
        if not opened_at:
            return

        try:
            opened_dt = datetime.fromisoformat(opened_at)
            age_seconds = (datetime.now() - opened_dt).total_seconds()
        except Exception:
            return

        if age_seconds < self._cfg.buy_order_adjust_seconds:
            return

        remaining = status.get("remaining_count", trade.get("size_contracts", 0))
        try:
            remaining_count = int(remaining)
        except Exception:
            remaining_count = int(trade.get("size_contracts", 0) or 0)
        if remaining_count <= 0:
            return

        side = trade.get("side", "yes")
        old_price = float(trade.get("entry_price") or 0.0)

        try:
            market = self._kalshi.get_market(trade["ticker"])
        except Exception as exc:
            logger.debug("Could not fetch market for stale repricing %s: %s", trade.get("ticker"), exc)
            return

        ask_price = market.yes_ask if side == "yes" else market.no_ask
        if ask_price <= 0:
            return

        # Increase bid price by 1c toward the ask so stale entries eventually fill.
        new_price = min(ask_price, round(old_price + 0.01, 2))
        if new_price <= old_price:
            return

        if not self._kalshi.cancel_order(order_id):
            return

        try:
            replacement = self._kalshi.place_order(
                ticker=trade["ticker"],
                side=side,
                action="buy",
                count=remaining_count,
                price=new_price,
                order_type="limit",
            )
            new_order_id = replacement.get("order_id")
            if new_order_id:
                self._db.update_trade_order(trade["id"], new_order_id, new_price)
                logger.info(
                    "Repriced stale BUY order %s: %s %.2f -> %.2f (age=%.0fs, remaining=%d)",
                    trade["id"],
                    trade["ticker"],
                    old_price,
                    new_price,
                    age_seconds,
                    remaining_count,
                )
        except Exception as exc:
            logger.warning("Failed to place replacement BUY order for %s: %s", trade.get("ticker"), exc)

    def _record_fill(self, trade: dict, filled: bool, partial: bool) -> None:
        """Record fill/miss and update the aggression heuristic."""
        # Increment fill counters
        if filled:
            key = "order_fills"
        else:
            key = "order_misses"

        raw = self._db.get_heuristic(key)
        count = int(raw) + 1 if raw and raw.isdigit() else 1
        self._db.set_heuristic(key, str(count))

        if partial:
            raw_p = self._db.get_heuristic("order_partials")
            count_p = int(raw_p) + 1 if raw_p and raw_p.isdigit() else 1
            self._db.set_heuristic("order_partials", str(count_p))

        # Recompute aggression based on fill rate
        fills_raw = self._db.get_heuristic("order_fills")
        misses_raw = self._db.get_heuristic("order_misses")
        fills = int(fills_raw) if fills_raw and fills_raw.isdigit() else 0
        misses = int(misses_raw) if misses_raw and misses_raw.isdigit() else 0
        total = fills + misses

        if total >= 3:
            fill_rate = fills / total
            # Target: 70% fill rate
            # If fill rate > 80%: we're paying too much, decrease aggression
            # If fill rate < 50%: we're missing fills, increase aggression
            current_raw = self._db.get_heuristic("order_aggression")
            current_agg = float(current_raw) if current_raw else 0.4

            if fill_rate > 0.80:
                # Too aggressive — become more patient
                new_agg = max(0.30, current_agg - 0.05)
            elif fill_rate < 0.50:
                # Too patient — become more aggressive
                new_agg = min(0.95, current_agg + 0.05)
            else:
                new_agg = current_agg  # Goldilocks zone

            if new_agg != current_agg:
                self._db.set_heuristic("order_aggression", f"{new_agg:.2f}")
                logger.info(
                    "LEARNED: fill_rate=%.0f%% (%d/%d) → aggression %.2f → %.2f",
                    fill_rate * 100, fills, total, current_agg, new_agg,
                )

    def _ensure_exit_brackets(self, trade: dict[str, Any]) -> None:
        """Place a resting stop-loss order after entry fill."""
        if trade.get("tp_order_id") or trade.get("sl_order_id"):
            return

        contracts = int(trade.get("size_contracts", 0) or 0)
        if contracts <= 0:
            return

        entry_price = float(trade.get("entry_price") or 0.0)
        tp_threshold, sl_threshold = self._get_exit_thresholds()
        tp_price = round(min(0.99, max(0.01, entry_price + tp_threshold * (1.0 - entry_price))), 2)
        sl_price = round(min(0.99, max(0.01, entry_price - sl_threshold * entry_price)), 2)

        sl_order_id: str | None = None
        try:
            sl_order = self._kalshi.place_order(
                ticker=trade["ticker"],
                side=trade["side"],
                action="sell",
                count=contracts,
                price=sl_price,
                order_type="limit",
            )
            sl_order_id = sl_order.get("order_id")
        except Exception as exc:
            if sl_order_id:
                self._kalshi.cancel_order(sl_order_id)
            logger.warning("Failed to place live stop-loss for %s: %s", trade.get("ticker"), exc)
            return

        if not sl_order_id:
            return

        self._db.set_trade_bracket_orders(
            trade_id=trade["id"],
            tp_order_id=None,
            tp_price=tp_price,
            sl_order_id=sl_order_id,
            sl_price=sl_price,
        )
        trade["tp_price"] = tp_price
        trade["sl_order_id"] = sl_order_id
        trade["sl_price"] = sl_price

        logger.info(
            "[LIVE] Stop-loss armed for %s: SL order=%s @ %.2f | TP trigger=%.2f",
            trade["ticker"],
            sl_order_id,
            sl_price,
            tp_price,
        )

    def _reconcile_exit_brackets(self, trade: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve trade when a live stop order fills, or re-arm if it disappears."""
        tp_order_id = trade.get("tp_order_id")
        sl_order_id = trade.get("sl_order_id")

        tp_status = None
        sl_status = None
        import requests
        try:
            tp_status = self._kalshi.get_order_status(tp_order_id) if tp_order_id else None
        except requests.HTTPError as exc:
            logger.warning(f"Failed to get TP order status for {tp_order_id}: {exc}")
        try:
            sl_status = self._kalshi.get_order_status(sl_order_id) if sl_order_id else None
        except requests.HTTPError as exc:
            logger.warning(f"Failed to get SL order status for {sl_order_id}: {exc}")

        tp_done = self._is_filled_order_status(tp_status)
        sl_done = self._is_filled_order_status(sl_status)
        if not tp_done and not sl_done:
            tp_terminal = self._is_terminal_order_status(tp_status) if tp_status else True
            sl_terminal = self._is_terminal_order_status(sl_status) if sl_status else False
            if tp_terminal and sl_terminal:
                self._db.clear_trade_bracket_orders(trade["id"])
                trade["tp_order_id"] = None
                trade["sl_order_id"] = None
                self._ensure_exit_brackets(trade)
            return None

        exit_status = tp_status if tp_done else sl_status
        sibling_order_id = sl_order_id if tp_done else tp_order_id
        exit_price = float((exit_status or {}).get("price") or trade.get("tp_price") or trade.get("sl_price") or trade["entry_price"])
        pnl = round((exit_price - float(trade["entry_price"])) * int(trade.get("size_contracts", 0) or 0), 2)
        trade_status = "exited_profit" if pnl > 0 else "exited_loss"

        if sibling_order_id:
            try:
                self._kalshi.cancel_order(sibling_order_id)
            except requests.HTTPError as exc:
                logger.warning(f"Failed to cancel sibling order {sibling_order_id}: {exc}")

        self._db.clear_trade_bracket_orders(trade["id"])
        self._db.resolve_trade(trade["id"], trade_status, pnl)

        resolved_trade = dict(trade)
        resolved_trade["status"] = trade_status
        resolved_trade["pnl"] = pnl

        logger.info(
            "[LIVE] Stop-loss exit filled: %s | entry=$%.2f -> exit=$%.2f | pnl=$%.2f",
            trade["ticker"],
            float(trade["entry_price"]),
            exit_price,
            pnl,
        )
        return resolved_trade

    def _get_exit_thresholds(self) -> tuple[float, float]:
        """Mirror the learned TP/SL thresholds used by the position monitor."""
        tp_threshold = DEFAULT_TP_THRESHOLD
        sl_threshold = DEFAULT_SL_THRESHOLD

        tp_count_raw = self._db.get_heuristic("exit_take_profit_count")
        td_tp_count_raw = self._db.get_heuristic("exit_time_decay_tp_count")
        tp_count = int(tp_count_raw) if tp_count_raw and tp_count_raw.isdigit() else 0
        td_tp_count = int(td_tp_count_raw) if td_tp_count_raw and td_tp_count_raw.isdigit() else 0
        total_tp = tp_count + td_tp_count
        if total_tp >= 3:
            tp_regret_raw = self._db.get_heuristic("exit_tp_regret_count")
            tp_regret = int(tp_regret_raw) if tp_regret_raw and tp_regret_raw.isdigit() else 0
            regret_rate = tp_regret / total_tp if total_tp > 0 else 0.0
            if regret_rate > 0.60:
                tp_threshold = min(0.80, DEFAULT_TP_THRESHOLD + 0.10)
            elif regret_rate < 0.25 and total_tp >= 5:
                tp_threshold = max(0.25, DEFAULT_TP_THRESHOLD - 0.05)

        sl_count_raw = self._db.get_heuristic("exit_stop_loss_count")
        td_sl_count_raw = self._db.get_heuristic("exit_time_decay_stop_count")
        sl_count = int(sl_count_raw) if sl_count_raw and sl_count_raw.isdigit() else 0
        td_sl_count = int(td_sl_count_raw) if td_sl_count_raw and td_sl_count_raw.isdigit() else 0
        total_sl = sl_count + td_sl_count
        if total_sl >= 3:
            sl_regret_raw = self._db.get_heuristic("exit_sl_regret_count")
            sl_regret = int(sl_regret_raw) if sl_regret_raw and sl_regret_raw.isdigit() else 0
            regret_rate = sl_regret / total_sl if total_sl > 0 else 0.0
            if regret_rate > 0.50:
                sl_threshold = min(0.75, DEFAULT_SL_THRESHOLD + 0.10)
            elif regret_rate < 0.20 and total_sl >= 5:
                sl_threshold = max(0.30, DEFAULT_SL_THRESHOLD - 0.05)

        return tp_threshold, sl_threshold

    def _save_model_votes(self, trade_id: int, decision: TradeDecision) -> None:
        """Persist ensemble sub-model votes so postmortem can score them later."""
        pred = decision.prediction
        if not pred or not pred.model_votes:
            return
        try:
            self._db.save_model_predictions(trade_id, pred.model_votes)
        except Exception as exc:
            logger.debug("Could not save model votes for trade %d: %s", trade_id, exc)

    @staticmethod
    def _is_filled_order_status(status: dict[str, Any] | None) -> bool:
        if not status:
            return False
        order_status = str(status.get("status", "")).lower()
        remaining = int(status.get("remaining_count", 0) or 0)
        if order_status in ("executed", "filled"):
            return True
        if order_status in {"canceled", "cancelled", "expired", "rejected", "failed"}:
            return False
        return remaining <= 0

    @staticmethod
    def _is_terminal_order_status(status: dict[str, Any] | None) -> bool:
        if not status:
            return False
        return str(status.get("status", "")).lower() in {
            "executed",
            "filled",
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "failed",
        }
