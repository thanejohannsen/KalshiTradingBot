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

            # Find entry price from fills API — actual execution prices
            side = "yes" if yes_count > 0 else "no"
            contracts = yes_count if side == "yes" else no_count
            entry_price = self._compute_entry_price_from_fills(ticker, side)

            # Last resort: current market ask price
            market_category = ""
            try:
                market = self._kalshi.get_market(ticker)
                if entry_price <= 0:
                    entry_price = market.yes_ask if side == "yes" else (market.no_ask or 0.0)
                market_category = market.category or ""
            except Exception:
                if entry_price <= 0:
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
                "manual": 1,
                "predicted_prob": None,
                "market_prob": None,
                "edge": None,
                "sentiment_score": None,
                "narrative": None,
                "confidence": None,
                "signals": None,
                "category": market_category,
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

    def _compute_entry_price_from_fills(self, ticker: str, side: str) -> float:
        """Compute weighted average entry price from Kalshi fills API.

        The fills endpoint returns actual execution records with the real
        price paid, which is more reliable than order prices.
        Falls back to order history if fills endpoint fails.
        """
        # Try fills API first (most accurate)
        try:
            fills = self._kalshi.get_fills(ticker=ticker)
            buy_fills = [
                f for f in fills
                if f.get("action") == "buy" and f.get("side") == side
            ]
            if buy_fills:
                total_cost = 0.0
                total_count = 0
                for f in buy_fills:
                    px = f.get("price") or 0.0
                    ct = f.get("count") or 0
                    if px > 0 and ct > 0:
                        total_cost += px * ct
                        total_count += ct
                if total_count > 0:
                    avg = total_cost / total_count
                    logger.debug(
                        "Fills API: %s %s → %d fills, avg $%.4f (%d contracts)",
                        ticker, side, len(buy_fills), avg, total_count,
                    )
                    return avg
        except Exception as exc:
            logger.debug("Fills API failed for %s: %s — trying orders", ticker, exc)

        # Fallback: order history
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
                total_cost = 0.0
                total_filled = 0
                for bo in buy_orders:
                    px = bo.get("price") or 0.0
                    filled = bo.get("filled_count") or bo.get("initial_count") or 0
                    if px > 0 and filled > 0:
                        total_cost += px * filled
                        total_filled += filled
                if total_filled > 0:
                    return total_cost / total_filled
        except Exception:
            pass

        return 0.0

    def resync_entry_prices(self) -> int:
        """Re-sync entry prices for all open manual trades using Kalshi fills.

        Fixes trades that were imported with wrong entry prices (e.g.,
        market_value fallback was used instead of actual fill price).
        Called at startup after import_untracked_positions().
        """
        if not self._cfg.is_live:
            return 0

        open_trades = self._db.get_open_trades()
        updated = 0

        for trade in open_trades:
            ticker = trade["ticker"]
            side = trade["side"]
            old_price = float(trade.get("entry_price") or 0)
            trade_id = trade["id"]

            fills_price = self._compute_entry_price_from_fills(ticker, side)
            if fills_price <= 0:
                continue

            # Only update if significantly different (>1% off)
            if old_price > 0 and abs(fills_price - old_price) / old_price < 0.01:
                continue

            contracts = int(trade.get("size_contracts") or 0)
            new_size_dollars = round(contracts * fills_price, 2)

            with self._db._connect() as conn:
                conn.execute(
                    "UPDATE trades SET entry_price = ?, size_dollars = ? WHERE id = ?",
                    (fills_price, new_size_dollars, trade_id),
                )

            logger.info(
                "Resynced entry price: trade #%d %s %s | $%.4f → $%.4f",
                trade_id, ticker, side, old_price, fills_price,
            )
            updated += 1

        return updated

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
        tp_resolved = 0
        for trade in open_trades:
            # Skip manually placed trades — don't reconcile or cancel them
            if trade.get("manual"):
                continue

            ticker = trade["ticker"]
            order_id = trade.get("kalshi_order_id") or ""
            trade_id = trade["id"]

            in_positions = ticker in held_tickers
            in_orders    = order_id in active_order_ids

            if not in_positions and not in_orders:
                # Before cancelling, check if a TP order filled — that means
                # the position was closed profitably, not cancelled.
                tp_order_id = trade.get("tp_order_id")
                if tp_order_id and trade.get("entry_filled_at"):
                    try:
                        tp_status = self._kalshi.get_order_status(tp_order_id)
                        if self._is_filled_order_status(tp_status):
                            exit_price = float(
                                (tp_status or {}).get("price")
                                or trade.get("tp_price")
                                or trade["entry_price"]
                            )
                            entry_price = float(trade["entry_price"])
                            contracts = int(trade.get("size_contracts") or 0)
                            pnl = round((exit_price - entry_price) * contracts, 2)
                            status = "exited_profit" if pnl > 0 else "exited_loss"

                            self._db.clear_trade_bracket_orders(trade_id)
                            self._db.resolve_trade(trade_id, status, pnl)

                            logger.info(
                                "Reconcile: trade #%d %s TP filled — %s | entry=$%.2f exit=$%.2f pnl=$%.2f",
                                trade_id, ticker, status, entry_price, exit_price, pnl,
                            )
                            tp_resolved += 1
                            continue
                    except Exception as exc:
                        logger.debug("Reconcile: could not check TP for #%d: %s", trade_id, exc)

                self._db.cancel_trade(trade_id)
                logger.info(
                    "Reconcile: trade #%d %s not in Kalshi positions or orders — cancelled",
                    trade_id, ticker,
                )
                cancelled += 1

        if cancelled:
            logger.info("Reconciliation cancelled %d ghost trade(s)", cancelled)
        if tp_resolved:
            logger.info("Reconciliation resolved %d TP-filled trade(s)", tp_resolved)

        # Cancel stale resting sell orders left over from the old TP system.
        # Trailing stop now handles all exits — no resting sells should exist.
        stale_cancelled = 0
        for order in active_orders:
            if str(order.get("action", "")).lower() != "sell":
                continue
            oid = order.get("order_id", "")
            # Check if any open trade in DB still references this as a TP order
            is_tracked = any(
                t.get("tp_order_id") == oid
                for t in open_trades
            )
            if not is_tracked:
                try:
                    self._kalshi.cancel_order(oid)
                    stale_cancelled += 1
                    logger.info(
                        "Reconcile: cancelled stale resting sell order %s (%s %s x%s)",
                        oid,
                        order.get("side", "?"),
                        order.get("ticker", "?"),
                        order.get("remaining_count", "?"),
                    )
                except Exception as exc:
                    logger.debug("Could not cancel stale order %s: %s", oid, exc)
        if stale_cancelled:
            logger.info("Reconciliation cancelled %d stale resting sell order(s)", stale_cancelled)

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
            # Skip manually placed trades — don't touch them
            if trade.get("manual"):
                continue

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
        """Detect entry fills, place resting TP order, and reconcile TP fills.

        Stop-loss is handled entirely by the position monitor (active polling).
        Kalshi has no stop/conditional order type — a limit sell below the bid
        fills immediately, so resting SL orders are impossible.
        """
        if not self._cfg.is_live or trade.get("action") != "buy":
            return None

        entry_status = str((status_info or {}).get("status", "")).lower()
        filled_count = int((status_info or {}).get("filled_count", 0) or 0)

        # Mark entry filled on full fill OR partial fill (any contracts held)
        if not trade.get("entry_filled_at") and (
            entry_status in ("executed", "filled") or filled_count > 0
        ):
            self._db.mark_trade_entry_filled(trade["id"])
            trade["entry_filled_at"] = datetime.utcnow().isoformat()
            self._ensure_exit_brackets(trade)

        if trade.get("entry_filled_at") and not trade.get("tp_order_id"):
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
        """
        Check order fill status from Kalshi (source of truth) and sync DB.

        Handles three states:
          1. Fully filled   → mark entry filled, record fill stat
          2. Partial fill   → update DB contracts to filled count, mark entry
                              filled so position monitor tracks the held portion
          3. Cancelled/gone → cancel in DB, or cancel only unfilled portion
                              if some contracts already filled
        """
        try:
            status = self._kalshi.get_order_status(order_id)
        except Exception as exc:
            exc_str = str(exc)
            if "404" in exc_str or "not_found" in exc_str:
                # Order gone from Kalshi.  If we hold contracts for this ticker
                # (checked via reconcile), the trade stays open with whatever
                # the position monitor found.  Otherwise reconcile will cancel.
                if order_id not in self._counted_orders:
                    self._counted_orders.add(order_id)
                    logger.info(
                        "Order %s for %s not found on Kalshi (404)",
                        order_id, trade["ticker"],
                    )
            else:
                logger.debug("Could not check fill status for %s: %s", order_id, exc)
            return None

        order_status = str(status.get("status", "")).lower()
        filled_count = int(status.get("filled_count", 0) or 0)
        remaining = int(status.get("remaining_count", 0) or 0)
        original_contracts = int(trade.get("size_contracts", 0) or 0)
        entry_price = float(trade.get("entry_price") or 0)
        trade_id = trade["id"]

        # Skip if we already fully processed this order
        if order_id in self._counted_orders:
            return status

        # ── Fully filled ───────────────────────────────────────────
        if order_status in ("executed", "filled"):
            self._counted_orders.add(order_id)
            # Sync DB to actual fill count (may differ from original)
            if filled_count > 0 and filled_count != original_contracts:
                self._db.update_trade_fill(
                    trade_id, filled_count,
                    round(filled_count * entry_price, 2),
                )
                logger.info(
                    "Fill sync: trade #%d %s | ordered=%d filled=%d → DB updated",
                    trade_id, trade["ticker"], original_contracts, filled_count,
                )
            self._record_fill(trade, filled=True, partial=False)
            return status

        # ── Cancelled / expired ────────────────────────────────────
        if order_status in ("canceled", "cancelled", "expired"):
            self._counted_orders.add(order_id)

            if filled_count > 0:
                # Partial fill: some contracts filled before cancellation.
                # Keep the trade open with the filled portion.
                self._db.update_trade_fill(
                    trade_id, filled_count,
                    round(filled_count * entry_price, 2),
                )
                self._record_fill(trade, filled=True, partial=True)
                logger.info(
                    "Partial fill on cancelled order: trade #%d %s | "
                    "filled=%d of %d — DB updated, trade stays open",
                    trade_id, trade["ticker"], filled_count, original_contracts,
                )
                # Update in-memory trade dict so downstream code sees the truth
                trade["size_contracts"] = filled_count
                trade["size_dollars"] = round(filled_count * entry_price, 2)
            else:
                # Nothing filled — fully cancel
                self._record_fill(trade, filled=False, partial=False)
                self._db.cancel_trade(trade_id)
                logger.info(
                    "Order %s for %s fully cancelled (0 filled) — trade #%d cancelled",
                    order_id, trade["ticker"], trade_id,
                )
            return status

        # ── Still resting — check for partial fill in progress ─────
        if filled_count > 0 and remaining > 0:
            # Some filled, some still resting.  Update DB to reflect
            # the filled portion so position monitor can track it.
            if filled_count != original_contracts:
                self._db.update_trade_fill(
                    trade_id, filled_count,
                    round(filled_count * entry_price, 2),
                )
                trade["size_contracts"] = filled_count
                trade["size_dollars"] = round(filled_count * entry_price, 2)
                logger.info(
                    "Partial fill in progress: trade #%d %s | "
                    "filled=%d remaining=%d — DB synced to filled",
                    trade_id, trade["ticker"], filled_count, remaining,
                )
            self._record_fill(trade, filled=True, partial=True)

        return status

    def _maybe_reprice_stale_buy_order(
        self,
        trade: dict[str, Any],
        status: dict[str, Any] | None,
    ) -> None:
        """Cancel/replace stale buy orders so they don't rest indefinitely.

        Only reprices the unfilled remainder — already-filled contracts are
        tracked in the DB by _check_fill_status and won't be re-ordered.
        """
        if not status:
            return
        if trade.get("action") != "buy":
            return

        order_id = trade.get("kalshi_order_id")
        if not order_id:
            return

        order_status = str(status.get("status", "")).lower()
        if order_status in ("executed", "filled", "canceled", "cancelled", "expired"):
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

        # Only reprice the unfilled remainder from Kalshi (source of truth)
        remaining_count = int(status.get("remaining_count", 0) or 0)
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
                    "Repriced stale BUY: trade #%d %s %.2f -> %.2f "
                    "(age=%.0fs, remaining=%d of original)",
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
        """No-op — trailing stop replaced fixed TP resting orders.

        Exit management is now handled entirely by the position monitor's
        trailing stop logic (high-water mark tracking + trail percentage).
        This method is kept so _sync_entry_and_brackets doesn't break, but
        it no longer places any orders.
        """
        return

    def _reconcile_exit_brackets(self, trade: dict[str, Any]) -> dict[str, Any] | None:
        """Check if the resting TP order has filled; resolve the trade if so.

        If the TP order was cancelled or disappeared, re-arm it.
        Legacy SL order IDs are also handled for backwards compatibility.
        """
        tp_order_id = trade.get("tp_order_id")
        sl_order_id = trade.get("sl_order_id")  # legacy — may exist on older trades

        # Cancel any legacy SL orders that predate this fix
        if sl_order_id:
            import requests
            try:
                self._kalshi.cancel_order(sl_order_id)
                logger.info("[LIVE] Cancelled legacy SL order %s for %s", sl_order_id, trade["ticker"])
            except requests.HTTPError:
                pass
            self._db.set_trade_bracket_orders(
                trade_id=trade["id"],
                tp_order_id=tp_order_id,
                tp_price=trade.get("tp_price"),
                sl_order_id=None,
                sl_price=None,
            )
            trade["sl_order_id"] = None
            trade["sl_price"] = None

        if not tp_order_id:
            return None

        import requests
        tp_status = None
        try:
            tp_status = self._kalshi.get_order_status(tp_order_id)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            if status_code == 404:
                logger.info("TP order %s no longer exists (404) — clearing (trailing stop handles exits)", tp_order_id)
                self._db.clear_trade_bracket_orders(trade["id"])
                trade["tp_order_id"] = None
                # Legacy TP gone — trailing stop handles exits now
            else:
                logger.warning("Failed to get TP order status for %s: %s", tp_order_id, exc)
            return None

        tp_done = self._is_filled_order_status(tp_status)
        if not tp_done:
            # If TP order was cancelled/expired, just clear it (trailing stop handles exits now)
            if self._is_terminal_order_status(tp_status):
                self._db.clear_trade_bracket_orders(trade["id"])
                trade["tp_order_id"] = None
            return None

        # TP filled — resolve the trade
        exit_price = float((tp_status or {}).get("price") or trade.get("tp_price") or trade["entry_price"])
        pnl = round((exit_price - float(trade["entry_price"])) * int(trade.get("size_contracts", 0) or 0), 2)
        trade_status = "exited_profit" if pnl > 0 else "exited_loss"

        self._db.clear_trade_bracket_orders(trade["id"])
        self._db.resolve_trade(trade["id"], trade_status, pnl)

        resolved_trade = dict(trade)
        resolved_trade["status"] = trade_status
        resolved_trade["pnl"] = pnl

        logger.info(
            "[LIVE] Take-profit filled: %s | entry=$%.2f → exit=$%.2f | pnl=$%.2f",
            trade["ticker"],
            float(trade["entry_price"]),
            exit_price,
            pnl,
        )
        return resolved_trade

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
