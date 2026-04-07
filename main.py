"""
MAIN ORCHESTRATOR — runs the full trading pipeline in a continuous loop.

Pipeline:  SCAN → RESEARCH → PREDICT → RISK → EXECUTE → MONITOR → POSITION MONITOR → POSTMORTEM → SLEEP → REPEAT
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from typing import Any

# Prevent Intel MKL/Fortran from hijacking Ctrl+C
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from config import Config
from database import Database
from data_sources.kalshi_client import KalshiAPIClient
from data_sources.reddit_client import RedditClient
from data_sources.news_client import NewsClient
from data_sources.sportsbook_client import SportsbookClient
from models.sentiment import SentimentPipeline
from models.claude_client import ClaudeClient
from agents.scan_agent import ScanAgent
from agents.research_agent import ResearchAgent
from agents.prediction_agent import PredictionAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.position_monitor_agent import PositionMonitorAgent
from agents.postmortem_agent import PostmortemAgent
from agents.strategy_evolution_agent import StrategyEvolutionAgent
from utils import (
    drain_position_monitor_console_buffer,
    set_position_monitor_console_deferred,
    setup_logging,
)

logger = logging.getLogger("kalshi_bot")

# Graceful shutdown flag
_shutdown = False


def set_shutdown() -> None:
    """Set shutdown flag from external callers (e.g., TUI)."""
    global _shutdown
    _shutdown = True


def is_shutdown() -> bool:
    return _shutdown


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown
    if logger:
        logger.info("Shutdown signal received (sig=%d). Finishing current cycle\u2026", signum)
    _shutdown = True
    # Raise KeyboardInterrupt so ThreadPoolExecutor futures get interrupted
    raise KeyboardInterrupt


class TradingBot:
    """Orchestrates the full multi-agent trading pipeline."""

    def __init__(self, cfg: Config, state: Any = None) -> None:
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self._state = state  # Optional BotState for TUI integration

        # ── API clients ─────────────────────────────────────────────
        self.kalshi = KalshiAPIClient(cfg, shutdown_check=lambda: _shutdown)

        self.reddit: RedditClient | None = None
        if cfg.reddit_enabled:
            self.reddit = RedditClient(
                client_id=cfg.reddit_client_id,
                client_secret=cfg.reddit_client_secret,
                user_agent=cfg.reddit_user_agent,
            )

        self.news: NewsClient | None = None
        if cfg.newsapi_enabled:
            self.news = NewsClient(newsapi_key=cfg.newsapi_key)
        else:
            self.news = NewsClient()  # RSS-only mode

        self.sportsbook: SportsbookClient | None = None
        if cfg.odds_api_enabled:
            self.sportsbook = SportsbookClient(api_key=cfg.odds_api_key)
            logger.info("Sportsbook odds client enabled (The Odds API)")

        self.claude: ClaudeClient | None = None
        if cfg.claude_enabled:
            self.claude = ClaudeClient(api_key=cfg.anthropic_api_key)

        # ── Models ──────────────────────────────────────────────────
        self.sentiment = SentimentPipeline(use_finbert=True)

        # ── Agents ──────────────────────────────────────────────────
        self.scan_agent = ScanAgent(cfg, self.db, self.kalshi)
        self.research_agent = ResearchAgent(
            cfg, self.sentiment, self.reddit, self.news, self.claude,
            shutdown_check=lambda: _shutdown,
            db=self.db,
            sportsbook=self.sportsbook,
        )
        self.prediction_agent = PredictionAgent(cfg, self.db)
        self.risk_agent = RiskAgent(cfg, self.db, self.kalshi)
        self.execution_agent = ExecutionAgent(cfg, self.db, self.kalshi)
        self.position_monitor = PositionMonitorAgent(cfg, self.db, self.kalshi)
        self.postmortem_agent = PostmortemAgent(cfg, self.db, self.claude)
        self.strategy_evolution = StrategyEvolutionAgent(cfg, self.db)

        # Continuous position-monitor thread state
        self._pm_stop = threading.Event()
        self._pm_lock = threading.Lock()
        self._pm_thread: threading.Thread | None = None

    def startup(self) -> None:
        """Connect to all external services."""
        logger.info("=" * 60)
        logger.info("KALSHI TRADING BOT — Starting up")
        logger.info("Mode: %s", self.cfg.mode.upper())
        logger.info("Bankroll: $%.2f", self.cfg.bankroll)
        logger.info("=" * 60)

        # ── Print scan parameters + learned changes ─────────────────
        self._print_parameters()

        # Purge stale research cache
        purged = self.db.purge_old_research(max_age_hours=24)
        if purged:
            logger.info("Purged %d stale research cache entries", purged)

        # Purge old market snapshots (keep last 14 days + latest per ticker)
        snap_purged = self.db.purge_old_snapshots(keep_days=14)
        if snap_purged:
            logger.info("Purged %d old market snapshots", snap_purged)

        # Connect Kalshi
        self.kalshi.connect()

        # Bootstrap baselines on cold start (no recent snapshots)
        if not self.db.has_recent_snapshots(max_age_hours=2):
            logger.info("No recent snapshots — bootstrapping baselines…")
            self.scan_agent.bootstrap_baselines(top_n=300)
        else:
            logger.info("Recent snapshots found — skipping bootstrap.")

        # Connect Reddit
        if self.reddit:
            self.reddit.connect()

        # Log balance
        if self.cfg.is_live:
            try:
                balance = self.kalshi.get_balance()
                logger.info("Kalshi balance: $%.2f", balance)
                if self._state:
                    self._state.set_balance(balance)
            except Exception as exc:
                logger.warning("Could not fetch balance: %s", exc)

        # Import any positions held on Kalshi that aren't in the DB
        # (manual trades, trades placed before bot started, etc.)
        imported = self.execution_agent.import_untracked_positions()
        if imported:
            logger.info("Imported %d untracked position(s) from Kalshi API", imported)

        # Re-sync entry prices for existing trades using fills API
        # (fixes trades imported with wrong entry prices)
        resynced = self.execution_agent.resync_entry_prices()
        if resynced:
            logger.info("Resynced entry prices for %d trade(s)", resynced)

        self._log_trade_history()
        self.strategy_evolution.log_status()
        self._start_position_monitor_thread()

    def _set_phase(self, phase: str) -> None:
        if self._state:
            self._state.set_phase(phase)

    def run_cycle(self) -> None:
        """Execute one full pipeline cycle."""
        cycle_start = time.monotonic()
        if self._state:
            self._state.increment_cycle()
        self._set_phase("monitoring")
        logger.info("─" * 50)
        logger.info("CYCLE START")

        # Print trade history every cycle, then monitor open positions
        # before research starts so this report appears ahead of research progress.
        self._log_trade_history()
        with self._pm_lock:
            self.position_monitor.monitor_positions(
                report=True,
                allow_exits=False,
                log_holds=True,
            )

        # ── STEP 1: SCAN ────────────────────────────────────────────
        self._set_phase("scanning")
        candidates = self.scan_agent.scan()
        if not candidates:
            logger.info("No candidates found this cycle.")
        else:
            # Store scan results for TUI
            if self._state:
                self._state.set_scan_results(candidates)

            # ── STEP 2: RESEARCH (parallel) ─────────────────────────────
            self._set_phase("researching")
            set_position_monitor_console_deferred(True)
            try:
                research_results = self.research_agent.research_batch(candidates)
            finally:
                set_position_monitor_console_deferred(False)
                for line in drain_position_monitor_console_buffer():
                    sys.stdout.write(line + "\n")
                sys.stdout.flush()

            # Pair candidates with research results by ticker
            research_map = {r.ticker: r for r in research_results}

            # ── STEP 3: PREDICT ─────────────────────────────────────────
            self._set_phase("predicting")
            predictions = []
            for cand in candidates:
                research = research_map.get(cand.market.ticker)
                if not research:
                    continue
                pred = self.prediction_agent.predict(cand, research)
                if pred:
                    predictions.append((cand, research, pred))

            if not predictions:
                logger.info("No actionable edges found this cycle.")
            else:
                logger.info("%d predictions with edge above threshold", len(predictions))

            # ── STEP 4: RISK + STEP 5: EXECUTE ─────────────────────────
            self._set_phase("executing")
            # Get current bankroll (live) or use config (paper).
            # In live mode, Kalshi's get_balance() already returns spendable
            # cash net of any reserved funds — no further adjustment needed.
            bankroll = self.cfg.bankroll
            if self.cfg.is_live:
                try:
                    bankroll = self.kalshi.get_balance()
                except Exception:
                    pass

            blocked_count = 0
            approved_count = 0
            risk_decisions: list[dict[str, Any]] = []

            for cand, research, pred in predictions:
                # In live mode, re-fetch the real balance before each trade so
                # we never overcommit if previous orders in this cycle already
                # reserved funds on Kalshi's side.
                if self.cfg.is_live:
                    try:
                        bankroll = self.kalshi.get_balance()
                    except Exception:
                        pass  # keep last known value

                decision = self.risk_agent.evaluate(pred, research, cand, bankroll)
                risk_decisions.append({
                    "ticker": pred.ticker,
                    "approved": decision.approved,
                    "side": decision.side,
                    "edge": pred.edge,
                    "quality": pred.quality_score,
                    "reasoning": decision.reasoning,
                    "size": decision.size_dollars,
                })
                if decision.approved:
                    approved_count += 1
                    logger.info(
                        "Risk PASS: %s %s | edge=%+.3f | quality=%.0f | size=$%.2f",
                        decision.side.upper(),
                        pred.ticker,
                        pred.edge,
                        pred.quality_score,
                        decision.size_dollars,
                    )
                    trade_id = self.execution_agent.execute(decision)
                    if trade_id:
                        logger.info("Trade #%d placed for %s", trade_id, pred.ticker)
                        # For paper mode, deduct locally since there's no real balance to fetch
                        if not self.cfg.is_live:
                            bankroll -= decision.size_dollars
                        if bankroll < self.cfg.min_trade_dollars:
                            logger.info("Remaining bankroll $%.2f below min trade $%.2f — stopping trades this cycle", bankroll, self.cfg.min_trade_dollars)
                            break
                else:
                    blocked_count += 1
                    # Shadow-track blocked trades to learn from missed opportunities
                    side = "yes" if pred.edge > 0 else "no"
                    entry_price = cand.market.yes_ask if side == "yes" else (cand.market.no_ask or (1.0 - cand.market.yes_bid))
                    self.db.insert_shadow_trade({
                        "ticker": pred.ticker,
                        "side": side,
                        "entry_price": entry_price,
                        "predicted_prob": pred.true_probability,
                        "market_prob": pred.market_probability,
                        "edge": pred.edge,
                        "block_reason": decision.reasoning,
                    })

            logger.info(
                "Risk summary: approved=%d blocked=%d",
                approved_count,
                blocked_count,
            )
            if self._state:
                self._state.set_risk_decisions(risk_decisions, approved_count, blocked_count)

        # ── STEP 5 (continued): MONITOR SETTLEMENTS ─────────────────
        self._set_phase("monitoring")
        with self._pm_lock:
            resolved = self.execution_agent.monitor_open_trades()

        # ── STEP 5b: MONITOR SHADOW TRADES ──────────────────────────
        self._monitor_shadow_trades()

        # ── STEP 7: RECORD OUTCOMES + POSTMORTEM ────────────────────
        for trade in resolved:
            # Record win/loss for category profitability tracking
            self.postmortem_agent.record_outcome(trade)
            # Deep analysis for all losses (settled or early-exited)
            if trade.get("status") in ("lost", "exited_loss"):
                self.postmortem_agent.analyze(trade)
            # Check if any previously exited trades' markets have now settled
            # (for regret analysis)

        # ── Regret analysis for settled markets with prior early exits ──
        self._check_exit_regrets()

        # ── Learn adaptive edge threshold from blocked-trade outcomes ──
        self._learn_edge_threshold()

        # ── STEP 8: STRATEGY EVOLUTION (every 10 cycles) ────────────
        self.strategy_evolution.maybe_evolve()

        self._set_phase("sleeping")
        elapsed = time.monotonic() - cycle_start
        logger.info("CYCLE COMPLETE in %.1fs", elapsed)

    def _log_trade_history(self) -> None:
        stats = self.db.get_trade_stats()
        logger.info(
            "Trade history: %d total | %d realized W / %d realized L | %d early exits (%d profit, %d loss) | "
            "settled=%d W / %d L | "
            "Win rate: %.1f%% | P&L: $%.2f",
            stats["total_trades"],
            stats.get("realized_wins") or 0,
            stats.get("realized_losses") or 0,
            (stats.get("early_exits_profit") or 0) + (stats.get("early_exits_loss") or 0),
            stats.get("early_exits_profit") or 0,
            stats.get("early_exits_loss") or 0,
            stats["wins"] or 0,
            stats["losses"] or 0,
            stats["win_rate"] * 100,
            stats["total_pnl"],
        )

    def run_loop(self) -> None:
        """Main continuous loop."""
        self.startup()

        while not _shutdown:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Interrupted — shutting down.")
                break
            except Exception as exc:
                logger.error("Cycle error: %s", exc, exc_info=True)

            if _shutdown:
                break

            logger.info(
                "Sleeping %ds until next scan…", self.cfg.scan_interval_seconds
            )
            # Sleep in small increments to respond to shutdown quickly.
            for _ in range(self.cfg.scan_interval_seconds):
                if _shutdown:
                    break
                time.sleep(1)

        self._stop_position_monitor_thread()
        logger.info("Bot shut down gracefully.")
        self._print_summary()

    def _start_position_monitor_thread(self) -> None:
        if self._pm_thread and self._pm_thread.is_alive():
            return

        self._pm_stop.clear()

        def _loop() -> None:
            while not _shutdown and not self._pm_stop.is_set():
                try:
                    with self._pm_lock:
                        resolved = self.execution_agent.monitor_open_trades()
                        exits = self.position_monitor.monitor_positions(
                            report=True,
                            allow_exits=True,
                            log_holds=False,
                        )
                    for trade in resolved:
                        self.postmortem_agent.record_outcome(trade)
                        if trade.get("status") in ("lost", "exited_loss"):
                            self.postmortem_agent.analyze(trade)
                    for trade in exits:
                        self.postmortem_agent.record_outcome(trade)
                        if trade.get("status") in ("lost", "exited_loss"):
                            self.postmortem_agent.analyze(trade)
                except Exception as exc:
                    logger.debug("Position monitor thread: %s", exc)

                # Poll every 20s continuously (including research/scan phases).
                self._pm_stop.wait(20)

        self._pm_thread = threading.Thread(
            target=_loop,
            name="position-monitor",
            daemon=True,
        )
        self._pm_thread.start()
        logger.info("Position monitor background thread started (20s interval)")

    def _stop_position_monitor_thread(self) -> None:
        self._pm_stop.set()
        if self._pm_thread and self._pm_thread.is_alive():
            self._pm_thread.join(timeout=2.0)
        self._pm_thread = None

    def _print_summary(self) -> None:
        stats = self.db.get_trade_stats()
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("  Total trades: %d", stats["total_trades"])
        logger.info("  Realized wins: %d", stats.get("realized_wins") or 0)
        logger.info("  Realized losses: %d", stats.get("realized_losses") or 0)
        logger.info("  Settled wins: %d", stats["wins"] or 0)
        logger.info("  Settled losses: %d", stats["losses"] or 0)
        logger.info("  Early exits (profit): %d", stats.get("early_exits_profit") or 0)
        logger.info("  Early exits (loss): %d", stats.get("early_exits_loss") or 0)
        logger.info("  Open: %d", stats["open_trades"] or 0)
        logger.info("  Win rate: %.1f%%", stats["win_rate"] * 100)
        logger.info("  Net P&L: $%.2f", stats["total_pnl"])

        # Shadow trade stats
        shadow = self.db.get_shadow_stats()
        shadow_resolved = (shadow["would_have_won"] or 0) + (shadow["would_have_lost"] or 0)
        if shadow_resolved > 0 or (shadow["pending"] or 0) > 0:
            logger.info("  --- Blocked trades (shadow) ---")
            logger.info("  Tracked: %d | Pending: %d", shadow["total"], shadow["pending"] or 0)
            if shadow_resolved > 0:
                logger.info("  Would-have-won: %d | Would-have-lost: %d", shadow["would_have_won"] or 0, shadow["would_have_lost"] or 0)
                logger.info("  Shadow win rate: %.1f%% | Shadow P&L: $%.2f", shadow["shadow_win_rate"] * 100, shadow["shadow_pnl"])

        logger.info("=" * 60)

    def _check_exit_regrets(self) -> None:
        """
        Look for trades that were exited early whose markets have now settled.
        Run regret analysis so the position monitor can learn.
        """
        all_heuristics = self.db.get_all_heuristics()
        for key, value in all_heuristics.items():
            if not key.startswith("last_exit_") or not value:
                continue
            try:
                exit_data = __import__("json").loads(value)
                ticker = exit_data.get("ticker", "")
                trade_id = exit_data.get("trade_id")
                if not ticker or not trade_id:
                    continue

                # Check if the market has settled
                market = self._kalshi_check_settled(ticker)
                if market is None:
                    continue

                # Get the trade record
                trade = self.db.get_trade(trade_id)
                if trade:
                    self.postmortem_agent.analyze_exit_regret(trade, market)

            except Exception as exc:
                logger.warning("Exit regret analysis failed for key %s: %s", key, exc)

    def _kalshi_check_settled(self, ticker: str):
        """Return market data if settled, else None."""
        try:
            market = self.kalshi.get_market(ticker)
            if market.status in ("settled", "closed", "finalized"):
                return market
        except Exception:
            pass
        return None

    def _monitor_shadow_trades(self) -> None:
        """Check pending shadow trades for settlement and record hypothetical P&L."""
        pending = self.db.get_pending_shadow_trades()
        if not pending:
            return

        for shadow in pending:
            ticker = shadow["ticker"]
            market = self._kalshi_check_settled(ticker)
            if market is None:
                continue

            entry_price = shadow["entry_price"]
            side = shadow["side"]
            result_price = market.last_price

            if side == "yes":
                pnl = (1.0 - entry_price) if result_price >= 0.90 else -entry_price
            else:
                pnl = (1.0 - entry_price) if result_price <= 0.10 else -entry_price

            status = "shadow_won" if pnl > 0 else "shadow_lost"
            self.db.resolve_shadow_trade(shadow["id"], status, round(pnl, 4))
            logger.info(
                "Shadow trade #%d (%s) resolved: %s | hypothetical P&L=$%.2f",
                shadow["id"], ticker, status, pnl,
            )

        # Periodic summary
        stats = self.db.get_shadow_stats()
        resolved = (stats["would_have_won"] or 0) + (stats["would_have_lost"] or 0)
        if resolved > 0:
            logger.info(
                "Shadow stats: %d resolved | %d would-have-won | %d would-have-lost | "
                "shadow_win_rate=%.1f%% | shadow_pnl=$%.2f",
                resolved,
                stats["would_have_won"] or 0,
                stats["would_have_lost"] or 0,
                stats["shadow_win_rate"] * 100,
                stats["shadow_pnl"],
            )

    def _learn_edge_threshold(self) -> None:
        """Adjust min edge threshold based on shadow-trade outcomes and decay if inactive."""
        stats = self.db.get_shadow_stats()
        resolved = (stats["would_have_won"] or 0) + (stats["would_have_lost"] or 0)
        # Decay config
        decay_hours = 6  # Decay if no trades in last 6 hours
        decay_rate = 0.5  # Move halfway back to baseline per decay event

        # Start from learned value if present, otherwise config baseline.
        raw = self.db.get_heuristic("learned_min_edge_threshold")
        current = self.cfg.min_edge_threshold
        if raw:
            try:
                current = float(raw)
            except (ValueError, TypeError):
                current = self.cfg.min_edge_threshold

        # Decay if no trades opened recently
        recent_trades = self.db.get_open_trades()
        recent_trade_found = False
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        for t in recent_trades:
            opened_at = t.get("opened_at")
            if opened_at:
                try:
                    t_open = datetime.fromisoformat(opened_at)
                    if (now - t_open).total_seconds() < decay_hours * 3600:
                        recent_trade_found = True
                        break
                except Exception:
                    continue
        if not recent_trade_found and abs(current - self.cfg.min_edge_threshold) > 1e-6:
            # Decay toward baseline
            new_threshold = current - (current - self.cfg.min_edge_threshold) * decay_rate
            logger.info(
                "DECAY: edge_threshold %.4f -> %.4f (no trades in last %dh)",
                current, new_threshold, decay_hours
            )
            self.db.set_heuristic("learned_min_edge_threshold", f"{new_threshold:.4f}")
            return

        if resolved < 10:
            return  # wait for enough evidence

        shadow_win_rate = stats.get("shadow_win_rate", 0.0) or 0.0
        shadow_pnl = stats.get("shadow_pnl", 0.0) or 0.0
        step = 0.0025  # 0.25 percentage points

        new_threshold = current
        direction = "hold"

        # If blocked trades look good, loosen threshold to allow more trades.
        if shadow_win_rate >= 0.60 and shadow_pnl > 0:
            new_threshold = max(0.005, current - step)
            direction = "loosen"
        # If blocked trades look bad, tighten threshold to be more selective.
        elif shadow_win_rate <= 0.45 or shadow_pnl < 0:
            new_threshold = min(0.05, current + step)
            direction = "tighten"

        if abs(new_threshold - current) > 1e-9:
            self.db.set_heuristic("learned_min_edge_threshold", f"{new_threshold:.4f}")
            logger.info(
                "LEARNED: edge_threshold %s %.4f -> %.4f "
                "(shadow_win_rate=%.1f%%, shadow_pnl=$%.2f, resolved=%d)",
                direction,
                current,
                new_threshold,
                shadow_win_rate * 100,
                shadow_pnl,
                resolved,
            )

    def _print_parameters(self) -> None:
        """Print current scan/trade parameters and how they've changed from defaults."""
        cfg = self.cfg

        # Baseline defaults — what we started with
        DEFAULTS = {
            "MIN_VOLUME_24H": 50,
            "MIN_LIQUIDITY_DOLLARS": 100.0,
            "MAX_TIME_TO_RESOLUTION_DAYS": 45,
            "MIN_TIME_TO_RESOLUTION_HOURS": 1,
            "WIDE_SPREAD_CENTS": 4,
            "PRICE_MOVE_THRESHOLD_PCT": 7.0,
            "VOLUME_SPIKE_MULTIPLIER": 2.0,
            "DIVERGENCE_CENTS": 5,
            "MIN_EDGE_THRESHOLD": 0.01,
            "MAX_EXPOSURE_PCT": 0.05,
            "KELLY_FRACTION": 0.25,
            "MAX_OPEN_TRADES": 0,
            "MIN_TRADE_DOLLARS": 1.0,
            "SCAN_INTERVAL_SECONDS": 300,
            "MAX_MARKETS": 5000,
        }

        # Current values
        current = {
            "MIN_VOLUME_24H": cfg.min_volume_24h,
            "MIN_LIQUIDITY_DOLLARS": cfg.min_liquidity_dollars,
            "MAX_TIME_TO_RESOLUTION_DAYS": cfg.max_time_to_resolution_days,
            "MIN_TIME_TO_RESOLUTION_HOURS": cfg.min_time_to_resolution_hours,
            "WIDE_SPREAD_CENTS": cfg.wide_spread_cents,
            "PRICE_MOVE_THRESHOLD_PCT": cfg.price_move_threshold_pct,
            "VOLUME_SPIKE_MULTIPLIER": cfg.volume_spike_multiplier,
            "DIVERGENCE_CENTS": cfg.divergence_cents,
            "MIN_EDGE_THRESHOLD": cfg.min_edge_threshold,
            "MAX_EXPOSURE_PCT": cfg.max_exposure_pct,
            "KELLY_FRACTION": cfg.kelly_fraction,
            "MAX_OPEN_TRADES": cfg.max_open_trades,
            "MIN_TRADE_DOLLARS": cfg.min_trade_dollars,
            "SCAN_INTERVAL_SECONDS": cfg.scan_interval_seconds,
            "MAX_MARKETS": cfg.max_markets,
        }

        logger.info("─── Scan & Trade Parameters ───")
        for key in current:
            cur = current[key]
            base = DEFAULTS[key]
            if cur != base:
                pct = ((cur - base) / base) * 100 if base else 0
                logger.info("  %-30s %10s  (baseline: %s, %+.0f%%)", key, cur, base, pct)
            else:
                logger.info("  %-30s %10s", key, cur)
        
        learned_edge = self.db.get_heuristic("learned_min_edge_threshold")
        if learned_edge:
            logger.info("  LEARNED_MIN_EDGE_THRESHOLD = %s (from DB heuristics)", learned_edge)

        # ── Learned heuristic adjustments from DB ───────────────────
        all_h = self.db.get_all_heuristics()
        if not all_h:
            logger.info("─── Learned Adjustments: none yet (first run) ───")
            return

        logger.info("─── Learned Adjustments ───")

        # Signal win/loss stats
        signal_names = ["wide_spread", "price_move", "volume_spike", "divergence", "new_high_activity"]
        for sig in signal_names:
            wins = int(all_h.get(f"signal_wins_{sig}", "0") or "0")
            losses = int(all_h.get(f"signal_losses_{sig}", "0") or "0")
            total = wins + losses
            if total > 0:
                win_rate = wins / total * 100
                status = "SUPPRESSED" if (total >= 4 and losses / total >= 0.75) else "active"
                logger.info(
                    "  signal %-20s %dW / %dL (%.0f%% win) — %s",
                    sig, wins, losses, win_rate, status,
                )

        # Sentiment discounts
        for k, v in sorted(all_h.items()):
            if k.startswith("sentiment_discount_"):
                cat = k.replace("sentiment_discount_", "")
                logger.info("  sentiment discount %-10s reduced by %.0f%%", cat, float(v) * 100)

        # Avoided categories
        avoided = [k.replace("avoid_category_", "") for k, v in all_h.items()
                   if k.startswith("avoid_category_") and v == "true"]
        if avoided:
            logger.info("  avoided categories: %s", ", ".join(avoided))

        # Loss counts by category
        for k, v in sorted(all_h.items()):
            if k.startswith("losses_") and not k.startswith("losses__"):
                cat = k.replace("losses_", "")
                logger.info("  category %-10s %s losses", cat, v)

        # Category profitability bias
        cat_stats = self.db.get_category_stats()
        if cat_stats:
            logger.info("  ── Category Performance ──")
            for cat, stats in cat_stats.items():
                wins = stats.get("wins", 0) or 0
                losses = stats.get("losses", 0) or 0
                total = wins + losses
                pnl = stats.get("total_pnl", 0) or 0
                wr = stats.get("win_rate", 0) * 100
                # Compute the same bias the scan agent uses
                if total >= 2:
                    rate_factor = (wr / 100 - 0.5) * 2.0
                    pnl_factor = 0.1 if pnl > 0 else -0.1 if pnl < 0 else 0
                    bias = max(0.1, min(1.0 + rate_factor * 0.5 + pnl_factor, 2.0))
                    logger.info(
                        "  %-18s %dW/%dL (%.0f%%) P&L=$%+.2f → bias=%.2f",
                        cat, wins, losses, wr, pnl, bias,
                    )
                else:
                    logger.info(
                        "  %-18s %dW/%dL (%.0f%%) P&L=$%+.2f → (too few trades)",
                        cat, wins, losses, wr, pnl,
                    )

        # Order execution learning
        fills = int(all_h.get("order_fills", "0") or "0")
        misses = int(all_h.get("order_misses", "0") or "0")
        partials = int(all_h.get("order_partials", "0") or "0")
        agg = all_h.get("order_aggression", "0.40")
        if fills + misses > 0:
            fill_rate = fills / (fills + misses) * 100
            logger.info(
                "  order execution: %d fills / %d misses (%d partials) = %.0f%% fill rate, aggression=%.2f",
                fills, misses, partials, fill_rate, float(agg),
            )

        # Market trust learning
        market_trust = all_h.get("market_trust")
        if market_trust:
            logger.info("  market_trust = %s (default 0.50)", market_trust)

        # Kelly fraction learning
        kelly = all_h.get("learned_kelly_fraction")
        if kelly:
            logger.info("  learned_kelly_fraction = %s (default 0.25)", kelly)

        # Loss streak
        streak = all_h.get("loss_streak", "0")
        logger.info("  loss_streak = %s", streak)

        logger.info("─" * 40)


def main() -> None:
    global logger

    cfg = Config()
    logger = setup_logging(
        level=cfg.log_level,
        log_file=cfg.log_file,
        quiet_research_console=cfg.quiet_research_console,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bot = TradingBot(cfg)
    bot.run_loop()


if __name__ == "__main__":
    main()
