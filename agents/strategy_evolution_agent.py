"""
STRATEGY EVOLUTION AGENT — B component of the B+C+E system.

Runs periodically (every N cycles) to review what's working and write
DB-side updates that the other agents will pick up next cycle:

  1. Model weights  — updates 'model_weight_adj_<name>' heuristics so
                      the EnsembleProbabilityModel favours sub-models with
                      better recent win rates.

  2. Category profiles — updates 'category_profile_<CAT>' JSON blobs in
                         the heuristics table to tighten/loosen kelly_fraction
                         and edge_threshold for each category based on P&L.

  3. Scan priorities   — writes 'category_scan_priority_<CAT>' heuristics
                         so ScanAgent pushes winning categories to the top
                         and demotes persistent losers.

The agent intentionally does NOT touch code defaults in category_profiles.py.
All mutations go through the heuristics DB so they are fully inspectable
and can be cleared by deleting the relevant heuristic keys.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import Config
from database import Database
from models.category_profiles import (
    CODE_DEFAULT_PROFILES,
    CategoryProfileLoader,
    KNOWN_CATEGORIES,
)

logger = logging.getLogger("kalshi_bot.strategy_evolution")

# Minimum resolved trades before touching a category's profile
MIN_TRADES_FOR_EVOLUTION = 5

# How many cycles between full evolution passes
EVOLUTION_INTERVAL_CYCLES = 10

# Scan priority tiers (multiplier applied on top of category_bias)
PRIORITY_HIGH  = 1.5   # strong winner
PRIORITY_MED   = 1.0   # neutral
PRIORITY_LOW   = 0.5   # underperforming
PRIORITY_AVOID = 0.0   # flagged for avoidance (postmortem already handles this)


class StrategyEvolutionAgent:
    """Periodic strategy review and parameter evolution."""

    def __init__(self, cfg: Config, db: Database) -> None:
        self._cfg = cfg
        self._db = db
        self._profile_loader = CategoryProfileLoader(db)
        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Main entry — called every cycle from main.py
    # ------------------------------------------------------------------
    def maybe_evolve(self) -> None:
        """Run a full evolution pass every EVOLUTION_INTERVAL_CYCLES cycles."""
        self._cycle_count += 1
        if self._cycle_count % EVOLUTION_INTERVAL_CYCLES != 0:
            return
        logger.info("=== STRATEGY EVOLUTION: running full review (cycle %d) ===",
                    self._cycle_count)
        self.run_evolution()

    def run_evolution(self) -> None:
        self._update_model_weights()
        self._review_category_profiles()
        self._update_scan_priorities()
        logger.info("Strategy evolution complete.")

    # ------------------------------------------------------------------
    # 1. Model weight updates
    # ------------------------------------------------------------------
    def _update_model_weights(self) -> None:
        """
        For each sub-model read model_wins/losses from heuristics.
        Write 'model_weight_adj_<name>' as a float multiplier.
        The EnsembleProbabilityModel already applies this via the heuristics
        dict (win_rate / 0.5, clamped 0.4–2.0).  This method just logs
        the current state so operators can see what's happening.
        """
        perf = self._db.get_all_model_performance()
        for model_name, stats in perf.items():
            total = stats.get("total", 0)
            if total < 5:
                continue
            win_rate = stats.get("win_rate", 0.5)
            adj = max(0.4, min(2.0, win_rate / 0.5))
            direction = "↑" if adj > 1.0 else ("↓" if adj < 1.0 else "=")
            logger.info(
                "  Model %-22s WR=%.0f%% (%d trades) weight_adj=%.2f %s",
                model_name, win_rate * 100, total, adj, direction,
            )

    # ------------------------------------------------------------------
    # 2. Category profile updates
    # ------------------------------------------------------------------
    def _review_category_profiles(self) -> None:
        """
        For each category with enough trade history:
          - Underperforming (win_rate < 35% or P&L deeply negative):
              reduce kelly_fraction by 20%, raise edge_threshold by 25%,
              set is_underperforming = True
          - Recovering (was underperforming, now win_rate > 45%):
              clear underperforming flag
          - Outperforming (win_rate > 65% and P&L positive):
              boost kelly_fraction by 10%, lower edge_threshold by 10%
        """
        cat_stats = self._db.get_category_stats()

        for category, stats in cat_stats.items():
            wins   = stats.get("wins", 0) or 0
            losses = stats.get("losses", 0) or 0
            total  = wins + losses
            pnl    = stats.get("total_pnl", 0) or 0

            if total < MIN_TRADES_FOR_EVOLUTION:
                continue

            win_rate = wins / total

            # Load current effective profile (code default + any existing override)
            profile = self._profile_loader.get_profile(category)
            overrides: dict[str, Any] = {}

            # Read existing override blob if any
            raw = self._db.get_heuristic(f"category_profile_{category.upper()}")
            if raw:
                try:
                    overrides = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    overrides = {}

            changed = False

            if win_rate < 0.35 or (pnl < -5.0 and win_rate < 0.45):
                # Underperforming
                if not overrides.get("is_underperforming"):
                    logger.info(
                        "EVOLUTION: %s underperforming (WR=%.0f%%, P&L=$%.2f) "
                        "— tightening parameters",
                        category, win_rate * 100, pnl,
                    )
                    overrides["is_underperforming"] = True
                    overrides["kelly_fraction"] = round(
                        max(0.05, profile.kelly_fraction * 0.80), 3
                    )
                    overrides["edge_threshold"] = round(
                        min(0.05, profile.edge_threshold * 1.25), 4
                    )
                    changed = True

            elif overrides.get("is_underperforming") and win_rate > 0.45:
                # Recovering — clear flag, restore toward defaults
                logger.info(
                    "EVOLUTION: %s recovering (WR=%.0f%%) — clearing underperforming flag",
                    category, win_rate * 100,
                )
                overrides["is_underperforming"] = False
                # Partially relax back toward code default
                code_default = CODE_DEFAULT_PROFILES.get(category.upper(),
                               CODE_DEFAULT_PROFILES["DEFAULT"])
                current_kelly = overrides.get("kelly_fraction", profile.kelly_fraction)
                overrides["kelly_fraction"] = round(
                    min(code_default.kelly_fraction, current_kelly * 1.10), 3
                )
                changed = True

            elif win_rate > 0.65 and pnl > 0 and total >= 8:
                # Outperforming — modestly loosen parameters
                code_default = CODE_DEFAULT_PROFILES.get(category.upper(),
                               CODE_DEFAULT_PROFILES["DEFAULT"])
                new_kelly = round(
                    min(code_default.kelly_fraction * 1.20, profile.kelly_fraction * 1.10), 3
                )
                new_edge = round(
                    max(code_default.edge_threshold * 0.80, profile.edge_threshold * 0.90), 4
                )
                if abs(new_kelly - profile.kelly_fraction) > 0.005 or \
                   abs(new_edge - profile.edge_threshold) > 0.001:
                    logger.info(
                        "EVOLUTION: %s outperforming (WR=%.0f%%, P&L=$%.2f) "
                        "— loosening kelly %.3f→%.3f edge %.4f→%.4f",
                        category, win_rate * 100, pnl,
                        profile.kelly_fraction, new_kelly,
                        profile.edge_threshold, new_edge,
                    )
                    overrides["kelly_fraction"] = new_kelly
                    overrides["edge_threshold"] = new_edge
                    changed = True

            if changed:
                self._db.set_heuristic(
                    f"category_profile_{category.upper()}",
                    json.dumps(overrides),
                )

    # ------------------------------------------------------------------
    # 3. Scan priority updates
    # ------------------------------------------------------------------
    def _update_scan_priorities(self) -> None:
        """
        Write 'category_scan_priority_<CAT>' heuristics so that
        ScanAgent multiplies candidate scores by this value.

        Tiers:
          HIGH  (WR > 60% and P&L > $5 and total >= 5)  → 1.5
          LOW   (WR < 35% or P&L < -$3)                  → 0.5
          MED   (everything else)                         → 1.0
          AVOID (flagged by postmortem)                   → 0.0 (already handled
                                                              via avoid_category_*)
        """
        cat_stats = self._db.get_category_stats()

        for category, stats in cat_stats.items():
            wins   = stats.get("wins", 0) or 0
            losses = stats.get("losses", 0) or 0
            total  = wins + losses
            pnl    = stats.get("total_pnl", 0) or 0

            if total < 3:
                # Not enough data — neutral priority
                priority = PRIORITY_MED
            elif wins / total > 0.60 and pnl > 5.0 and total >= 5:
                priority = PRIORITY_HIGH
            elif wins / total < 0.35 or pnl < -3.0:
                priority = PRIORITY_LOW
            else:
                priority = PRIORITY_MED

            # Don't override explicit avoidance
            avoid_raw = self._db.get_heuristic(f"avoid_category_{category}")
            if avoid_raw == "true":
                priority = PRIORITY_AVOID

            current_raw = self._db.get_heuristic(f"category_scan_priority_{category}")
            try:
                current = float(current_raw) if current_raw else PRIORITY_MED
            except (ValueError, TypeError):
                current = PRIORITY_MED

            if abs(priority - current) > 0.05:
                tier_name = {
                    PRIORITY_HIGH:  "HIGH",
                    PRIORITY_MED:   "MED",
                    PRIORITY_LOW:   "LOW",
                    PRIORITY_AVOID: "AVOID",
                }.get(priority, "MED")
                logger.info(
                    "EVOLUTION: %s scan priority %.1f → %.1f (%s) "
                    "[WR=%.0f%% total=%d P&L=$%.2f]",
                    category, current, priority, tier_name,
                    (wins / total * 100) if total > 0 else 0, total, pnl,
                )
                self._db.set_heuristic(
                    f"category_scan_priority_{category}", str(priority)
                )

    # ------------------------------------------------------------------
    # Summary log — called on startup to show current evolution state
    # ------------------------------------------------------------------
    def log_status(self) -> None:
        """Log current category profile overrides and scan priorities."""
        logger.info("─── Strategy Evolution Status ───")
        cat_stats = self._db.get_category_stats()

        for category in sorted(set(list(cat_stats.keys()) + KNOWN_CATEGORIES)):
            profile = self._profile_loader.get_profile(category)
            stats   = cat_stats.get(category, {})
            wins    = stats.get("wins", 0) or 0
            losses  = stats.get("losses", 0) or 0
            total   = wins + losses
            pnl     = stats.get("total_pnl", 0) or 0
            priority_raw = self._db.get_heuristic(f"category_scan_priority_{category}")
            priority = float(priority_raw) if priority_raw else 1.0

            flag = " [UNDERPERFORMING]" if profile.is_underperforming else ""
            logger.info(
                "  %-14s kelly=%.2f edge=%.3f priority=%.1f | "
                "%dW/%dL P&L=$%.2f%s",
                category,
                profile.kelly_fraction,
                profile.edge_threshold,
                priority,
                wins, losses, pnl,
                flag,
            )
