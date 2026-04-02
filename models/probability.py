"""
Probability estimation model — estimates "true" probability of a market outcome.

V1: Weighted heuristic model (no training data required).
V2 (future): XGBoost trained on accumulated trade history.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

logger = logging.getLogger("kalshi_bot.probability")


class ProbabilityModel:
    """
    Heuristic probability estimator.

    Adjusts the market-implied probability based on:
      - External sentiment score
      - Narrative confidence
      - Historical heuristics from postmortems

    This is intentionally simple — it can be replaced with a trained
    ML model once enough trade data exists.
    """

    # How much sentiment can shift the probability estimate
    SENTIMENT_WEIGHT = 0.10
    # Confidence multiplier: high=1.0, medium=0.6, low=0.3
    CONFIDENCE_MULTIPLIERS = {"high": 1.0, "medium": 0.6, "low": 0.3}
    # Default market trust: 0.0 = ignore market, 1.0 = fully trust market
    DEFAULT_MARKET_TRUST = 0.5

    def __init__(self, heuristics: dict[str, str] | None = None) -> None:
        self._heuristics = heuristics or {}
        self._load_heuristic_adjustments()

    def _load_heuristic_adjustments(self) -> None:
        """Parse any learned adjustments from the heuristics store."""
        self._adjustments: dict[str, float] = {}
        raw = self._heuristics.get("probability_adjustments")
        if raw:
            try:
                self._adjustments = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        # Load per-category sentiment discounts written by postmortem
        self._sentiment_discounts: dict[str, float] = {}
        for k, v in self._heuristics.items():
            if k.startswith("sentiment_discount_"):
                category = k.replace("sentiment_discount_", "")
                try:
                    self._sentiment_discounts[category] = float(v)
                except (ValueError, TypeError):
                    pass

        # Load market trust weight (how much to trust market-implied probability)
        self._market_trust = self.DEFAULT_MARKET_TRUST
        raw_trust = self._heuristics.get("market_trust")
        if raw_trust:
            try:
                self._market_trust = max(0.0, min(1.0, float(raw_trust)))
            except (ValueError, TypeError):
                pass

    def estimate(
        self,
        market_probability: float,
        sentiment_score: float,
        narrative_confidence: str,
        data_point_count: int = 0,
        market_ticker: str = "",
    ) -> dict[str, Any]:
        """
        Estimate the "true" probability of the YES outcome.

        Args:
            market_probability:  From market mid-price (0–1).
            sentiment_score:     From research agent (-1 to +1).
            narrative_confidence: "low" / "medium" / "high".
            data_point_count:    How many research data points were collected.
            market_ticker:       For ticker-specific heuristic lookup.

        Returns:
            {
                "true_probability": float,
                "edge": float,
                "confidence": str,
                "reasoning": str,
            }
        """
        confidence_mult = self.CONFIDENCE_MULTIPLIERS.get(narrative_confidence, 0.3)

        # Base: blend market price with neutral prior based on trust level
        # trust=1.0 → fully trust market, trust=0.0 → start at 50/50
        true_prob = self._market_trust * market_probability + (1.0 - self._market_trust) * 0.50

        # Adjustment 1: Sentiment-based shift (reduced if sentiment was
        # historically misleading for this market category)
        category = market_ticker[:4] if market_ticker else ""
        discount = self._sentiment_discounts.get(category, 0.0)
        effective_weight = max(0.01, self.SENTIMENT_WEIGHT - discount)
        sentiment_shift = sentiment_score * effective_weight * confidence_mult
        true_prob += sentiment_shift

        # Adjustment 2: Volume of evidence (more data = slightly more trust)
        if data_point_count > 30:
            evidence_bonus = min(0.02, (data_point_count - 30) * 0.001)
            true_prob += evidence_bonus * (1 if sentiment_score > 0 else -1)

        # Adjustment 3: Heuristic adjustments (from postmortems)
        heuristic_adj = self._adjustments.get(market_ticker, 0.0)
        true_prob += heuristic_adj

        # Clamp to [0.01, 0.99]
        true_prob = max(0.01, min(0.99, true_prob))

        edge = true_prob - market_probability

        # Determine overall confidence
        if narrative_confidence == "high" and data_point_count >= 10 and abs(edge) > 0.05:
            confidence = "high"
        elif narrative_confidence == "low" or data_point_count < 5:
            confidence = "low"
        else:
            confidence = "medium"

        reasoning = (
            f"Market implies {market_probability:.1%} (trust={self._market_trust:.2f}). "
            f"Sentiment {sentiment_score:+.3f} (conf={narrative_confidence}) "
            f"shifts estimate by {sentiment_shift:+.3f}. "
            f"Based on {data_point_count} data points. "
            f"Estimated true prob: {true_prob:.1%}. Edge: {edge:+.1%}."
        )

        return {
            "true_probability": round(true_prob, 4),
            "edge": round(edge, 4),
            "confidence": confidence,
            "reasoning": reasoning,
        }
