"""
Ensemble Probability Model — C component of the B+C+E system.

Sub-models each estimate the true probability of YES.
The EnsembleProbabilityModel combines them with learned per-model weights.

Sub-models
----------
CalibrationModel   — ML model trained on 10,000+ resolved Kalshi markets.
                     Highest-signal model — knows where markets systematically misprice.
MarketAnchorModel  — Returns the market price as the baseline estimate.
SentimentModel     — Adjusts probability based on external sentiment.
MomentumModel      — Uses price movement from baseline to detect directional drift.
VolumeModel        — High volume spikes confirm the current market direction.
ConsensusModel     — Uses bullish vs bearish source agreement from research.
SportsbookModel    — Uses sportsbook lines as an independent pricing signal.

DESIGN PRINCIPLE: The market price is assumed correct by default.  Edges
come ONLY from information signals (calibration model, sentiment, sportsbook
data, momentum) that move the estimate away from market price.  No model
should pull prices toward 50% — that creates phantom edges on every
non-50c contract.

Weight learning
---------------
After each resolved trade, postmortem_agent increments:
    heuristic "model_wins_<ModelName>"   — model was on the right side
    heuristic "model_losses_<ModelName>" — model was on the wrong side

EnsembleProbabilityModel reads these at instantiation and multiplies the
base weights (from CategoryProfile.active_models) by a win-rate adjustment
factor, clamped to [0.4x, 2.0x].  Models with < MIN_TRADES_FOR_LEARNING
resolved trades keep their base weight unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from models.category_profiles import CategoryProfile

logger = logging.getLogger("kalshi_bot.ensemble")

# Require this many resolved trades before adjusting a model's weight
MIN_TRADES_FOR_LEARNING = 5

# Confidence multipliers for narrative_confidence string
_CONF_MULT: dict[str, float] = {"high": 1.0, "medium": 0.6, "low": 0.3}


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class MarketAnchorModel:
    """
    Returns the market-implied probability as the baseline estimate.

    The market price IS the anchor.  This model does NOT blend toward 50%.
    That old formula (trust * market + (1-trust) * 0.50) created phantom
    edges on every contract not near 50c — a 5c contract got predicted
    at 30c, showing a fake +25% edge.

    Now: prob = market_prob.  Period.  Confidence scales with market_trust
    so that in efficient categories (FINANCIALS, trust=0.85) this model
    dominates the ensemble and prevents other models from moving the
    estimate far.  In less efficient categories (ENTERTAINMENT, trust=0.55)
    it has less weight, letting sentiment/sportsbook models contribute.
    """

    name = "MarketAnchorModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp = inputs["market_probability"]
        trust = inputs.get("market_trust", 0.70)
        confidence = trust  # high trust = high confidence in market price
        return {"model": self.name, "probability": _clamp(mp), "confidence": confidence}


class SentimentModel:
    """
    Adjusts market probability based on external sentiment.

    Formula:
        shift = sentiment_score * sentiment_weight * confidence_multiplier
        prob  = market_prob + shift  (capped at ±3pp)

    High narrative confidence amplifies the shift; low confidence dampens it.
    Returns market_prob (confidence=0) when sentiment data is weak.
    """

    name = "SentimentModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp   = inputs["market_probability"]
        sent = inputs.get("sentiment_score", 0.0)
        sw   = inputs.get("sentiment_weight", 0.10)
        conf = inputs.get("narrative_confidence", "low")
        mult = _CONF_MULT.get(conf, 0.3)

        shift = sent * sw * mult
        # Cap at ±3pp — sentiment alone shouldn't create a large edge
        shift = max(-0.03, min(0.03, shift))
        prob = mp + shift

        raw_conf = abs(sent) * mult
        confidence = min(0.6, max(0.05, raw_conf))
        return {"model": self.name, "probability": _clamp(prob), "confidence": confidence}


class MomentumModel:
    """
    Detects directional drift by comparing current price to baseline.

    Formula:
        price_change_pct = (current - baseline) / baseline
        momentum_adj = price_change_pct * 0.15  (capped at ±0.05)
        prob = market_prob + momentum_adj

    Rationale: a market moving strongly in one direction is more likely to
    continue in that direction than to reverse — especially near resolution.
    """

    name = "MomentumModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp       = inputs["market_probability"]
        baseline = inputs.get("baseline_price")
        current  = inputs.get("current_price", mp)

        if not baseline or baseline <= 0:
            return {"model": self.name, "probability": _clamp(mp), "confidence": 0.1}

        pct_change = (current - baseline) / baseline
        momentum_adj = max(-0.05, min(0.05, pct_change * 0.15))
        prob = mp + momentum_adj

        confidence = min(0.8, abs(pct_change) * 3.0)
        return {"model": self.name, "probability": _clamp(prob), "confidence": confidence}


class VolumeModel:
    """
    High volume confirms the current market direction; low volume is neutral.

    Formula:
        if volume_ratio > 1.5:
            direction = +1 if market_prob > 0.5 else -1
            vol_adj = min(0.03, (volume_ratio - 1.0) * 0.02) * direction
        prob = market_prob + vol_adj

    Rationale: unusually high trading activity means new information is
    being priced in — the current price direction is likely correct.
    """

    name = "VolumeModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp    = inputs["market_probability"]
        ratio = inputs.get("volume_ratio", 1.0) or 1.0

        if ratio <= 1.2:
            return {"model": self.name, "probability": _clamp(mp), "confidence": 0.1}

        direction = 1.0 if mp >= 0.5 else -1.0
        vol_adj = min(0.03, (ratio - 1.0) * 0.02) * direction
        prob = mp + vol_adj

        confidence = min(0.7, (ratio - 1.0) * 0.3)
        return {"model": self.name, "probability": _clamp(prob), "confidence": confidence}


class SportsbookModel:
    """
    Compares the sportsbook consensus implied probability to the Kalshi price.

    The gap between them represents the fan-bias inefficiency in prediction
    markets: retail bettors over-back underdogs, so favorites' Kalshi prices
    are chronically suppressed below their true probability.

    Formula:
        gap = sportsbook_prob - market_prob
        prob = market_prob + gap * 0.70

    The 0.70 factor means we lean 70% toward the sportsbook and keep 30%
    weight on the Kalshi price (in case the gap is noise, not signal).
    Confidence scales with gap size — a 10-point gap is high confidence,
    a 1-point gap is low confidence.

    Returns market_prob unchanged (confidence=0) when no sportsbook data.
    """

    name = "SportsbookModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp = inputs["market_probability"]
        sb_prob = inputs.get("sportsbook_prob")

        if sb_prob is None:
            return {"model": self.name, "probability": _clamp(mp), "confidence": 0.0}

        gap = sb_prob - mp
        adjusted = mp + gap * 0.70
        # Confidence grows with gap size: 5% gap → 0.20, 15% gap → 0.60, 25%+ → 0.90
        confidence = min(0.90, abs(gap) * 3.5)

        return {"model": self.name, "probability": _clamp(adjusted), "confidence": confidence}


class ConsensusModel:
    """
    Uses the bullish vs bearish split across news/social sources.

    Formula:
        consensus_score = (bullish_pct - bearish_pct)   # -1 to +1 range
        consensus_adj   = consensus_score * 0.05  (capped at ±4pp)
        prob = market_prob + consensus_adj

    Requires 5+ data points to have meaningful confidence.
    """

    name = "ConsensusModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp          = inputs["market_probability"]
        bullish_pct = inputs.get("bullish_pct", 0.0) or 0.0
        bearish_pct = inputs.get("bearish_pct", 0.0) or 0.0
        data_points = inputs.get("data_point_count", 0) or 0

        if data_points < 5:
            return {"model": self.name, "probability": _clamp(mp), "confidence": 0.05}

        consensus = bullish_pct - bearish_pct          # -1.0 to +1.0
        consensus_adj = max(-0.04, min(0.04, consensus * 0.05))
        prob = mp + consensus_adj

        confidence = min(0.6, data_points / 40.0)
        return {"model": self.name, "probability": _clamp(prob), "confidence": confidence}


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

_ALL_MODELS: dict[str, Any] = {
    "MarketAnchorModel": MarketAnchorModel(),
    "SentimentModel":    SentimentModel(),
    "MomentumModel":     MomentumModel(),
    "VolumeModel":       VolumeModel(),
    "ConsensusModel":    ConsensusModel(),
    "SportsbookModel":   SportsbookModel(),
}

# CalibrationModel is optional — only registered if the trained model exists.
# This avoids import errors when lightgbm/joblib aren't installed.
try:
    from models.calibration import CalibrationModel as _CalibModel
    if _CalibModel().estimate({"market_probability": 0.5}).get("confidence", 0) > 0:
        _ALL_MODELS["CalibrationModel"] = _CalibModel()
        logger.info("CalibrationModel registered in ensemble (ML model loaded)")
    else:
        _ALL_MODELS["CalibrationModel"] = _CalibModel()
        logger.info("CalibrationModel registered (model file not yet trained — will return market price)")
except ImportError:
    logger.info("CalibrationModel not available (lightgbm/joblib not installed)")
except Exception as exc:
    logger.warning("CalibrationModel failed to load: %s", exc)


class EnsembleProbabilityModel:
    """
    Weighted average of active sub-models.

    At construction, weights are:
        effective_weight = base_weight * win_rate_adj
    where win_rate_adj = clamp(win_rate / 0.50, 0.4, 2.0)
    for models with >= MIN_TRADES_FOR_LEARNING resolved trades.

    Models not in profile.active_models are excluded entirely.
    """

    def __init__(self, profile: CategoryProfile, heuristics: dict[str, str]) -> None:
        self._profile = profile
        self._weights = self._compute_weights(profile, heuristics)

    # ------------------------------------------------------------------
    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run each active sub-model and return a weighted-average probability.

        Returns a dict compatible with the old ProbabilityModel.estimate() shape:
            {
                true_probability: float,
                edge: float,
                confidence: str ("low" / "medium" / "high"),
                reasoning: str,
                model_votes: list[{model, probability, weight}],
            }
        """
        market_prob = inputs["market_probability"]
        votes: list[dict] = []

        total_weight = 0.0
        weighted_prob = 0.0

        for model_name, weight in self._weights.items():
            model = _ALL_MODELS.get(model_name)
            if model is None:
                continue
            result = model.estimate(inputs)
            votes.append({
                "model":       model_name,
                "probability": result["probability"],
                "weight":      round(weight, 4),
                "confidence":  result["confidence"],
            })
            weighted_prob += result["probability"] * weight
            total_weight  += weight

        if total_weight == 0:
            # Fallback: trust market price
            true_prob = market_prob
        else:
            true_prob = weighted_prob / total_weight

        true_prob = _clamp(true_prob)
        edge = true_prob - market_prob

        # Overall confidence from weighted average of model confidences
        if votes and total_weight > 0:
            avg_conf = sum(
                v["confidence"] * self._weights.get(v["model"], 0.0)
                for v in votes
            ) / total_weight
        else:
            avg_conf = 0.3

        if avg_conf >= 0.65:
            confidence = "high"
        elif avg_conf >= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        # Build human-readable reasoning
        vote_strs = [
            f"{v['model']}={v['probability']:.3f}(w={v['weight']:.2f})"
            for v in votes
        ]
        reasoning = (
            f"Ensemble [{self._profile.category}]: {' | '.join(vote_strs)}. "
            f"Market={market_prob:.3f} → True={true_prob:.3f} Edge={edge:+.3f}."
        )

        return {
            "true_probability": round(true_prob, 4),
            "edge":             round(edge, 4),
            "confidence":       confidence,
            "reasoning":        reasoning,
            "model_votes":      votes,
        }

    # ------------------------------------------------------------------
    def _compute_weights(
        self, profile: CategoryProfile, heuristics: dict[str, str]
    ) -> dict[str, float]:
        """
        Build the effective weight dict for active models.

        Starts from profile.active_models, then scales by win-rate adjustment.
        Normalises so weights sum to 1.0.
        """
        raw: dict[str, float] = {}

        for model_name, base_weight in profile.active_models.items():
            wins_raw   = heuristics.get(f"model_wins_{model_name}")
            losses_raw = heuristics.get(f"model_losses_{model_name}")
            wins   = int(wins_raw)   if wins_raw   and wins_raw.isdigit()   else 0
            losses = int(losses_raw) if losses_raw and losses_raw.isdigit() else 0
            total = wins + losses

            if total >= MIN_TRADES_FOR_LEARNING:
                win_rate = wins / total
                # Scale: 50% WR → 1.0x, 70% WR → 1.4x, 30% WR → 0.6x, etc.
                adj = win_rate / 0.50
                adj = max(0.4, min(2.0, adj))
                effective = base_weight * adj
                if abs(adj - 1.0) > 0.05:
                    logger.debug(
                        "  %s weight adj: %.2f→%.2f (WR=%.0f%% over %d trades)",
                        model_name, base_weight, effective, win_rate * 100, total,
                    )
            else:
                effective = base_weight

            raw[model_name] = max(0.0, effective)

        # Normalise to sum to 1.0
        total_w = sum(raw.values())
        if total_w == 0:
            # Fallback: equal weight over all active models
            n = len(raw) or 1
            return {k: 1.0 / n for k in raw}

        return {k: v / total_w for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(p: float) -> float:
    return max(0.01, min(0.99, p))
