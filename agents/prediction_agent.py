"""
PREDICTION AGENT — detects edges by comparing estimated true probability
to the market-implied probability.

Now uses the B+C+E ensemble system:
  - Category profile (E) sets which models are active, sentiment weight,
    market trust, and the per-category edge threshold.
  - EnsembleProbabilityModel (C) runs 2-5 sub-models weighted by their
    recent win rates.  StrategyEvolutionAgent (B) tunes weights over time.

Quality score (0–100) gates trades beyond just edge:
  - Edge strength        (0–25 pts)
  - Signal count         (0–20 pts)
  - Research quality     (0–20 pts)
  - Sentiment alignment  (0–15 pts)
  - Category track record(0–10 pts)
  - Market quality       (0–10 pts)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config import Config  # noqa: F401 (resolved at runtime via sys.path)
from database import Database
from models.category_profiles import CategoryProfile, CategoryProfileLoader
from models.ensemble import EnsembleProbabilityModel
from utils import CandidateMarket, PredictionResult, ResearchResult

logger = logging.getLogger("kalshi_bot.prediction_agent")

# Minimum quality score to consider a trade (0-100 scale)
MIN_QUALITY_SCORE = 35


class PredictionAgent:
    def __init__(self, cfg: Config, db: Database) -> None:
        self._cfg = cfg
        self._db = db
        self._profile_loader = CategoryProfileLoader(db)

    def _get_ensemble(self, category: str) -> EnsembleProbabilityModel:
        """Build a fresh ensemble for the given category using latest heuristics."""
        profile = self._profile_loader.get_profile(category)
        heuristics = self._db.get_all_heuristics()
        return EnsembleProbabilityModel(profile=profile, heuristics=heuristics)

    def _get_profile(self, candidate: CandidateMarket) -> CategoryProfile:
        """Resolve category profile for a candidate market."""
        category = self._profile_loader.category_from_market(
            candidate.market.category, candidate.market.ticker
        )
        return self._profile_loader.get_profile(category)

    @staticmethod
    def _days_to_resolution(close_time: str | None) -> float | None:
        """Return days until market closes, or None if close_time is missing/unparseable."""
        if not close_time:
            return None
        try:
            ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0.0, (ct - now).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _apply_longshot_bias(edge: float, market_prob: float) -> float:
        """
        Adjust edge for the well-documented longshot bias in prediction markets.

        Research (Becker 2026, 72M Kalshi trades) shows:
          - YES contracts below 15¢ have -41% EV (retail "hope" premium)
          - NO contracts below 15¢ have +23% EV (fading retail optimism)
          - The bias is strongest at the most extreme prices

        This adjustment penalizes YES bets on longshots and rewards NO bets,
        with symmetric logic for near-certainties (>85¢).
        """
        if market_prob < 0.15:
            # Longshot zone: YES is overpriced by retail optimists
            extremity = (0.15 - market_prob) / 0.15  # 0→1 as price drops
            if edge > 0:
                # YES on longshot — penalize (retail trap)
                edge *= (1.0 - extremity * 0.60)  # up to 60% edge reduction
            else:
                # NO on longshot — boost (fading retail optimism = +EV)
                edge *= (1.0 + extremity * 0.30)  # up to 30% edge boost
        elif market_prob > 0.85:
            # Near-certainty zone: NO buyers are fading favorites (also a trap)
            extremity = (market_prob - 0.85) / 0.15  # 0→1 as price rises
            if edge < 0:
                # NO on favorite — penalize
                edge *= (1.0 - extremity * 0.40)
            else:
                # YES on favorite — scalping, slight boost
                edge *= (1.0 + extremity * 0.15)

        return edge

    @staticmethod
    def _time_decay_edge_multiplier(days: float | None) -> float:
        """
        Returns an edge threshold multiplier based on days to resolution.

        Rationale:
          < 0.5 days  — market nearly resolved, pricing is very efficient → 2.0x
          0.5-1 day   — still efficient, almost done              → 1.5x
          1-3 days    — sweet spot: uncertainty remains, good edge zone → 1.0x
          3-7 days    — allow slightly lower bar, thesis has time to play out → 0.85x
          7-30 days   — longer horizon, harder to predict         → 1.10x
          30+ days    — high noise, require more conviction       → 1.30x
        """
        if days is None:
            return 1.0
        if days < 0.5:
            return 2.0
        if days < 1.0:
            return 1.5
        if days < 3.0:
            return 1.0
        if days < 7.0:
            return 0.85
        if days < 30.0:
            return 1.10
        return 1.30

    def _effective_edge_threshold(self, profile: CategoryProfile) -> float:
        """
        Return the stricter of:
          - The category profile's edge_threshold
          - Any globally learned threshold from DB (postmortem-driven)
        """
        # Global learned override
        raw = self._db.get_heuristic("learned_min_edge_threshold")
        learned = None
        if raw is not None:
            try:
                learned = max(0.005, min(0.05, float(raw)))
            except (ValueError, TypeError):
                pass

        base = profile.edge_threshold
        if learned is not None:
            return max(base, learned)
        return base

    def _compute_quality_score(
        self,
        candidate: CandidateMarket,
        research: ResearchResult,
        edge: float,
        profile: CategoryProfile,
    ) -> tuple[float, str]:
        """
        Compute a multi-factor quality score (0-100) for a trade candidate.
        Returns (score, breakdown_string).
        """
        scores: dict[str, float] = {}

        # ── Factor 1: Edge Strength (0-25 pts) ─────────────────────
        edge_pts = min(25.0, abs(edge) * 500)
        scores["edge"] = edge_pts

        # ── Factor 2: Signal Count & Diversity (0-20 pts) ──────────
        signals = candidate.signals
        # Preferred signals for this category get a bonus
        preferred_hits = sum(
            1 for s in signals if any(s.startswith(p) for p in profile.preferred_signals)
        )
        unique_types = len(set(s.split("(")[0] for s in signals))
        signal_pts = min(20.0, len(signals) * 4 + unique_types * 2 + preferred_hits * 3)
        scores["signals"] = signal_pts

        # ── Factor 3: Research Quality (0-20 pts) ──────────────────
        research_pts = 0.0
        if research.data_point_count >= 20:
            research_pts += 8
        elif research.data_point_count >= 10:
            research_pts += 5
        elif research.data_point_count >= 5:
            research_pts += 2
        conf_map = {"high": 7, "medium": 4, "low": 1}
        research_pts += conf_map.get(research.narrative_confidence, 1)
        research_pts += min(5.0, abs(research.sentiment_score) * 10)
        scores["research"] = min(20.0, research_pts)

        # ── Factor 4: Sentiment Alignment (0-15 pts) ───────────────
        alignment_pts = 0.0
        if edge > 0 and research.sentiment_score > 0:
            alignment_pts = min(15.0, research.sentiment_score * 15)
        elif edge < 0 and research.sentiment_score < 0:
            alignment_pts = min(15.0, abs(research.sentiment_score) * 15)
        elif abs(research.sentiment_score) < 0.05:
            alignment_pts = 5.0  # neutral — doesn't contradict
        else:
            alignment_pts = 0.0  # disagreement — penalise
        scores["alignment"] = alignment_pts

        # ── Factor 5: Category Track Record (0-10 pts) ─────────────
        cat = candidate.market.category or candidate.market.ticker[:7]
        cat_stats = self._db.get_category_stats()
        cat_pts = 5.0
        if cat in cat_stats:
            cs = cat_stats[cat]
            wins = cs.get("wins", 0) or 0
            losses = cs.get("losses", 0) or 0
            total = wins + losses
            if total >= 3:
                win_rate = wins / total
                cat_pts = min(10.0, win_rate * 12.5)
                if win_rate < 0.30:
                    cat_pts = 0.0
        scores["category"] = cat_pts

        # ── Factor 6: Market Quality (0-10 pts) ────────────────────
        # Mid-tier volume (5k-50k) is the sweet spot: liquid enough to
        # fill but not so active that MMs have arb'd out all edge.
        mkt = candidate.market
        mq_pts = 0.0
        if 5_000 <= mkt.volume_24h < 50_000:
            mq_pts += 5  # sweet spot — best taker edge
        elif mkt.volume_24h >= 500:
            mq_pts += 3
        elif mkt.volume_24h >= 100:
            mq_pts += 1
        spread_cents = mkt.spread * 100
        if spread_cents <= 3:
            mq_pts += 3
        elif spread_cents <= 6:
            mq_pts += 1
        if mkt.yes_bid > 0 and mkt.yes_ask > 0:
            mq_pts += 2
        scores["market_quality"] = min(10.0, mq_pts)

        total_score = sum(scores.values())
        breakdown = " | ".join(f"{k}={v:.0f}" for k, v in scores.items())
        return round(total_score, 1), breakdown

    def predict(
        self,
        candidate: CandidateMarket,
        research: ResearchResult,
    ) -> PredictionResult | None:
        """
        Estimate true probability via ensemble and compute edge + quality score.
        Returns PredictionResult if trade quality is sufficient, else None.
        """
        ticker = candidate.market.ticker
        market_prob = candidate.market.implied_probability

        # Direction-aware extreme price filter.
        #
        # ALLOWED — scalping the favourite:
        #   market_prob > 0.92, edge > 0  → bet YES on a near-certain event
        #   market_prob < 0.08, edge < 0  → bet NO on a near-impossible event
        #   Both are fine: high win rate, predictable outcome.
        #
        # BLOCKED — fading the favourite (longshot gambling):
        #   market_prob > 0.92, edge < 0  → bet NO on a 95% market (5¢ to lose 95¢)
        #   market_prob < 0.08, edge > 0  → bet YES on a 2% market (2¢ to lose 98¢)
        #   These have terrible risk/reward and the model cannot reliably
        #   detect true mis-pricing at extremes — it's just noise.
        #
        # We don't know edge yet, so we defer this check to after estimation.
        # Flag it here for use below.
        _at_high_extreme = market_prob > 0.92
        _at_low_extreme  = market_prob < 0.08

        # Resolve category and load strategy profile
        category = self._profile_loader.category_from_market(
            candidate.market.category, ticker
        )
        profile = self._profile_loader.get_profile(category)

        logger.info(
            "Predicting: %s [%s] | market_prob=%.2f | sentiment=%.3f",
            ticker, category, market_prob, research.sentiment_score,
        )

        # Build input dict for ensemble sub-models
        baseline_price = candidate.baseline_price or market_prob
        baseline_volume = candidate.baseline_volume or 1.0
        current_volume = candidate.market.volume_24h or 0.0
        volume_ratio = (current_volume / baseline_volume) if baseline_volume > 0 else 1.0

        inputs: dict[str, Any] = {
            "market_probability":   market_prob,
            "market_trust":         profile.market_trust,
            "sentiment_score":      research.sentiment_score,
            "sentiment_weight":     profile.sentiment_weight,
            "narrative_confidence": research.narrative_confidence,
            "data_point_count":     research.data_point_count,
            "market_ticker":        ticker,
            "bullish_pct":          research.bullish_pct,
            "bearish_pct":          research.bearish_pct,
            "baseline_price":       baseline_price,
            "current_price":        candidate.market.last_price or market_prob,
            "volume_ratio":         volume_ratio,
            "sportsbook_prob":      research.sportsbook_prob,  # None for non-SPORTS
            # Fields for CalibrationModel
            "category":             category,
            "spread":               candidate.market.spread,
            "volume_24h":           candidate.market.volume_24h,
            "open_interest":        candidate.market.open_interest,
            "volume":               candidate.market.volume_24h,  # best proxy available
            "yes_bid":              candidate.market.yes_bid,
            "yes_ask":              candidate.market.yes_ask,
            "no_bid":               candidate.market.no_bid,
            "no_ask":               candidate.market.no_ask,
        }

        # Run ensemble
        ensemble = self._get_ensemble(category)
        estimate = ensemble.estimate(inputs)

        true_prob   = estimate["true_probability"]
        raw_edge    = estimate["edge"]
        confidence  = estimate["confidence"]
        reasoning   = estimate["reasoning"]
        model_votes = estimate.get("model_votes", [])

        # Apply longshot bias adjustment — penalise YES longshots, reward NO longshots
        edge = self._apply_longshot_bias(raw_edge, market_prob)
        if abs(edge - raw_edge) > 0.001:
            reasoning += f" | longshot_adj: {raw_edge:+.4f}→{edge:+.4f}"

        # Multi-factor quality score
        quality_score, quality_breakdown = self._compute_quality_score(
            candidate, research, edge, profile
        )

        edge_threshold = self._effective_edge_threshold(profile)
        days = self._days_to_resolution(candidate.market.close_time)
        edge_threshold *= self._time_decay_edge_multiplier(days)

        # Gate: direction-aware extreme price block
        # Fading a 92%+ favourite (betting NO) or backing a 8%- longshot (betting YES)
        # is blocked — model noise at extremes, terrible risk/reward.
        # Scalping the favourite in the correct direction is allowed.
        if (_at_high_extreme and edge < 0) or (_at_low_extreme and edge > 0):
            logger.info(
                "  %s [%s]: market_prob=%.3f edge=%+.3f — fading extreme, blocked",
                ticker, category, market_prob, edge,
            )
            return None

        # Gate: minimum edge threshold (quality score can override for near-misses)
        if abs(edge) < edge_threshold:
            if quality_score < 60:
                logger.info(
                    "  %s [%s]: edge %.3f below threshold %.3f, quality=%.0f — skipping",
                    ticker, category, edge, edge_threshold, quality_score,
                )
                return None
            else:
                logger.info(
                    "  %s [%s]: edge %.3f below threshold but quality=%.0f overrides",
                    ticker, category, edge, quality_score,
                )

        # Gate: minimum quality score
        if quality_score < MIN_QUALITY_SCORE:
            logger.info(
                "  %s [%s]: quality_score=%.0f below minimum %d — skipping [%s]",
                ticker, category, quality_score, MIN_QUALITY_SCORE, quality_breakdown,
            )
            return None

        # Gate: low confidence + low quality
        if confidence == "low" and quality_score < 50:
            logger.info(
                "  %s [%s]: confidence low and quality=%.0f — skipping",
                ticker, category, quality_score,
            )
            return None

        result = PredictionResult(
            ticker=ticker,
            true_probability=true_prob,
            market_probability=market_prob,
            edge=edge,
            confidence=confidence,
            reasoning=reasoning,
            quality_score=quality_score,
            model_votes=model_votes,
        )

        logger.info(
            "  %s [%s]: TRUE=%.2f MARKET=%.2f EDGE=%+.3f CONF=%s QUALITY=%.0f [%s]",
            ticker, category, true_prob, market_prob, edge,
            confidence, quality_score, quality_breakdown,
        )
        return result
