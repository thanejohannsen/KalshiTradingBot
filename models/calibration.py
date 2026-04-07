"""
Calibration Model — loads a pre-trained LightGBM model and predicts
the true probability of YES for a given market.

Used by:
  - EnsembleProbabilityModel as the highest-signal sub-model
  - PostmortemAgent to evaluate whether a trade's edge was real
  - ScanAgent to prioritize markets with largest calibration gaps

The model was trained on 10,000+ resolved Kalshi markets (see
train_calibration_model.py) and learns systematic biases like:
  - Sports favorites at 70c actually resolve YES 78% of the time
  - Longshot YES contracts below 15c resolve at half the implied rate
  - Entertainment markets are 4-5pp less efficient than financial markets

The gap between model prediction and market price = estimated real edge.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("kalshi_bot.calibration")

# Where the model files live (alongside this module)
_MODEL_DIR = Path(__file__).parent
_MODEL_PATH = _MODEL_DIR / "calibration_model.pkl"
_META_PATH = _MODEL_DIR / "calibration_meta.json"

# Category code mapping — must match training. Loaded from meta.
_category_mapping: dict[str, int] = {}
_feature_cols: list[str] = []
_calibration_by_price: dict[str, dict] = {}
_calibration_by_category: dict[str, dict] = {}
_model = None
_loaded = False


def _load_model() -> bool:
    """Lazy-load the model and metadata. Returns True if available."""
    global _model, _category_mapping, _feature_cols, _loaded
    global _calibration_by_price, _calibration_by_category

    if _loaded:
        return _model is not None

    _loaded = True

    if not _MODEL_PATH.exists():
        logger.warning("Calibration model not found at %s — running without ML calibration", _MODEL_PATH)
        return False

    try:
        import joblib
        _model = joblib.load(_MODEL_PATH)
        logger.info("Calibration model loaded from %s", _MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load calibration model: %s", exc)
        return False

    if _META_PATH.exists():
        try:
            with open(_META_PATH) as f:
                meta = json.load(f)
            _category_mapping = meta.get("category_mapping", {})
            _feature_cols = meta.get("feature_cols", [])
            _calibration_by_price = meta.get("calibration_by_price", {})
            _calibration_by_category = meta.get("calibration_by_category", {})
            logger.info("Calibration metadata loaded: %d features, %d categories",
                        len(_feature_cols), len(_category_mapping))
        except Exception as exc:
            logger.warning("Failed to load calibration metadata: %s", exc)

    return True


def is_available() -> bool:
    """Check if the calibration model is loaded and ready."""
    return _load_model()


def predict_true_probability(
    market_price: float,
    category: str = "DEFAULT",
    spread: float = 0.0,
    volume_24h: float = 0.0,
    open_interest: float = 0.0,
    volume: float = 0.0,
    yes_bid: float = 0.0,
    yes_ask: float = 0.0,
    no_bid: float = 0.0,
    no_ask: float = 0.0,
    hours_to_close: float = 168.0,
    trade_size: float = 1.0,
    taker_is_yes: int = 1,
) -> float | None:
    """
    Predict the true probability of YES for a market.

    Returns the calibrated probability, or None if the model isn't available.
    """
    if not _load_model():
        return None

    features = _build_features(
        market_price=market_price,
        category=category,
        spread=spread,
        volume_24h=volume_24h,
        open_interest=open_interest,
        volume=volume,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        hours_to_close=hours_to_close,
        trade_size=trade_size,
        taker_is_yes=taker_is_yes,
    )

    try:
        pred = _model.predict([features])[0]
        return float(np.clip(pred, 0.01, 0.99))
    except Exception as exc:
        logger.debug("Calibration prediction failed: %s", exc)
        return None


def get_calibration_edge(market_price: float, category: str = "DEFAULT", **kwargs) -> float | None:
    """
    Return the estimated edge (model_pred - market_price).

    Positive = market underprices YES (we should buy YES).
    Negative = market overprices YES (we should buy NO or pass).
    """
    pred = predict_true_probability(market_price=market_price, category=category, **kwargs)
    if pred is None:
        return None
    return pred - market_price


def get_price_bucket_stats(market_price: float) -> dict | None:
    """Return historical calibration stats for the price bucket containing market_price."""
    _load_model()
    if not _calibration_by_price:
        return None

    # Find the matching bucket
    for bucket_label, stats in _calibration_by_price.items():
        # Parse "10%-20%" format
        parts = bucket_label.replace("%", "").split("-")
        if len(parts) == 2:
            try:
                lo = float(parts[0]) / 100
                hi = float(parts[1]) / 100
                if lo <= market_price < hi:
                    return stats
            except ValueError:
                continue
    return None


def get_category_stats(category: str) -> dict | None:
    """Return historical calibration stats for a category."""
    _load_model()
    return _calibration_by_category.get(category)


def _build_features(
    market_price: float,
    category: str,
    spread: float,
    volume_24h: float,
    open_interest: float,
    volume: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    hours_to_close: float,
    trade_size: float,
    taker_is_yes: int,
) -> list[float]:
    """Build feature vector matching the order used in training."""
    mp = np.clip(market_price, 0.01, 0.99)

    # Must match the exact feature names from training
    feature_dict = {
        "trade_price": mp,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "spread": spread if spread > 0 else max(0, yes_ask - yes_bid),
        "price_squared": mp ** 2,
        "price_deviation_from_50": abs(mp - 0.50),
        "is_longshot": int(mp < 0.20),
        "is_favorite": int(mp > 0.80),
        "is_mid_range": int(0.35 <= mp <= 0.65),
        "log_volume": np.log1p(volume),
        "log_volume_24h": np.log1p(volume_24h),
        "log_open_interest": np.log1p(open_interest),
        "trade_size": trade_size,
        "log_trade_size": np.log1p(trade_size),
        "taker_is_yes": taker_is_yes,
        "category_code": _category_mapping.get(category, _category_mapping.get("DEFAULT", 0)),
        "hours_to_close": np.clip(hours_to_close, 0, 8760),
        "log_hours_to_close": np.log1p(np.clip(hours_to_close, 0, 8760)),
    }

    # Return in the order specified by training metadata
    if _feature_cols:
        return [feature_dict.get(col, 0.0) for col in _feature_cols]
    else:
        return list(feature_dict.values())


# ---------------------------------------------------------------------------
# Ensemble sub-model interface
# ---------------------------------------------------------------------------

class CalibrationModel:
    """
    Ensemble sub-model that uses the pre-trained LightGBM calibration model.

    This is the highest-signal model in the ensemble — it's trained on
    thousands of resolved markets and knows exactly where Kalshi misprices.

    When the model isn't available (first run, before training), it
    falls back to returning market price with zero confidence.
    """

    name = "CalibrationModel"

    def estimate(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mp = inputs["market_probability"]

        if not is_available():
            return {"model": self.name, "probability": mp, "confidence": 0.0}

        # Key feature availability check — the model was trained with real
        # volume and bid/ask data.  When these are missing (all zeros), the
        # model produces unreliable predictions, so fall back to market price.
        has_volume = (inputs.get("volume_24h", 0) or 0) > 0
        has_bids = (inputs.get("yes_bid", 0) or 0) > 0
        if not has_volume and not has_bids:
            return {"model": self.name, "probability": mp, "confidence": 0.0}

        pred = predict_true_probability(
            market_price=mp,
            category=inputs.get("category", "DEFAULT"),
            spread=inputs.get("spread", 0.0),
            volume_24h=inputs.get("volume_24h", 0.0),
            open_interest=inputs.get("open_interest", 0.0),
            volume=inputs.get("volume", 0.0),
            yes_bid=inputs.get("yes_bid", 0.0),
            yes_ask=inputs.get("yes_ask", 0.0),
            no_bid=inputs.get("no_bid", 0.0),
            no_ask=inputs.get("no_ask", 0.0),
            hours_to_close=inputs.get("hours_to_close", 168.0),
        )

        if pred is None:
            return {"model": self.name, "probability": mp, "confidence": 0.0}

        edge = abs(pred - mp)
        # Confidence scales with the size of the calibration gap
        # Small gap (< 2pp) = low confidence, large gap (> 8pp) = high confidence
        confidence = min(0.90, edge * 8.0)

        return {
            "model": self.name,
            "probability": float(np.clip(pred, 0.01, 0.99)),
            "confidence": round(confidence, 3),
        }
