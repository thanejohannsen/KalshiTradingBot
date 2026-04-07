"""
Train a LightGBM calibration model on historical Kalshi trade-level data.

Usage:
    python models/train_calibration_model.py --data-dir /path/to/prediction-market-analysis/data

The model learns: "When a trade happens at price P in category C with spread S
and volume V, what is the actual probability the market resolves YES?"

The gap between the model's predicted probability and the trade price IS the
edge that should be exploited.

Output:
    models/calibration_model.pkl    — serialized LightGBM model
    models/calibration_meta.json    — feature names, training stats, calibration curves
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_calibration")

# Category mapping — maps Kalshi event_ticker prefixes to canonical categories
_EVENT_PREFIX_TO_CATEGORY: dict[str, str] = {
    "INX": "SPORTS", "MLB": "SPORTS", "NBA": "SPORTS", "NFL": "SPORTS",
    "NHL": "SPORTS", "MLS": "SPORTS", "WNBA": "SPORTS", "NCAAB": "SPORTS",
    "NCAAF": "SPORTS", "MMA": "SPORTS", "UFC": "SPORTS", "PGA": "SPORTS",
    "ATP": "SPORTS", "WTA": "SPORTS", "SOC": "SPORTS", "F1": "SPORTS",
    "EPL": "SPORTS", "LIGA": "SPORTS",
    "BTC": "CRYPTO", "ETH": "CRYPTO", "SOL": "CRYPTO", "XRP": "CRYPTO",
    "DOGE": "CRYPTO",
    "SP5": "FINANCIALS", "NASDAQ": "FINANCIALS", "DJI": "FINANCIALS",
    "INXD": "FINANCIALS", "INXU": "FINANCIALS", "GOLD": "FINANCIALS",
    "OIL": "FINANCIALS", "TSLA": "FINANCIALS", "AAPL": "FINANCIALS",
    "HIGHTEMP": "WEATHER", "LOWTEMP": "WEATHER", "RAIN": "WEATHER",
    "SNOW": "WEATHER", "HURR": "WEATHER", "QUAKE": "WEATHER",
    "FED": "ECONOMICS", "CPI": "ECONOMICS", "GDP": "ECONOMICS",
    "UNEMP": "ECONOMICS", "JOBS": "ECONOMICS", "FOMC": "ECONOMICS",
    "PRES": "POLITICS", "SENATE": "POLITICS", "HOUSE": "POLITICS",
    "GOV": "POLITICS", "SCOTUS": "POLITICS", "ELECT": "POLITICS",
    "OSCAR": "ENTERTAINMENT", "EMMY": "ENTERTAINMENT",
    "GRAMMY": "ENTERTAINMENT", "MOVIE": "ENTERTAINMENT",
}


def infer_category(event_ticker: str) -> str:
    if not event_ticker:
        return "DEFAULT"
    upper = event_ticker.upper()
    for length in range(min(10, len(upper)), 2, -1):
        prefix = upper[:length]
        if prefix in _EVENT_PREFIX_TO_CATEGORY:
            return _EVENT_PREFIX_TO_CATEGORY[prefix]
    parts = upper.split("-")
    for p in parts:
        if p in _EVENT_PREFIX_TO_CATEGORY:
            return _EVENT_PREFIX_TO_CATEGORY[p]
    return "DEFAULT"


def load_and_prepare_data(data_dir: Path, sample_frac: float = 0.10) -> pd.DataFrame:
    """
    Load trades joined to resolved markets.

    Uses trade-level yes_price as the feature (the actual price at time of trade),
    NOT the market's last_price (which is the post-resolution snapshot).

    Samples to keep memory manageable (72M trades → ~7M sample).
    """
    markets_dir = data_dir / "kalshi" / "markets"
    trades_dir = data_dir / "kalshi" / "trades"

    logger.info("Loading market metadata...")
    markets = pd.read_parquet(
        markets_dir,
        columns=["ticker", "event_ticker", "status", "result",
                 "yes_bid", "yes_ask", "no_bid", "no_ask",
                 "volume", "volume_24h", "open_interest",
                 "created_time", "close_time"],
    )
    resolved = markets[
        (markets["status"] == "finalized") &
        (markets["result"].isin(["yes", "no"]))
    ].copy()
    resolved["outcome"] = (resolved["result"] == "yes").astype(np.int8)
    logger.info("Resolved markets: %d", len(resolved))

    # Build per-market metadata
    market_meta = resolved[["ticker", "event_ticker", "outcome",
                            "yes_bid", "yes_ask", "no_bid", "no_ask",
                            "volume", "volume_24h", "open_interest",
                            "created_time", "close_time"]].copy()
    del markets, resolved

    logger.info("Loading trades (sampling %.0f%%)...", sample_frac * 100)
    # Read in chunks to manage memory
    trades_chunks = []
    for parquet_file in sorted(trades_dir.glob("*.parquet")):
        chunk = pd.read_parquet(parquet_file)
        # Sample early to keep memory down
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=42)
        trades_chunks.append(chunk)

    trades = pd.concat(trades_chunks, ignore_index=True)
    del trades_chunks
    logger.info("Sampled trades: %d", len(trades))

    # Join trades to market metadata
    df = trades.merge(market_meta, on="ticker", how="inner")
    del trades, market_meta
    logger.info("Joined dataset: %d rows", len(df))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for the calibration model from trade-level data."""
    out = pd.DataFrame()

    # Trade price (in cents 1-99) → probability (0.01-0.99)
    out["trade_price"] = (df["yes_price"].clip(1, 99) / 100.0)

    # Market-level bid/ask at snapshot (also cents)
    out["yes_bid"] = (df["yes_bid"].fillna(0) / 100.0).clip(0, 1)
    out["yes_ask"] = (df["yes_ask"].fillna(0) / 100.0).clip(0, 1)
    out["no_bid"] = (df["no_bid"].fillna(0) / 100.0).clip(0, 1)
    out["no_ask"] = (df["no_ask"].fillna(0) / 100.0).clip(0, 1)
    out["spread"] = (out["yes_ask"] - out["yes_bid"]).clip(0, 1)

    # Price-derived features
    out["price_squared"] = out["trade_price"] ** 2
    out["price_deviation_from_50"] = abs(out["trade_price"] - 0.50)
    out["is_longshot"] = (out["trade_price"] < 0.20).astype(np.int8)
    out["is_favorite"] = (out["trade_price"] > 0.80).astype(np.int8)
    out["is_mid_range"] = ((out["trade_price"] >= 0.35) & (out["trade_price"] <= 0.65)).astype(np.int8)

    # Volume and liquidity
    out["log_volume"] = np.log1p(df["volume"].fillna(0))
    out["log_volume_24h"] = np.log1p(df["volume_24h"].fillna(0))
    out["log_open_interest"] = np.log1p(df["open_interest"].fillna(0))

    # Trade microstructure
    out["trade_size"] = df["count"].fillna(1).clip(1, 10000)
    out["log_trade_size"] = np.log1p(out["trade_size"])
    out["taker_is_yes"] = (df["taker_side"] == "yes").astype(np.int8)

    # Category
    out["category"] = df["event_ticker"].apply(infer_category)
    cat_codes = out["category"].astype("category")
    out["category_code"] = cat_codes.cat.codes
    cat_mapping = dict(zip(cat_codes.cat.categories, range(len(cat_codes.cat.categories))))

    # Time features
    try:
        close_times = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
        trade_times = pd.to_datetime(df["created_time_x"] if "created_time_x" in df.columns
                                     else df["created_time"], utc=True, errors="coerce")
        hours_to_close = (close_times - trade_times).dt.total_seconds() / 3600
        out["hours_to_close"] = hours_to_close.fillna(168).clip(0, 8760)
        out["log_hours_to_close"] = np.log1p(out["hours_to_close"])
    except Exception:
        out["hours_to_close"] = 168.0
        out["log_hours_to_close"] = np.log1p(168.0)

    # Target
    out["outcome"] = df["outcome"].values

    # Store metadata
    out.attrs["category_mapping"] = cat_mapping

    return out


def train_model(features: pd.DataFrame) -> dict:
    """Train LightGBM calibration model on trade-level data."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, brier_score_loss

    feature_cols = [c for c in features.columns if c not in ("outcome", "category")]
    X = features[feature_cols]
    y = features["outcome"]

    logger.info("Feature matrix: %d rows x %d features", X.shape[0], X.shape[1])
    logger.info("Features: %s", feature_cols)
    logger.info("Outcome distribution: YES=%.1f%%, NO=%.1f%%",
                y.mean() * 100, (1 - y.mean()) * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50),
        ],
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    test_logloss = log_loss(y_test, y_pred_test)
    test_brier = brier_score_loss(y_test, y_pred_test)

    # Baseline: use trade price as probability
    baseline_pred = X_test["trade_price"].values
    baseline_logloss = log_loss(y_test, baseline_pred.clip(0.01, 0.99))
    baseline_brier = brier_score_loss(y_test, baseline_pred.clip(0.01, 0.99))

    logger.info("=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info("Test LogLoss:  %.6f  (baseline: %.6f, %.2f%% better)",
                test_logloss, baseline_logloss,
                (1 - test_logloss / baseline_logloss) * 100)
    logger.info("Test Brier:    %.6f  (baseline: %.6f, %.2f%% better)",
                test_brier, baseline_brier,
                (1 - test_brier / baseline_brier) * 100)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Feature importance (gain):")
    for feat, imp in sorted_imp[:15]:
        logger.info("  %-30s  %.0f", feat, imp)

    # Calibration by price bucket
    logger.info("\nCalibration by price bucket (test set):")
    test_df = pd.DataFrame({
        "trade_price": X_test["trade_price"].values,
        "model_pred": y_pred_test,
        "actual": y_test.values,
    })

    buckets = [(0.01, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
               (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
               (0.80, 0.90), (0.90, 1.00)]

    calibration_data = {}
    for lo, hi in buckets:
        mask = (test_df["trade_price"] >= lo) & (test_df["trade_price"] < hi)
        subset = test_df[mask]
        if len(subset) < 100:
            continue
        actual_rate = float(subset["actual"].mean())
        avg_trade_px = float(subset["trade_price"].mean())
        avg_model = float(subset["model_pred"].mean())
        market_error = abs(actual_rate - avg_trade_px)
        model_error = abs(actual_rate - avg_model)
        n = len(subset)

        bucket_label = f"{lo:.0%}-{hi:.0%}"
        calibration_data[bucket_label] = {
            "actual_rate": round(actual_rate, 4),
            "market_avg": round(avg_trade_px, 4),
            "model_avg": round(avg_model, 4),
            "market_error": round(market_error, 4),
            "model_error": round(model_error, 4),
            "edge": round(avg_model - avg_trade_px, 4),
            "count": n,
        }
        better = "MODEL" if model_error < market_error else "MARKET"
        logger.info(
            "  %8s | actual=%.3f trade_px=%.3f model=%.3f | "
            "mkt_err=%.3f mdl_err=%.3f | edge=%+.3f | n=%d | %s",
            bucket_label, actual_rate, avg_trade_px, avg_model,
            market_error, model_error, avg_model - avg_trade_px, n, better,
        )

    # Per-category calibration
    logger.info("\nCalibration by category (test set):")
    cat_data = features.loc[X_test.index, "category"]
    test_df["category"] = cat_data.values

    category_calibration = {}
    for cat in sorted(test_df["category"].unique()):
        mask = test_df["category"] == cat
        subset = test_df[mask]
        if len(subset) < 50:
            continue
        actual_rate = float(subset["actual"].mean())
        avg_trade_px = float(subset["trade_price"].mean())
        avg_model = float(subset["model_pred"].mean())
        edge = avg_model - avg_trade_px
        n = len(subset)
        category_calibration[cat] = {
            "actual_rate": round(actual_rate, 4),
            "avg_market_price": round(avg_trade_px, 4),
            "avg_model_pred": round(avg_model, 4),
            "systematic_edge": round(edge, 4),
            "count": n,
        }
        logger.info(
            "  %-15s | actual=%.3f trade_px=%.3f model=%.3f | edge=%+.3f | n=%d",
            cat, actual_rate, avg_trade_px, avg_model, edge, n,
        )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "category_mapping": features.attrs.get("category_mapping", {}),
        "metrics": {
            "test_logloss": round(test_logloss, 6),
            "test_brier": round(test_brier, 6),
            "baseline_logloss": round(baseline_logloss, 6),
            "baseline_brier": round(baseline_brier, 6),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        "feature_importance": dict(sorted_imp),
        "calibration_by_price": calibration_data,
        "calibration_by_category": category_calibration,
    }


def save_model(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "calibration_model.pkl"
    meta_path = output_dir / "calibration_meta.json"

    joblib.dump(results["model"], model_path)
    logger.info("Model saved to %s", model_path)

    meta = {k: v for k, v in results.items() if k != "model"}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)


def main():
    parser = argparse.ArgumentParser(description="Train Kalshi calibration model")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent.parent.parent / "prediction-market-analysis" / "data",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--sample-frac", type=float, default=0.10,
                        help="Fraction of trades to sample (default 10%%)")
    args = parser.parse_args()

    df = load_and_prepare_data(args.data_dir, sample_frac=args.sample_frac)
    features = engineer_features(df)
    del df

    features = features.dropna(subset=["outcome"])
    logger.info("Feature-engineered dataset: %d rows", len(features))

    if len(features) < 1000:
        logger.error("Not enough data (%d rows, need 1000+)", len(features))
        sys.exit(1)

    results = train_model(features)
    save_model(results, args.output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
