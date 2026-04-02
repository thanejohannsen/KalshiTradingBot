"""
Sentiment analysis pipeline — FinBERT for accuracy, VADER for speed.
Both run locally with no API calls.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from utils import DataPoint

logger = logging.getLogger("kalshi_bot.sentiment")


@dataclass
class SentimentScore:
    """Result from a single text analysis."""
    positive: float
    negative: float
    neutral: float
    compound: float   # single score: -1 (very bearish) to +1 (very bullish)


# ── VADER (lightweight, no GPU) ─────────────────────────────────────


class VADERAnalyzer:
    """Fast rule-based sentiment — good for bulk screening."""

    def __init__(self) -> None:
        self._analyzer = None

    def _ensure_loaded(self) -> None:
        if self._analyzer is not None:
            return
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER analyzer loaded")
        except ImportError:
            logger.error("vaderSentiment not installed — run: pip install vaderSentiment")
            raise

    def score(self, text: str) -> SentimentScore:
        self._ensure_loaded()
        scores = self._analyzer.polarity_scores(text)  # type: ignore[union-attr]
        return SentimentScore(
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            compound=scores["compound"],
        )

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        return [self.score(t) for t in texts]


# ── FinBERT (transformer-based, financial domain) ───────────────────


class FinBERTAnalyzer:
    """
    Finance-tuned BERT model (ProsusAI/finbert).
    Downloads ~420 MB on first use.  Runs on CPU by default.
    """

    MODEL_NAME = "ProsusAI/finbert"
    _lock = threading.Lock()
    _shared_pipeline = None

    def __init__(self, device: str = "cpu", batch_size: int = 16) -> None:
        self._device = device
        self._batch_size = batch_size

    @property
    def _pipeline(self):
        return FinBERTAnalyzer._shared_pipeline

    def _ensure_loaded(self) -> None:
        if FinBERTAnalyzer._shared_pipeline is not None:
            return
        with FinBERTAnalyzer._lock:
            if FinBERTAnalyzer._shared_pipeline is not None:
                return
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                    pipeline as hf_pipeline,
                )
                tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_NAME,
                    dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
                model = model.to("cpu")
                FinBERTAnalyzer._shared_pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device="cpu",
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT model loaded on %s", self._device)
            except ImportError:
                logger.error("transformers/torch not installed — run: pip install transformers torch")
                raise

    def score(self, text: str) -> SentimentScore:
        self._ensure_loaded()
        result = self._pipeline(text[:512])[0]  # type: ignore[index]
        return self._map_label(result)

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        self._ensure_loaded()
        truncated = [t[:512] for t in texts]
        results = self._pipeline(truncated, batch_size=self._batch_size)  # type: ignore[misc]
        return [self._map_label(r) for r in results]

    @staticmethod
    def _map_label(result: dict[str, Any]) -> SentimentScore:
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return SentimentScore(positive=score, negative=0.0, neutral=1 - score, compound=score)
        elif label == "negative":
            return SentimentScore(positive=0.0, negative=score, neutral=1 - score, compound=-score)
        else:
            return SentimentScore(positive=0.0, negative=0.0, neutral=score, compound=0.0)


# ── Aggregation ─────────────────────────────────────────────────────


class SentimentPipeline:
    """
    Two-tier pipeline:
    1. VADER on ALL data points (fast)
    2. FinBERT on top-N most-engaged data points (accurate)
    3. Weighted merge → final score
    """

    def __init__(
        self,
        finbert_top_n: int = 20,
        finbert_weight: float = 0.6,
        vader_weight: float = 0.4,
        use_finbert: bool = True,
    ) -> None:
        self._vader = VADERAnalyzer()
        self._finbert = FinBERTAnalyzer() if use_finbert else None
        self._top_n = finbert_top_n
        self._fw = finbert_weight if use_finbert else 0.0
        self._vw = vader_weight if use_finbert else 1.0

    def analyze(self, data_points: list[DataPoint]) -> dict[str, Any]:
        """
        Run the full sentiment pipeline on a list of data points.

        Returns dict with:
            sentiment_score: float (-1 to +1)
            bullish_pct: float (0 to 1)
            bearish_pct: float (0 to 1)
            sample_size: int
        """
        if not data_points:
            return {
                "sentiment_score": 0.0,
                "bullish_pct": 0.0,
                "bearish_pct": 0.0,
                "sample_size": 0,
            }

        texts = [dp.text for dp in data_points]

        # 1. VADER on all texts
        vader_scores = self._vader.score_batch(texts)
        vader_compound = sum(s.compound for s in vader_scores) / len(vader_scores)

        # 2. FinBERT on top-N most engaged (optional)
        finbert_compound = 0.0
        if self._finbert is not None:
            sorted_dps = sorted(data_points, key=lambda dp: dp.engagement, reverse=True)
            top_texts = [dp.text for dp in sorted_dps[: self._top_n]]
            if top_texts:
                try:
                    fb_scores = self._finbert.score_batch(top_texts)
                    finbert_compound = sum(s.compound for s in fb_scores) / len(fb_scores)
                except Exception as exc:
                    logger.warning("FinBERT failed, using VADER only: %s", exc)
                    self._fw = 0.0
                    self._vw = 1.0

        # 3. Weighted merge
        total_weight = self._fw + self._vw
        sentiment_score = (
            (finbert_compound * self._fw + vader_compound * self._vw) / total_weight
            if total_weight > 0
            else 0.0
        )

        # Classification percentages
        bullish = sum(1 for s in vader_scores if s.compound > 0.05)
        bearish = sum(1 for s in vader_scores if s.compound < -0.05)
        total = len(vader_scores)

        return {
            "sentiment_score": round(sentiment_score, 4),
            "bullish_pct": round(bullish / total, 4) if total else 0.0,
            "bearish_pct": round(bearish / total, 4) if total else 0.0,
            "sample_size": total,
        }
