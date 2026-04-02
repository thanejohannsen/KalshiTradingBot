"""
Claude / Anthropic LLM client — optional module for:
  - Narrative summarization
  - Misinformation detection
  - Postmortem insight generation

Disabled if ANTHROPIC_API_KEY is not set in config.
"""

from __future__ import annotations

import logging
from typing import Any

from utils import DataPoint

logger = logging.getLogger("kalshi_bot.claude")


class ClaudeClient:
    """Wrapper around the Anthropic Python SDK."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._api_key = api_key
        self._model = model
        self._client = None
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                logger.info("Claude client initialized (model=%s)", model)
            except ImportError:
                logger.warning("anthropic package not installed — Claude features disabled")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Narrative summarization
    # ------------------------------------------------------------------
    def summarize_narrative(
        self,
        market_title: str,
        data_points: list[DataPoint],
        max_tokens: int = 300,
    ) -> str:
        """
        Given a market and collected data points, produce a concise
        narrative summary of what's driving belief / price movement.
        """
        if not self.available:
            return ""

        texts = "\n".join(
            f"[{dp.source}] {dp.text[:200]}" for dp in data_points[:30]
        )
        prompt = (
            f"You are analyzing a prediction market: \"{market_title}\".\n\n"
            f"Below are recent social media posts and news snippets about this topic:\n\n"
            f"{texts}\n\n"
            f"In 2-3 sentences, summarize the dominant narrative driving public opinion "
            f"on this market. Note any conflicting viewpoints or potential misinformation."
        )

        return self._call(prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Misinformation detection
    # ------------------------------------------------------------------
    def detect_misinformation(
        self,
        market_title: str,
        data_points: list[DataPoint],
        max_tokens: int = 200,
    ) -> str:
        """Flag any data points that may contain misinformation or lagging narratives."""
        if not self.available:
            return ""

        texts = "\n".join(
            f"[{dp.source}] {dp.text[:200]}" for dp in data_points[:20]
        )
        prompt = (
            f"Market: \"{market_title}\"\n\n"
            f"Data points:\n{texts}\n\n"
            f"Identify any claims that appear to be misinformation, outdated, or "
            f"based on a lagging narrative that no longer reflects the current situation. "
            f"Be concise. If none are found, say 'No misinformation detected'."
        )

        return self._call(prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Postmortem analysis
    # ------------------------------------------------------------------
    def analyze_loss(
        self,
        trade_info: dict[str, Any],
        max_tokens: int = 400,
    ) -> str:
        """Generate a human-readable postmortem insight for a losing trade."""
        if not self.available:
            return ""

        prompt = (
            f"A prediction market trade resulted in a loss. Analyze what went wrong.\n\n"
            f"Trade details:\n"
            f"- Market: {trade_info.get('ticker', 'unknown')}\n"
            f"- Side: {trade_info.get('side', 'unknown')}\n"
            f"- Entry price: {trade_info.get('entry_price', 'unknown')}\n"
            f"- Predicted probability: {trade_info.get('predicted_prob', 'unknown')}\n"
            f"- Market probability at entry: {trade_info.get('market_prob', 'unknown')}\n"
            f"- Edge estimated: {trade_info.get('edge', 'unknown')}\n"
            f"- Sentiment score: {trade_info.get('sentiment_score', 'unknown')}\n"
            f"- Thesis: {trade_info.get('thesis', 'none')}\n"
            f"- Result: LOSS\n\n"
            f"In 2-3 sentences, explain the most likely reason for this loss and "
            f"what pattern should be avoided in the future."
        )

        return self._call(prompt, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _call(self, prompt: str, max_tokens: int = 300) -> str:
        try:
            response = self._client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            logger.error("Claude API call failed: %s", exc)
            return ""
