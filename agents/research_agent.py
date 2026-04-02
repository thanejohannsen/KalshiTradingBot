"""
RESEARCH AGENT — gathers external data and performs sentiment analysis
for each candidate market.  Data sources run in parallel.
"""

from __future__ import annotations

import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from config import Config
from database import Database
from data_sources.news_client import NewsClient
from data_sources.reddit_client import RedditClient
from data_sources.scraper import scrape_headlines
from data_sources.sportsbook_client import SportsbookClient
from data_sources.twitter_scraper import search_tweets
from models.category_profiles import CategoryProfileLoader
from models.claude_client import ClaudeClient
from models.sentiment import SentimentPipeline
from utils import CandidateMarket, DataPoint, ResearchResult

logger = logging.getLogger("kalshi_bot.research_agent")


class ResearchAgent:
    def __init__(
        self,
        cfg: Config,
        sentiment: SentimentPipeline,
        reddit: RedditClient | None = None,
        news: NewsClient | None = None,
        claude: ClaudeClient | None = None,
        shutdown_check: Callable[[], bool] | None = None,
        db: Database | None = None,
        sportsbook: SportsbookClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._sentiment = sentiment
        self._reddit = reddit
        self._news = news
        self._claude = claude
        self._shutdown_check = shutdown_check or (lambda: False)
        self._db = db
        self._sportsbook = sportsbook
        self._profile_loader = CategoryProfileLoader(db) if db else None
        self._cache_ttl_minutes = 30

    def research(self, candidate: CandidateMarket) -> ResearchResult:
        """
        For a single candidate market:
          1. Extract search keywords from title
          2. Fetch data from all sources in parallel
          3. Run sentiment pipeline
          4. Optionally call Claude for narrative summarization
          5. Return aggregated ResearchResult
        """
        ticker = candidate.market.ticker
        title = candidate.market.title
        query = self._extract_query(title)

        # Check cache first
        cached = self._get_from_cache(ticker)
        if cached is not None:
            logger.debug("Cache HIT for %s (age < %d min)", ticker, self._cache_ttl_minutes)
            return cached

        logger.debug("Researching: %s — query='%s'", ticker, query)

        # Gather data from all sources in parallel
        data_points = self._gather_data(query)

        if not data_points:
            logger.warning("No data points found for %s", ticker)
            return ResearchResult(
                ticker=ticker,
                sentiment_score=0.0,
                narrative_summary="Insufficient data",
                narrative_confidence="low",
                data_point_count=0,
            )

        # Run sentiment analysis
        sentiment_result = self._sentiment.analyze(data_points)

        # Generate narrative summary
        narrative = self._generate_narrative(title, data_points, sentiment_result)

        # Determine confidence
        confidence = self._assess_confidence(data_points, sentiment_result)

        # For SPORTS markets, fetch sportsbook consensus implied probability
        sportsbook_prob: float | None = None
        if self._sportsbook is not None and self._profile_loader is not None:
            category = self._profile_loader.category_from_market(
                candidate.market.category, ticker
            )
            if category == "SPORTS":
                try:
                    sportsbook_prob = self._sportsbook.get_implied_prob(title, ticker=ticker)
                    if sportsbook_prob is not None:
                        logger.info(
                            "Sportsbook line: %s → %.1f%% implied prob",
                            ticker, sportsbook_prob * 100,
                        )
                except Exception as exc:
                    logger.debug("Sportsbook lookup failed for %s: %s", ticker, exc)

        result = ResearchResult(
            ticker=ticker,
            sentiment_score=sentiment_result["sentiment_score"],
            narrative_summary=narrative,
            narrative_confidence=confidence,
            data_point_count=sentiment_result["sample_size"],
            bullish_pct=sentiment_result["bullish_pct"],
            bearish_pct=sentiment_result["bearish_pct"],
            data_points=data_points,
            sportsbook_prob=sportsbook_prob,
        )

        # Cache the result (don't store raw data_points)
        self._save_to_cache(ticker, query, result)

        logger.debug(
            "Research complete: %s | sentiment=%.3f | confidence=%s | sources=%d",
            ticker,
            result.sentiment_score,
            result.narrative_confidence,
            result.data_point_count,
        )
        return result

    def research_batch(self, candidates: list[CandidateMarket]) -> list[ResearchResult]:
        """Research multiple candidates in parallel (thread-per-candidate)."""
        if not candidates:
            return []

        ui_logger = logging.getLogger("kalshi_bot")
        use_inline = sys.stdout.isatty()
        last_logged_percent = -1

        def progress(done: int, total: int, *, force_log: bool = False) -> None:
            nonlocal last_logged_percent
            width = 20
            filled = int((done / total) * width) if total > 0 else 0
            bar = "#" * filled + "-" * (width - filled)
            pct = int((done / total) * 100) if total > 0 else 100

            if use_inline:
                sys.stdout.write(f"\rResearch progress [{bar}] {done}/{total} ({pct}%)")
                sys.stdout.flush()
                if done >= total:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                return

            # Non-interactive output (logs/files): emit coarse checkpoints only.
            if force_log or pct >= last_logged_percent + 5 or done == total:
                last_logged_percent = pct
                ui_logger.info("Research progress [%s] %d/%d (%d%%)", bar, done, total, pct)

        results: list[ResearchResult] = []
        total = len(candidates)
        completed = 0
        progress(completed, total, force_log=not use_inline)

        with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as executor:
            futures = {
                executor.submit(self.research, c): c for c in candidates
            }
            try:
                for future in as_completed(futures):
                    if self._shutdown_check():
                        # Cancel remaining futures and bail out
                        for f in futures:
                            f.cancel()
                        logger.info("Research interrupted — shutdown requested")
                        break
                    cand = futures[future]
                    try:
                        results.append(future.result(timeout=60))
                    except Exception as exc:
                        logger.error("Research failed for %s: %s", cand.market.ticker, exc)
                        results.append(
                            ResearchResult(
                                ticker=cand.market.ticker,
                                sentiment_score=0.0,
                                narrative_summary=f"Research error: {exc}",
                                narrative_confidence="low",
                                data_point_count=0,
                            )
                        )
                    finally:
                        completed += 1
                        progress(completed, total)
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                logger.info("Research interrupted by Ctrl+C")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        if use_inline and completed < total:
            # Ensure we end the current terminal line on interrupts/early exits.
            sys.stdout.write("\n")
            sys.stdout.flush()
        return results

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_from_cache(self, ticker: str) -> ResearchResult | None:
        if self._db is None:
            return None
        try:
            row = self._db.get_cached_research(ticker, self._cache_ttl_minutes)
            if row is None:
                return None
            return ResearchResult(
                ticker=row["ticker"],
                sentiment_score=row["sentiment_score"],
                narrative_summary=row["narrative_summary"],
                narrative_confidence=row["narrative_confidence"],
                data_point_count=row["data_point_count"],
                bullish_pct=row["bullish_pct"],
                bearish_pct=row["bearish_pct"],
            )
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", ticker, exc)
            return None

    def _save_to_cache(self, ticker: str, query: str, result: ResearchResult) -> None:
        if self._db is None:
            return
        try:
            self._db.cache_research(ticker, query, {
                "sentiment_score": result.sentiment_score,
                "narrative_summary": result.narrative_summary,
                "narrative_confidence": result.narrative_confidence,
                "data_point_count": result.data_point_count,
                "bullish_pct": result.bullish_pct,
                "bearish_pct": result.bearish_pct,
            })
        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------
    def _gather_data(self, query: str) -> list[DataPoint]:
        """Fetch from all available sources in parallel."""
        all_points: list[DataPoint] = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            # Twitter
            futures[executor.submit(search_tweets, query, 30)] = "twitter"

            # Reddit
            if self._reddit:
                futures[executor.submit(self._reddit.search, query)] = "reddit"

            # News
            if self._news:
                futures[executor.submit(self._news.search, query)] = "news"

            # Scraper fallback
            futures[executor.submit(scrape_headlines, query, 10)] = "scraper"

            for future in as_completed(futures):
                source = futures[future]
                try:
                    points = future.result()
                    all_points.extend(points)
                    logger.debug("  %s returned %d data points", source, len(points))
                except Exception as exc:
                    logger.warning("  %s failed: %s", source, exc)

        return all_points

    # ------------------------------------------------------------------
    # Query extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_query(title: str) -> str:
        """
        Turn a market title like
        "Will Bitcoin exceed $100,000 by end of March 2026?"
        into a search-friendly query: "Bitcoin exceed $100,000 March 2026"
        """
        # Remove common question words and punctuation
        cleaned = re.sub(r"\b(will|the|a|an|be|is|are|was|were|by|of|in|on|at|to|for)\b", "", title, flags=re.IGNORECASE)
        cleaned = re.sub(r"[?!.,;:\"']", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Take first ~8 meaningful words
        words = cleaned.split()[:8]
        return " ".join(words)

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------
    def _generate_narrative(
        self,
        title: str,
        data_points: list[DataPoint],
        sentiment: dict[str, Any],
    ) -> str:
        """Build a narrative summary — uses Claude if available, else heuristic."""
        # Try Claude first
        if self._claude and self._claude.available:
            summary = self._claude.summarize_narrative(title, data_points)
            if summary:
                return summary

        # Heuristic fallback: summarize from top-engagement sources
        score = sentiment["sentiment_score"]
        direction = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
        top_sources = sorted(data_points, key=lambda dp: dp.engagement, reverse=True)[:3]
        snippets = "; ".join(dp.text[:80] for dp in top_sources)

        return (
            f"Overall sentiment is {direction} ({score:+.3f}). "
            f"Key sources: {snippets}"
        )

    # ------------------------------------------------------------------
    # Confidence assessment
    # ------------------------------------------------------------------
    @staticmethod
    def _assess_confidence(
        data_points: list[DataPoint],
        sentiment: dict[str, Any],
    ) -> str:
        """
        Heuristic confidence: high if many agreeing sources, low if few or conflicting.
        """
        n = len(data_points)
        bullish = sentiment["bullish_pct"]
        bearish = sentiment["bearish_pct"]

        # Need at least 10 data points for medium, 25 for high
        if n < 5:
            return "low"
        if n < 10:
            # Even with few points, if agreement is strong…
            if max(bullish, bearish) > 0.7:
                return "medium"
            return "low"

        # High agreement + enough data = high confidence
        agreement = max(bullish, bearish)
        if agreement > 0.65 and n >= 25:
            return "high"
        if agreement > 0.55 and n >= 15:
            return "medium"

        return "low"
