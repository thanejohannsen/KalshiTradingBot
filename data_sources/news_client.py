"""
News data source — combines NewsAPI (free tier) and Google News RSS.
Falls back to RSS-only if NewsAPI key is not configured.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests

from utils import DataPoint

logger = logging.getLogger("kalshi_bot.news")

# Google News RSS search endpoint (no API key required)
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


class NewsClient:
    """Aggregates news from NewsAPI + Google News RSS."""

    def __init__(self, newsapi_key: str = "") -> None:
        self._newsapi_key = newsapi_key
        self._newsapi_client = None
        if newsapi_key:
            try:
                from newsapi import NewsApiClient
                self._newsapi_client = NewsApiClient(api_key=newsapi_key)
                logger.info("NewsAPI client initialized")
            except ImportError:
                logger.warning("newsapi-python not installed, using RSS only")

    def search(
        self,
        query: str,
        max_results: int = 20,
        days_back: int = 7,
    ) -> list[DataPoint]:
        """
        Search news articles matching *query* from all available sources.
        Combines NewsAPI + RSS, deduplicates by URL.
        """
        results: list[DataPoint] = []
        seen_urls: set[str] = set()

        # 1. NewsAPI (if available)
        if self._newsapi_client:
            for dp in self._search_newsapi(query, max_results, days_back):
                if dp.url not in seen_urls:
                    seen_urls.add(dp.url)
                    results.append(dp)

        # 2. Google News RSS (always available)
        for dp in self._search_google_rss(query, max_results):
            if dp.url not in seen_urls:
                seen_urls.add(dp.url)
                results.append(dp)

        logger.info("News: fetched %d articles for query '%s'", len(results), query)
        return results[:max_results]

    # ------------------------------------------------------------------
    # NewsAPI
    # ------------------------------------------------------------------
    def _search_newsapi(
        self, query: str, max_results: int, days_back: int
    ) -> list[DataPoint]:
        if not self._newsapi_client:
            return []
        try:
            from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            resp = self._newsapi_client.get_everything(
                q=query,
                from_param=from_date,
                sort_by="relevancy",
                page_size=min(max_results, 100),
                language="en",
            )
            articles = resp.get("articles", [])
            results: list[DataPoint] = []
            for art in articles:
                text = (art.get("title") or "") + " " + (art.get("description") or "")
                results.append(
                    DataPoint(
                        text=text.strip(),
                        source="news",
                        timestamp=art.get("publishedAt", ""),
                        engagement=0.0,
                        url=art.get("url", ""),
                    )
                )
            return results
        except Exception as exc:
            logger.warning("NewsAPI search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Google News RSS
    # ------------------------------------------------------------------
    def _search_google_rss(self, query: str, max_results: int) -> list[DataPoint]:
        try:
            url = GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
            feed = feedparser.parse(url)
            results: list[DataPoint] = []
            for entry in feed.entries[:max_results]:
                text = entry.get("title", "")
                summary = entry.get("summary", "")
                if summary:
                    text += " " + summary
                results.append(
                    DataPoint(
                        text=text.strip(),
                        source="news_rss",
                        timestamp=entry.get("published", ""),
                        engagement=0.0,
                        url=entry.get("link", ""),
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Google News RSS failed: %s", exc)
            return []
