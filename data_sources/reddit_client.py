"""
Reddit data source — uses PRAW to search subreddits for market-related discussion.
Falls back gracefully if credentials are not configured.
"""

from __future__ import annotations

import logging
from typing import Any

from utils import DataPoint

logger = logging.getLogger("kalshi_bot.reddit")


class RedditClient:
    """Thin wrapper around PRAW for searching Reddit."""

    # Subreddits most likely to discuss prediction markets / current events
    DEFAULT_SUBREDDITS = [
        "kalshi",
        "predictit",
        "polymarket",
        "wallstreetbets",
        "news",
        "worldnews",
        "politics",
    ]

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "KalshiBot/1.0",
    ) -> None:
        self._reddit = None
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent

    def connect(self) -> bool:
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            # Validate credentials with a lightweight call
            _ = self._reddit.user.me()
            logger.info("Reddit client connected")
            return True
        except Exception as exc:
            logger.warning("Reddit connection failed (will run without Reddit): %s", exc)
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    user_agent=self._user_agent,
                )
                logger.info("Reddit client initialized in read-only mode")
                return True
            except Exception:
                return False

    def search(
        self,
        query: str,
        subreddits: list[str] | None = None,
        max_results_per_sub: int = 10,
        time_filter: str = "week",
    ) -> list[DataPoint]:
        """
        Search across subreddits for posts matching *query*.
        Returns combined, deduplicated results.
        """
        if self._reddit is None:
            return []

        subs = subreddits or self.DEFAULT_SUBREDDITS
        results: list[DataPoint] = []
        seen_ids: set[str] = set()

        for sub_name in subs:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                for post in subreddit.search(
                    query, sort="relevance", time_filter=time_filter, limit=max_results_per_sub
                ):
                    if post.id in seen_ids:
                        continue
                    seen_ids.add(post.id)

                    text = post.title
                    if post.selftext:
                        text += "\n" + post.selftext[:500]

                    results.append(
                        DataPoint(
                            text=text,
                            source="reddit",
                            timestamp=str(post.created_utc),
                            engagement=float(post.score),
                            url=f"https://reddit.com{post.permalink}",
                        )
                    )
            except Exception as exc:
                logger.warning("Reddit search failed for r/%s: %s", sub_name, exc)
                continue

        logger.info("Reddit: fetched %d posts for query '%s'", len(results), query)
        return results

    def get_comments(self, url: str, limit: int = 20) -> list[DataPoint]:
        """Fetch top comments from a Reddit post URL."""
        if self._reddit is None:
            return []

        try:
            submission = self._reddit.submission(url=url)
            submission.comments.replace_more(limit=0)
            results: list[DataPoint] = []
            for comment in submission.comments[:limit]:
                results.append(
                    DataPoint(
                        text=comment.body[:500],
                        source="reddit",
                        timestamp=str(comment.created_utc),
                        engagement=float(comment.score),
                        url=url,
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Reddit comments fetch failed: %s", exc)
            return []
