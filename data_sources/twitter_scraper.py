"""
Twitter data source — uses snscrape for keyword-based tweet collection.
Falls back gracefully if snscrape is unavailable or blocked.
"""

from __future__ import annotations

import logging
import subprocess
import json
from typing import Any

from utils import DataPoint

logger = logging.getLogger("kalshi_bot.twitter")

_SNSCRAPE_AVAILABLE: bool | None = None


def _check_snscrape() -> bool:
    global _SNSCRAPE_AVAILABLE
    if _SNSCRAPE_AVAILABLE is not None:
        return _SNSCRAPE_AVAILABLE
    try:
        import snscrape  # noqa: F401
        _SNSCRAPE_AVAILABLE = True
    except ImportError:
        logger.warning("snscrape not installed — Twitter data will be unavailable")
        _SNSCRAPE_AVAILABLE = False
    return _SNSCRAPE_AVAILABLE


def search_tweets(
    query: str,
    max_results: int = 50,
) -> list[DataPoint]:
    """
    Search Twitter for recent tweets matching *query*.

    Uses snscrape's Python API if available, falls back to CLI subprocess.
    Returns an empty list on any failure (graceful degradation).
    """
    if not _check_snscrape():
        return []

    try:
        return _search_via_python(query, max_results)
    except Exception as exc:
        logger.warning("snscrape Python API failed (%s), trying CLI", exc)
        try:
            return _search_via_cli(query, max_results)
        except Exception as exc2:
            logger.warning("snscrape CLI also failed: %s", exc2)
            return []


def _search_via_python(query: str, max_results: int) -> list[DataPoint]:
    """Use snscrape's Python module directly."""
    import snscrape.modules.twitter as sntwitter  # type: ignore[import-untyped]

    results: list[DataPoint] = []
    scraper = sntwitter.TwitterSearchScraper(query)
    for i, tweet in enumerate(scraper.get_items()):
        if i >= max_results:
            break
        results.append(
            DataPoint(
                text=tweet.rawContent if hasattr(tweet, "rawContent") else str(tweet),
                source="twitter",
                timestamp=tweet.date.isoformat() if hasattr(tweet, "date") else "",
                engagement=float(
                    getattr(tweet, "likeCount", 0) or 0
                ) + float(getattr(tweet, "retweetCount", 0) or 0),
                url=tweet.url if hasattr(tweet, "url") else "",
            )
        )
    logger.debug("Twitter: fetched %d tweets for query '%s'", len(results), query)
    return results


def _search_via_cli(query: str, max_results: int) -> list[DataPoint]:
    """Fallback: use snscrape CLI and parse JSON output."""
    cmd = [
        "snscrape",
        "--jsonl",
        "--max-results", str(max_results),
        "twitter-search", query,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"snscrape exited with {proc.returncode}: {proc.stderr[:200]}")

    results: list[DataPoint] = []
    for line in proc.stdout.strip().splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        results.append(
            DataPoint(
                text=obj.get("rawContent", obj.get("content", "")),
                source="twitter",
                timestamp=obj.get("date", ""),
                engagement=float(obj.get("likeCount", 0)) + float(obj.get("retweetCount", 0)),
                url=obj.get("url", ""),
            )
        )
    return results
