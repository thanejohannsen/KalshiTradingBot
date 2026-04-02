"""
Generic web scraper fallback — grabs headlines from financial news sites
when dedicated APIs are unavailable or exhausted.
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

from utils import DataPoint

logger = logging.getLogger("kalshi_bot.scraper")

# Sites that have useful headlines scrapable from their front / search pages
_SOURCES: list[dict[str, str]] = [
    {
        "name": "Reuters",
        "url": "https://www.reuters.com/site-search/?query={query}",
        "selector": "h3",
    },
    {
        "name": "AP News",
        "url": "https://apnews.com/search?q={query}",
        "selector": "h2, h3",
    },
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


def scrape_headlines(
    query: str,
    max_results: int = 15,
    timeout: int = 10,
) -> list[DataPoint]:
    """
    Scrape news headlines for *query* from a curated list of sites.
    This is a best-effort fallback — results vary by site structure.
    """
    results: list[DataPoint] = []

    for source in _SOURCES:
        if len(results) >= max_results:
            break
        try:
            url = source["url"].format(query=requests.utils.quote(query))
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup.select(source["selector"]):
                text = tag.get_text(strip=True)
                if not text or len(text) < 10:
                    continue

                link = ""
                a_tag = tag.find("a") if tag.name != "a" else tag
                if a_tag and a_tag.get("href"):
                    href = a_tag["href"]
                    if href.startswith("/"):
                        # Build absolute URL
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        href = f"{parsed.scheme}://{parsed.netloc}{href}"
                    link = href

                results.append(
                    DataPoint(
                        text=text,
                        source=f"scraper:{source['name']}",
                        url=link,
                    )
                )
                if len(results) >= max_results:
                    break

        except Exception as exc:
            logger.warning("Scraper failed for %s: %s", source["name"], exc)
            continue

    logger.debug("Scraper: fetched %d headlines for '%s'", len(results), query)
    return results
