"""
Kalshi API client — uses direct HTTP requests with RSA-PSS signing.
Matches the proven auth approach from PlaceKalshiOrder.py.
"""

from __future__ import annotations

import base64
import datetime
import logging
import time
import uuid
from typing import Any, Callable
from urllib.parse import urlparse, urlencode

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import Config
from utils import MarketData

logger = logging.getLogger("kalshi_bot.kalshi_client")


class KalshiAPIClient:
    """Direct HTTP client for the Kalshi API with RSA-PSS auth."""

    def __init__(self, cfg: Config, shutdown_check: Callable[[], bool] | None = None) -> None:
        self._cfg = cfg
        self._shutdown_check = shutdown_check or (lambda: False)
        self._base_url: str = cfg.kalshi_api_base
        self._api_key_id: str = cfg.kalshi_api_key
        self._private_key: Any = None
        self._session: requests.Session = requests.Session()
        self._last_call: float = 0.0
        self._rate_limit: int = 10  # requests per second

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Load private key and verify authentication with a balance check."""
        key_path = self._cfg.kalshi_private_key_path
        with open(key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        logger.info("Loaded private key from %s", key_path)
        logger.info("Connected to Kalshi API at %s", self._base_url)

    # ------------------------------------------------------------------
    # Auth / signing  (matches PlaceKalshiOrder.py exactly)
    # ------------------------------------------------------------------
    def _sign(self, method: str, path: str) -> dict[str, str]:
        """Create auth headers using RSA-PSS signature."""
        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        # Strip query parameters before signing
        path_without_query = path.split("?")[0]
        message = f"{timestamp}{method}{path_without_query}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        min_interval = 1.0 / max(self._rate_limit, 1)
        elapsed = time.monotonic() - self._last_call
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        """Make an HTTP request to the Kalshi API."""
        url = self._base_url + path
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

        # Signing uses the full URL path (e.g. /trade-api/v2/markets)
        sign_path = urlparse(url).path
        headers = self._sign(method.upper(), sign_path) if auth else {}
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        resp = self._session.request(
            method, url, headers=headers, json=json_body, timeout=30
        )
        if not resp.ok:
            detail = ""
            try:
                detail = resp.text.strip()
            except Exception:
                detail = ""
            if detail:
                raise requests.HTTPError(
                    f"{resp.status_code} Client Error: {detail}",
                    response=resp,
                )
            resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _call_with_retry(
        self,
        method: str,
        path: str,
        retries: int = 3,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        auth: bool = True,
    ) -> dict[str, Any]:
        for attempt in range(retries):
            if self._shutdown_check():
                raise RuntimeError("Shutdown requested")
            self._throttle()
            try:
                return self._request(method, path, params=params, json_body=json_body, auth=auth)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited, backing off %ds (attempt %d/%d)", wait, attempt + 1, retries)
                    time.sleep(wait)
                elif attempt == retries - 1:
                    raise
                else:
                    logger.warning("API error (attempt %d/%d): %s %s → %s", attempt + 1, retries, method, path, exc)
                    time.sleep(1)
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                logger.warning("Request error (attempt %d/%d): %s", attempt + 1, retries, exc)
                time.sleep(1)
        raise RuntimeError(f"API call failed after {retries} retries")

    # ------------------------------------------------------------------
    # Market data  (public — no auth needed, but auth works too)
    # ------------------------------------------------------------------
    def get_active_markets(
        self,
        limit: int = 200,
        cursor: str | None = None,
        status: str = "open",
        max_markets: int = 5000,
    ) -> list[MarketData]:
        """Fetch all active markets via the events endpoint.

        The /markets endpoint is flooded with MVE combo markets.
        Using /events?with_nested_markets=true returns only real markets.
        """
        all_markets: list[MarketData] = []
        page = 0
        while True:
            if self._shutdown_check():
                logger.info("Shutdown requested — stopping market fetch at %d markets", len(all_markets))
                break

            params: dict[str, Any] = {
                "limit": limit,
                "status": status,
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor

            data = self._call_with_retry("GET", "/events", params=params, auth=False)
            events = data.get("events", [])

            for event in events:
                event_category = event.get("category", "")
                for m in event.get("markets", []):
                    # Skip combo/parlay markets (MVE custom strikes)
                    if m.get("strike_type") == "custom" or m.get("mve_collection_ticker"):
                        continue
                    # Propagate event category to market if not already set
                    if event_category and not m.get("category"):
                        m["category"] = event_category
                    all_markets.append(self._parse_market(m))

            page += 1
            logger.debug("  page %d: %d events (total markets: %d)", page, len(events), len(all_markets))

            if len(all_markets) >= max_markets:
                logger.info("Reached max_markets cap (%d) — stopping pagination", max_markets)
                break

            cursor = data.get("cursor")
            if not cursor or len(events) < limit:
                break

            time.sleep(0.5)

        logger.info("Fetched %d active markets from %d event pages", len(all_markets), page)
        return all_markets

    _event_category_cache: dict[str, str] = {}

    def get_market(self, ticker: str) -> MarketData:
        data = self._call_with_retry("GET", f"/markets/{ticker}", auth=False)
        market = data.get("market", data)
        parsed = self._parse_market(market)
        # Single-market endpoint often lacks category — fetch from event if missing
        if not parsed.category and parsed.event_ticker:
            et = parsed.event_ticker
            if et in self._event_category_cache:
                parsed.category = self._event_category_cache[et]
            else:
                try:
                    evt = self._call_with_retry(
                        "GET", f"/events/{et}", auth=False,
                    )
                    event = evt.get("event", evt)
                    cat = event.get("category", "")
                    self._event_category_cache[et] = cat
                    if cat:
                        parsed.category = cat
                except Exception:
                    pass
        return parsed

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        data = self._call_with_retry(
            "GET", f"/markets/{ticker}/orderbook", params={"depth": depth}, auth=False
        )
        return {
            "yes": data.get("yes", []),
            "no": data.get("no", []),
        }

    def get_market_history(
        self, ticker: str, series_ticker: str, period_interval: int = 60
    ) -> list[dict[str, Any]]:
        """Fetch candlestick data for historical baseline."""
        try:
            import time
            now = int(time.time())
            start_ts = now - 86400  # 24 hours ago
            data = self._call_with_retry(
                "GET",
                f"/series/{series_ticker}/markets/{ticker}/candlesticks",
                params={"period_interval": period_interval, "start_ts": start_ts, "end_ts": now},
                auth=False,
            )
            return data.get("candlesticks", [])
        except Exception as exc:
            logger.warning("Could not fetch history for %s: %s", ticker, exc)
            return []

    # ------------------------------------------------------------------
    # Portfolio  (authenticated)
    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        """Return available balance in dollars."""
        data = self._call_with_retry("GET", "/portfolio/balance")
        # Try dollar field first, then legacy cents
        balance_str = data.get("balance_dollars")
        if balance_str is not None:
            return float(balance_str)
        balance = data.get("balance", 0)
        if isinstance(balance, str):
            return float(balance)
        # Legacy: balance in cents
        return balance / 100.0

    def get_positions(self) -> list[dict[str, Any]]:
        data = self._call_with_retry("GET", "/portfolio/positions")
        positions = data.get("market_positions", [])
        parsed: list[dict[str, Any]] = []
        for p in positions:
            ticker = p.get("ticker", "")

            # New API shape: position_fp is signed contracts as a string.
            # Positive means YES exposure, negative means NO exposure.
            position_fp = p.get("position_fp")
            yes_count = 0
            no_count = 0

            try:
                pos = float(position_fp) if position_fp is not None else None
            except (ValueError, TypeError):
                pos = None

            if pos is not None:
                if pos > 0:
                    yes_count = int(round(pos))
                elif pos < 0:
                    no_count = int(round(abs(pos)))
            else:
                # Legacy fallback fields
                try:
                    yes_count = int(float(p.get("position", 0) or 0))
                except (ValueError, TypeError):
                    yes_count = 0
                try:
                    no_count = int(float(p.get("no_count", 0) or 0))
                except (ValueError, TypeError):
                    no_count = 0

            parsed.append(
                {
                    "ticker": ticker,
                    "yes_count": yes_count,
                    "no_count": no_count,
                    "market_value": p.get("market_value", 0),
                    "resting_orders_count": p.get("resting_orders_count", 0),
                }
            )
        return parsed

    def get_position_count(self, ticker: str, side: str) -> int:
        """Return the number of contracts we actually hold for ticker+side on Kalshi.

        This is the source of truth — always call before selling to prevent
        overselling (which Kalshi treats as opening an opposite position).
        """
        try:
            positions = self.get_positions()
            for p in positions:
                if p.get("ticker") == ticker:
                    if side == "yes":
                        return int(p.get("yes_count", 0) or 0)
                    else:
                        return int(p.get("no_count", 0) or 0)
        except Exception as exc:
            logger.warning("Could not verify position for %s: %s", ticker, exc)
            # Fail safe: return 0 so the sell is blocked
            return 0
        return 0

    # ------------------------------------------------------------------
    # Fills  (authenticated)
    # ------------------------------------------------------------------
    def get_fills(self, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch portfolio fills (actual execution records).

        This is the most reliable source of entry prices — it returns the
        actual price and count for each fill, unlike orders which show the
        order price (which may differ from the execution price).
        """
        fills: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"limit": limit}
            if ticker:
                params["ticker"] = ticker
            if cursor:
                params["cursor"] = cursor

            data = self._call_with_retry("GET", "/portfolio/fills", params=params)
            batch = data.get("fills", [])

            for f in batch:
                price = _parse_dollar_field(
                    f.get("yes_price_dollars") if f.get("side") == "yes" else f.get("no_price_dollars"),
                    f.get("yes_price") if f.get("side") == "yes" else f.get("no_price"),
                )
                count = _parse_count_field(f.get("count_fp"), f.get("count"))
                fills.append({
                    "ticker": f.get("ticker", ""),
                    "side": f.get("side", ""),
                    "action": f.get("action", ""),
                    "price": price,
                    "count": count,
                    "order_id": f.get("order_id", ""),
                    "created_time": f.get("created_time", ""),
                })

            cursor = data.get("cursor")
            if not cursor or len(batch) < limit:
                break

        return fills

    # ------------------------------------------------------------------
    # Orders  (authenticated)
    # ------------------------------------------------------------------
    def place_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        price: float | None = None,
        order_type: str = "limit",
    ) -> dict[str, Any]:
        """Place an order on Kalshi.

        SAFETY: For sell orders, we verify against the Kalshi positions API
        that we actually hold enough contracts. If we hold fewer than requested,
        the count is clamped down. If we hold zero, the sell is blocked entirely.
        This prevents overselling, which Kalshi treats as opening an opposite
        position (e.g., selling YES you don't own = buying NO).
        """
        if action == "sell":
            held = self.get_position_count(ticker, side)
            if held <= 0:
                logger.error(
                    "SELL BLOCKED: %s %s x%d — Kalshi shows 0 contracts held. "
                    "Refusing to sell to prevent opening opposite position.",
                    side, ticker, count,
                )
                return {"order_id": None, "status": "blocked_no_position"}
            if count > held:
                logger.warning(
                    "SELL CAPPED: %s %s requested x%d but only x%d held on Kalshi. "
                    "Capping to x%d to prevent overselling.",
                    side, ticker, count, held, held,
                )
                count = held

        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "client_order_id": str(uuid.uuid4()),
        }

        if order_type == "limit" and price is not None:
            # Kalshi accepts yes_price / no_price in cents (integer)
            price_cents = int(round(price * 100))
            if side == "yes":
                body["yes_price"] = price_cents
            else:
                body["no_price"] = price_cents

        logger.info(
            "Placing order: %s %s %s x%d @ %s (%s)",
            action, side, ticker, count,
            f"${price:.4f}" if price else "market",
            order_type,
        )

        data = self._call_with_retry("POST", "/portfolio/orders", json_body=body)
        order = data.get("order", data)
        return {
            "order_id": order.get("order_id"),
            "status": order.get("status"),
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "price": price,
        }

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._call_with_retry("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        data = self._call_with_retry("GET", f"/portfolio/orders/{order_id}")
        order = data.get("order", data)
        return self._parse_order(order)

    def list_orders(self, limit: int = 200, only_active: bool = False) -> list[dict[str, Any]]:
        """Return portfolio orders, optionally filtering to only active orders."""
        orders: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"limit": limit}
            if cursor:
                params["cursor"] = cursor

            data = self._call_with_retry("GET", "/portfolio/orders", params=params)
            batch = data.get("orders", [])

            for raw_order in batch:
                order = self._parse_order(raw_order)
                if only_active and not self._is_active_order(order):
                    continue
                orders.append(order)

            cursor = data.get("cursor")
            if not cursor or len(batch) < limit:
                break

        return orders

    def _parse_order(self, order: dict[str, Any]) -> dict[str, Any]:
        price = _parse_dollar_field(
            order.get("yes_price_dollars") if order.get("side") == "yes" else order.get("no_price_dollars"),
            order.get("yes_price") if order.get("side") == "yes" else order.get("no_price"),
        )
        return {
            "order_id": order.get("order_id"),
            "status": order.get("status"),
            "ticker": order.get("ticker"),
            "side": order.get("side"),
            "action": order.get("action"),
            "price": price,
            "remaining_count": _parse_count_field(
                order.get("remaining_count_fp"),
                order.get("remaining_count"),
            ),
            "filled_count": _parse_count_field(
                order.get("fill_count_fp"),
                order.get("fill_count"),
            ),
            "initial_count": _parse_count_field(
                order.get("initial_count_fp"),
                order.get("initial_count"),
            ),
        }

    @staticmethod
    def _is_active_order(order: dict[str, Any]) -> bool:
        status = str(order.get("status", "")).lower()
        if status in {"executed", "filled", "canceled", "cancelled", "expired", "rejected", "failed"}:
            return False
        return int(order.get("remaining_count", 0) or 0) > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_market(m: Any) -> MarketData:
        """Convert API dict → MarketData dataclass.

        The Kalshi v2 API returns dollar-denominated string fields:
          yes_bid_dollars, yes_ask_dollars, no_bid_dollars, no_ask_dollars,
          last_price_dollars, liquidity_dollars, volume_24h_fp, open_interest_fp
        Older field names (yes_bid, yes_ask, etc.) may also appear.
        """
        get = m.get if isinstance(m, dict) else lambda k, d=None: getattr(m, k, d)

        return MarketData(
            ticker=get("ticker", ""),
            title=get("title", ""),
            yes_bid=_parse_dollar_field(get("yes_bid_dollars"), get("yes_bid")),
            yes_ask=_parse_dollar_field(get("yes_ask_dollars"), get("yes_ask")),
            no_bid=_parse_dollar_field(get("no_bid_dollars"), get("no_bid")),
            no_ask=_parse_dollar_field(get("no_ask_dollars"), get("no_ask")),
            volume_24h=_parse_float_field(get("volume_24h_fp"), get("volume_24h")),
            liquidity=_parse_float_field(get("liquidity_dollars"), get("liquidity")),
            open_interest=_parse_float_field(get("open_interest_fp"), get("open_interest")),
            last_price=_parse_dollar_field(get("last_price_dollars"), get("last_price")),
            close_time=get("close_time") or get("expected_expiration_time"),
            status=get("status", "open"),
            subtitle=get("subtitle", "") or get("yes_sub_title", "") or "",
            category=get("category", "") or "",
            event_ticker=get("event_ticker", "") or "",
        )


def _parse_dollar_field(dollar_str: Any, legacy_cents: Any = None) -> float:
    """Parse a dollar-denominated string field, falling back to legacy cents."""
    # Try the new _dollars field first (string like "0.5200")
    if dollar_str is not None:
        try:
            return float(dollar_str)
        except (ValueError, TypeError):
            pass
    # Fall back to legacy cent field
    if legacy_cents is not None:
        try:
            val = float(legacy_cents)
            return val / 100.0 if val > 1.0 else val
        except (ValueError, TypeError):
            pass
    return 0.0


def _parse_float_field(fp_str: Any, legacy_int: Any = None) -> float:
    """Parse a string float field, falling back to legacy integer."""
    if fp_str is not None:
        try:
            return float(fp_str)
        except (ValueError, TypeError):
            pass
    if legacy_int is not None:
        try:
            return float(legacy_int)
        except (ValueError, TypeError):
            pass
    return 0.0


def _parse_count_field(fp_str: Any, legacy_int: Any = None) -> int:
    """Parse count fields that may arrive as string floats."""
    if fp_str is not None:
        try:
            return int(round(float(fp_str)))
        except (ValueError, TypeError):
            pass
    if legacy_int is not None:
        try:
            return int(round(float(legacy_int)))
        except (ValueError, TypeError):
            pass
    return 0
