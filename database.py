"""
SQLite persistence layer — schema, helpers, and data access.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe SQLite wrapper for the trading bot."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        # Create schema on first init
        with self._connect() as conn:
            self._create_tables(conn)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, timeout=30)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.execute("PRAGMA busy_timeout=30000")
        return self._local.conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                title       TEXT,
                yes_bid     REAL,
                yes_ask     REAL,
                no_bid      REAL,
                no_ask      REAL,
                volume_24h  REAL,
                liquidity   REAL,
                open_interest REAL,
                last_price  REAL,
                captured_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
                ON market_snapshots(ticker);
            CREATE INDEX IF NOT EXISTS idx_snapshots_time
                ON market_snapshots(captured_at);

            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                side            TEXT    NOT NULL,          -- 'yes' or 'no'
                action          TEXT    NOT NULL,          -- 'buy' or 'sell'
                entry_price     REAL    NOT NULL,
                size_dollars    REAL    NOT NULL,
                size_contracts  INTEGER NOT NULL DEFAULT 0,
                order_type      TEXT    NOT NULL DEFAULT 'limit',
                thesis          TEXT,
                predicted_prob  REAL,
                market_prob     REAL,
                edge            REAL,
                sentiment_score REAL,
                narrative       TEXT,
                confidence      TEXT,                       -- 'low','medium','high'
                signals         TEXT,                       -- JSON list of scan signal names
                status          TEXT    NOT NULL DEFAULT 'open',  -- open / won / lost / cancelled
                kalshi_order_id TEXT,
                entry_filled_at TEXT,
                tp_order_id     TEXT,
                tp_price        REAL,
                sl_order_id     TEXT,
                sl_price        REAL,
                pnl             REAL,
                opened_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                resolved_at     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trades_status
                ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_ticker
                ON trades(ticker);

            CREATE TABLE IF NOT EXISTS postmortems (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id          INTEGER NOT NULL REFERENCES trades(id),
                prob_estimate_off REAL,
                sentiment_wrong   INTEGER DEFAULT 0,
                timing_incorrect  INTEGER DEFAULT 0,
                sizing_aggressive INTEGER DEFAULT 0,
                failure_reason    TEXT,
                corrected_insight TEXT,
                pattern_to_avoid  TEXT,
                created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS heuristics (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS research_cache (
                ticker              TEXT    NOT NULL,
                query               TEXT    NOT NULL,
                sentiment_score     REAL    NOT NULL,
                narrative_summary   TEXT    NOT NULL,
                narrative_confidence TEXT   NOT NULL,
                data_point_count    INTEGER NOT NULL DEFAULT 0,
                bullish_pct         REAL    NOT NULL DEFAULT 0.0,
                bearish_pct         REAL    NOT NULL DEFAULT 0.0,
                cached_at           TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (ticker)
            );

            CREATE INDEX IF NOT EXISTS idx_research_cache_time
                ON research_cache(cached_at);

            CREATE TABLE IF NOT EXISTS shadow_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                side            TEXT    NOT NULL,
                entry_price     REAL    NOT NULL,
                predicted_prob  REAL,
                market_prob     REAL,
                edge            REAL,
                block_reason    TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'pending',  -- pending / shadow_won / shadow_lost
                pnl             REAL,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                resolved_at     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_shadow_status
                ON shadow_trades(status);
            """
        )
        # Run lightweight migrations for schema changes
        self._migrate(conn)

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------
    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns that may be missing from earlier schema versions."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        if "signals" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN signals TEXT")
        if "category" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN category TEXT DEFAULT ''")
        if "entry_filled_at" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN entry_filled_at TEXT")
        if "tp_order_id" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN tp_order_id TEXT")
        if "tp_price" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN tp_price REAL")
        if "sl_order_id" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN sl_order_id TEXT")
        if "sl_price" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN sl_price REAL")
        if "spread_at_entry" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN spread_at_entry REAL DEFAULT 0.0")
        if "manual" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN manual INTEGER NOT NULL DEFAULT 0")
        if "high_water_mark" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN high_water_mark REAL")

        # model_predictions table (added for ensemble B+C+E system)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "model_predictions" not in tables:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id    INTEGER NOT NULL REFERENCES trades(id),
                    model_name  TEXT    NOT NULL,
                    probability REAL    NOT NULL,
                    weight_used REAL    NOT NULL,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_model_preds_trade
                    ON model_predictions(trade_id);
                CREATE INDEX IF NOT EXISTS idx_model_preds_model
                    ON model_predictions(model_name);
                """
            )

    # ------------------------------------------------------------------
    # Market snapshots
    # ------------------------------------------------------------------
    def save_snapshot(self, snapshot: dict[str, Any], captured_at: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO market_snapshots
                    (ticker, title, yes_bid, yes_ask, no_bid, no_ask,
                     volume_24h, liquidity, open_interest, last_price, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["ticker"],
                    snapshot.get("title"),
                    snapshot.get("yes_bid"),
                    snapshot.get("yes_ask"),
                    snapshot.get("no_bid"),
                    snapshot.get("no_ask"),
                    snapshot.get("volume_24h"),
                    snapshot.get("liquidity"),
                    snapshot.get("open_interest"),
                    snapshot.get("last_price"),
                    captured_at or _now_iso(),
                ),
            )

    def has_recent_snapshots(self, max_age_hours: int = 2) -> bool:
        """Return True if there are snapshots within the last N hours."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM market_snapshots
                WHERE captured_at >= datetime('now', ?)
                """,
                (f"-{max_age_hours} hours",),
            ).fetchone()
            return (row["cnt"] if row else 0) > 0

    def get_recent_snapshots(
        self, ticker: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM market_snapshots
                WHERE ticker = ?
                ORDER BY captured_at DESC
                LIMIT ?
                """,
                (ticker, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_baseline(self, ticker: str, hours: int = 24) -> dict[str, Any] | None:
        """Average price/volume over the last N hours for baseline comparison."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    AVG(last_price)  AS avg_price,
                    AVG(volume_24h)  AS avg_volume,
                    AVG(yes_bid)     AS avg_yes_bid,
                    AVG(yes_ask)     AS avg_yes_ask,
                    COUNT(*)         AS sample_count
                FROM market_snapshots
                WHERE ticker = ?
                  AND captured_at >= datetime('now', ?)
                """,
                (ticker, f"-{hours} hours"),
            ).fetchone()
            if row and row["sample_count"] > 0:
                return dict(row)
            return None

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def insert_trade(self, trade: dict[str, Any]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades
                    (ticker, side, action, entry_price, size_dollars,
                     size_contracts, order_type, thesis, predicted_prob,
                     market_prob, edge, sentiment_score, narrative,
                     confidence, signals, category, status, kalshi_order_id,
                     spread_at_entry, manual, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["ticker"],
                    trade["side"],
                    trade["action"],
                    trade["entry_price"],
                    trade["size_dollars"],
                    trade.get("size_contracts", 0),
                    trade.get("order_type", "limit"),
                    trade.get("thesis"),
                    trade.get("predicted_prob"),
                    trade.get("market_prob"),
                    trade.get("edge"),
                    trade.get("sentiment_score"),
                    trade.get("narrative"),
                    trade.get("confidence"),
                    trade.get("signals"),
                    trade.get("category", ""),
                    trade.get("status", "open"),
                    trade.get("kalshi_order_id"),
                    trade.get("spread_at_entry", 0.0),
                    trade.get("manual", 0),
                    _now_iso(),
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_open_trades(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY opened_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_trade(self, trade_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            return dict(row) if row else None

    def cancel_trade(self, trade_id: int) -> None:
        """Mark a trade as cancelled (unfilled order that no longer exists on Kalshi)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET status = 'cancelled', resolved_at = ? WHERE id = ?",
                (_now_iso(), trade_id),
            )

    def resolve_trade(
        self, trade_id: int, status: str, pnl: float, closing_prob: float | None = None
    ) -> None:
        with self._connect() as conn:
            # Add closing_prob column if it doesn't exist yet (migration)
            try:
                conn.execute("ALTER TABLE trades ADD COLUMN closing_prob REAL")
            except Exception:
                pass  # column already exists
            conn.execute(
                """
                UPDATE trades
                SET status = ?, pnl = ?, resolved_at = ?, closing_prob = ?
                WHERE id = ?
                """,
                (status, pnl, _now_iso(), closing_prob, trade_id),
            )

    def get_latest_snapshot_price(self, ticker: str) -> float | None:
        """
        Return the most recent last_price from market_snapshots for a ticker.
        Used to record the closing line for CLV calculation.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT last_price FROM market_snapshots
                WHERE ticker = ?
                ORDER BY captured_at DESC
                LIMIT 1
                """,
                (ticker,),
            ).fetchone()
            if row and row["last_price"] is not None:
                return float(row["last_price"])
            return None

    def update_trade_order(
        self,
        trade_id: int,
        kalshi_order_id: str,
        entry_price: float,
    ) -> None:
        """Update tracked order metadata after cancel/replace repricing."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET kalshi_order_id = ?, entry_price = ?
                WHERE id = ?
                """,
                (kalshi_order_id, entry_price, trade_id),
            )

    def mark_trade_entry_filled(self, trade_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET entry_filled_at = COALESCE(entry_filled_at, ?)
                WHERE id = ?
                """,
                (_now_iso(), trade_id),
            )

    def update_trade_fill(
        self,
        trade_id: int,
        filled_contracts: int,
        size_dollars: float,
    ) -> None:
        """Update a trade's contract count and dollar size after a partial fill."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET size_contracts = ?,
                    size_dollars = ?,
                    entry_filled_at = COALESCE(entry_filled_at, ?)
                WHERE id = ?
                """,
                (filled_contracts, round(size_dollars, 2), _now_iso(), trade_id),
            )

    def set_trade_bracket_orders(
        self,
        trade_id: int,
        tp_order_id: str | None,
        tp_price: float | None,
        sl_order_id: str | None,
        sl_price: float | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET tp_order_id = ?,
                    tp_price = ?,
                    sl_order_id = ?,
                    sl_price = ?
                WHERE id = ?
                """,
                (tp_order_id, tp_price, sl_order_id, sl_price, trade_id),
            )

    def clear_trade_bracket_orders(self, trade_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET tp_order_id = NULL,
                    tp_price = NULL,
                    sl_order_id = NULL,
                    sl_price = NULL
                WHERE id = ?
                """,
                (trade_id,),
            )

    def toggle_trade_manual(self, trade_id: int) -> bool:
        """Toggle the manual flag on a trade. Returns the new value."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET manual = CASE WHEN manual = 1 THEN 0 ELSE 1 END WHERE id = ?",
                (trade_id,),
            )
            row = conn.execute("SELECT manual FROM trades WHERE id = ?", (trade_id,)).fetchone()
            return bool(row["manual"]) if row else False

    def update_high_water_mark(self, trade_id: int, price: float) -> None:
        """Update the high-water mark for trailing stop tracking."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET high_water_mark = MAX(COALESCE(high_water_mark, 0), ?)
                WHERE id = ?
                """,
                (price, trade_id),
            )

    def has_open_position(self, ticker: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM trades WHERE ticker = ? AND status = 'open'",
                (ticker,),
            ).fetchone()
            return row["cnt"] > 0  # type: ignore[index]

    def was_recently_traded(self, ticker: str, hours: float = 6.0) -> bool:
        """True if this ticker had a trade opened or closed in the last N hours.

        Prevents churning: the bot kept buying the same market right after
        being stopped out, losing money on repeated round-trips.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS cnt FROM trades
                   WHERE ticker = ?
                     AND (opened_at > datetime('now', ? || ' hours')
                          OR resolved_at > datetime('now', ? || ' hours'))""",
                (ticker, str(-hours), str(-hours)),
            ).fetchone()
            return row["cnt"] > 0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Postmortems
    # ------------------------------------------------------------------
    def insert_postmortem(self, pm: dict[str, Any]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO postmortems
                    (trade_id, prob_estimate_off, sentiment_wrong,
                     timing_incorrect, sizing_aggressive,
                     failure_reason, corrected_insight, pattern_to_avoid, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pm["trade_id"],
                    pm.get("prob_estimate_off"),
                    int(pm.get("sentiment_wrong", False)),
                    int(pm.get("timing_incorrect", False)),
                    int(pm.get("sizing_aggressive", False)),
                    pm.get("failure_reason"),
                    pm.get("corrected_insight"),
                    pm.get("pattern_to_avoid"),
                    _now_iso(),
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_postmortems(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM postmortems ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    def set_heuristic(self, key: str, value: Any) -> None:
        val_str = json.dumps(value) if not isinstance(value, str) else value
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO heuristics (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               updated_at = excluded.updated_at
                """,
                (key, val_str, _now_iso()),
            )

    def get_heuristic(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM heuristics WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def get_all_heuristics(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM heuristics").fetchall()
            return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Research cache
    # ------------------------------------------------------------------
    def get_cached_research(self, ticker: str, max_age_minutes: int = 30) -> dict[str, Any] | None:
        """Return cached research for ticker if fresh enough, else None."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM research_cache
                WHERE ticker = ?
                  AND cached_at >= datetime('now', ?)
                """,
                (ticker, f"-{max_age_minutes} minutes"),
            ).fetchone()
            return dict(row) if row else None

    def cache_research(self, ticker: str, query: str, result: dict[str, Any]) -> None:
        """Upsert a research result into the cache."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_cache
                    (ticker, query, sentiment_score, narrative_summary,
                     narrative_confidence, data_point_count, bullish_pct,
                     bearish_pct, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    query                = excluded.query,
                    sentiment_score      = excluded.sentiment_score,
                    narrative_summary    = excluded.narrative_summary,
                    narrative_confidence = excluded.narrative_confidence,
                    data_point_count     = excluded.data_point_count,
                    bullish_pct          = excluded.bullish_pct,
                    bearish_pct          = excluded.bearish_pct,
                    cached_at            = excluded.cached_at
                """,
                (
                    ticker,
                    query,
                    result["sentiment_score"],
                    result["narrative_summary"],
                    result["narrative_confidence"],
                    result["data_point_count"],
                    result.get("bullish_pct", 0.0),
                    result.get("bearish_pct", 0.0),
                    _now_iso(),
                ),
            )

    def purge_old_research(self, max_age_hours: int = 24) -> int:
        """Delete cache entries older than max_age_hours. Returns count deleted."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM research_cache
                WHERE cached_at < datetime('now', ?)
                """,
                (f"-{max_age_hours} hours",),
            )
            return cur.rowcount

    def purge_old_snapshots(self, keep_days: int = 14) -> int:
        """
        Delete market_snapshots older than keep_days, but always keep the
        most recent snapshot per ticker regardless of age (so baseline data
        is never lost for active markets).

        Returns the number of rows deleted.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM market_snapshots
                WHERE captured_at < datetime('now', ?)
                  AND id NOT IN (
                      SELECT MAX(id)
                      FROM market_snapshots
                      GROUP BY ticker
                  )
                """,
                (f"-{keep_days} days",),
            )
            deleted = cur.rowcount
            if deleted:
                # Reclaim disk space
                conn.execute("PRAGMA incremental_vacuum")
            return deleted

    # ------------------------------------------------------------------
    # Shadow trades (blocked-trade tracking)
    # ------------------------------------------------------------------
    def insert_shadow_trade(self, shadow: dict[str, Any]) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO shadow_trades
                    (ticker, side, entry_price, predicted_prob, market_prob,
                     edge, block_reason, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    shadow["ticker"],
                    shadow["side"],
                    shadow["entry_price"],
                    shadow.get("predicted_prob"),
                    shadow.get("market_prob"),
                    shadow.get("edge"),
                    shadow["block_reason"],
                    _now_iso(),
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_pending_shadow_trades(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM shadow_trades WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve_shadow_trade(self, shadow_id: int, status: str, pnl: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE shadow_trades
                SET status = ?, pnl = ?, resolved_at = ?
                WHERE id = ?
                """,
                (status, pnl, _now_iso(), shadow_id),
            )

    def get_shadow_stats(self) -> dict[str, Any]:
        """Return stats on blocked trades that have resolved."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                              AS total,
                    SUM(CASE WHEN status='shadow_won'  THEN 1 ELSE 0 END) AS would_have_won,
                    SUM(CASE WHEN status='shadow_lost' THEN 1 ELSE 0 END) AS would_have_lost,
                    SUM(CASE WHEN status='pending'     THEN 1 ELSE 0 END) AS pending,
                    COALESCE(SUM(CASE WHEN status IN ('shadow_won','shadow_lost') THEN pnl ELSE 0 END), 0) AS shadow_pnl
                FROM shadow_trades
                """
            ).fetchone()
            d = dict(row)
            resolved = (d["would_have_won"] or 0) + (d["would_have_lost"] or 0)
            d["shadow_win_rate"] = (d["would_have_won"] or 0) / resolved if resolved > 0 else 0.0
            return d

    # ------------------------------------------------------------------
    def get_recent_trades(self, limit: int = 30) -> list[dict[str, Any]]:
        """Return the most recent trades (any status) for UI display."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_postmortems_with_trades(self, limit: int = 10) -> list[dict[str, Any]]:
        """Join postmortems with their trade for display context."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*, t.ticker, t.side, t.entry_price,
                       t.pnl AS trade_pnl, t.status AS trade_status
                FROM postmortems p
                JOIN trades t ON p.trade_id = t.id
                ORDER BY p.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # Stats
    # ------------------------------------------------------------------
    def get_trade_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                    AS total_trades,
                    SUM(CASE WHEN status='won'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_trades,
                    SUM(CASE WHEN status='exited_profit' THEN 1 ELSE 0 END) AS early_exits_profit,
                    SUM(CASE WHEN status='exited_loss'   THEN 1 ELSE 0 END) AS early_exits_loss,
                    COALESCE(SUM(pnl), 0)                       AS total_pnl
                FROM trades
                """
            ).fetchone()
            d = dict(row)
            d["realized_wins"] = (d["wins"] or 0) + (d["early_exits_profit"] or 0)
            d["realized_losses"] = (d["losses"] or 0) + (d["early_exits_loss"] or 0)
            total_resolved = (d["wins"] or 0) + (d["losses"] or 0) + \
                             (d["early_exits_profit"] or 0) + (d["early_exits_loss"] or 0)
            d["win_rate"] = (
                (d["realized_wins"] / total_resolved)
                if total_resolved > 0 else 0.0
            )
            return d

    # ------------------------------------------------------------------
    # Model predictions (ensemble B+C+E)
    # ------------------------------------------------------------------
    def save_model_predictions(self, trade_id: int, model_votes: list[dict]) -> None:
        """Store each sub-model's probability estimate for a trade."""
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO model_predictions (trade_id, model_name, probability, weight_used)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (trade_id, v["model"], v["probability"], v["weight"])
                    for v in model_votes
                    if "model" in v and "probability" in v
                ],
            )

    def get_model_predictions_for_trade(self, trade_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_predictions WHERE trade_id = ?", (trade_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_model_performance(self, model_name: str, lookback: int = 50) -> dict[str, Any]:
        """Win rate + avg probability error for a sub-model over recent trades."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)  AS total,
                    SUM(CASE WHEN t.status='won'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN t.status='lost' THEN 1 ELSE 0 END) AS losses,
                    AVG(ABS(mp.probability -
                        CASE WHEN t.status='won' THEN 1.0 ELSE 0.0 END)) AS avg_error
                FROM model_predictions mp
                JOIN trades t ON mp.trade_id = t.id
                WHERE t.status IN ('won','lost')
                  AND mp.model_name = ?
                ORDER BY t.resolved_at DESC
                LIMIT ?
                """,
                (model_name, lookback),
            ).fetchone()
            d = dict(row) if row else {}
            total = (d.get("wins") or 0) + (d.get("losses") or 0)
            d["win_rate"] = (d.get("wins") or 0) / total if total > 0 else 0.5
            d["total"] = total
            return d

    def get_all_model_performance(self) -> dict[str, dict[str, Any]]:
        """Return performance dict keyed by model name for all known sub-models."""
        model_names = [
            "MarketAnchorModel", "SentimentModel", "MomentumModel",
            "VolumeModel", "ConsensusModel",
        ]
        return {name: self.get_model_performance(name) for name in model_names}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_category_stats(self) -> dict[str, dict[str, Any]]:
        """Return per-category win/loss/pnl stats from resolved trades.

        Category strings are normalised to canonical uppercase names so that
        Kalshi API strings like 'Sports' or 'Climate and Weather' are merged
        with their canonical counterparts ('SPORTS', 'WEATHER', etc.).
        """
        from models.category_profiles import CategoryProfileLoader
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    COALESCE(NULLIF(category, ''), SUBSTR(ticker, 1, 7)) AS cat,
                    ticker,
                    COUNT(*)                                              AS total,
                    SUM(CASE WHEN status='won'  THEN 1 ELSE 0 END)       AS wins,
                    SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END)       AS losses,
                    COALESCE(SUM(pnl), 0)                                 AS total_pnl
                FROM trades
                WHERE status IN ('won', 'lost')
                GROUP BY cat
                HAVING total >= 1
                ORDER BY total_pnl DESC
                """
            ).fetchall()
            # Merge rows by normalised category name
            merged: dict[str, dict[str, Any]] = {}
            for r in rows:
                d = dict(r)
                raw_cat = d.pop("cat")
                ticker = d.pop("ticker", "")
                canonical = CategoryProfileLoader._normalise(raw_cat) \
                    if raw_cat else CategoryProfileLoader._normalise(ticker[:6])
                if canonical not in merged:
                    merged[canonical] = {"wins": 0, "losses": 0, "total": 0, "total_pnl": 0.0}
                merged[canonical]["wins"]      += d["wins"] or 0
                merged[canonical]["losses"]    += d["losses"] or 0
                merged[canonical]["total"]     += d["total"] or 0
                merged[canonical]["total_pnl"] += d["total_pnl"] or 0.0
            # Compute win_rate for each merged entry
            for cat, d in merged.items():
                total_resolved = d["wins"] + d["losses"]
                d["win_rate"] = d["wins"] / total_resolved if total_resolved > 0 else 0.0
            return merged
