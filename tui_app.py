"""
Tkinter GUI for the Kalshi Trading Bot.

Layout (PanedWindow — drag to resize):
  +------------------------------------------------------------------+
  |                        STATUS BAR                                 |
  +------------------+-----------------------------------------------+
  |    RESEARCH      |              POSITIONS                         |
  |    (narrow)      |              (wide)                            |
  |                  |                                                |
  +--------+---------+-----------------------------------------------+
  |LEARNING|                     LOG                                  |
  +--------+---------------------------------------------------------+
"""

from __future__ import annotations

import logging
import re
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from main import TradingBot
    from tui_state import BotState

logger = logging.getLogger("kalshi_bot.tui")

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG           = "#0d0d14"
BG_PANEL     = "#14141f"
BG_STATUS    = "#0a0a12"
BG_HEADER    = "#101020"
FG           = "#d0d0e0"
FG_DIM       = "#4a4a60"
FG_SUBTLE    = "#70708a"
FG_LABEL     = "#9090aa"
GREEN        = "#4ade80"
GREEN_SOFT   = "#22c55e"
RED          = "#f87171"
YELLOW       = "#fbbf24"
CYAN         = "#22d3ee"
MAGENTA      = "#c084fc"
BLUE         = "#60a5fa"
ACCENT       = "#818cf8"
BORDER       = "#1e1e30"
SASH         = "#2a2a40"

FONT         = ("Segoe UI", 10)
FONT_SM      = ("Segoe UI", 9)
FONT_BOLD    = ("Segoe UI Semibold", 10)
FONT_TITLE   = ("Segoe UI Semibold", 10)
FONT_STATUS  = ("Segoe UI", 10)
FONT_STATUS_B = ("Segoe UI Semibold", 10)
FONT_LOG     = ("Consolas", 9)
FONT_MONO    = ("Consolas", 10)

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _strip_md(text: str) -> str:
    return _MD_BOLD_RE.sub(r"\1", text)


def _clean_title(title: str) -> str:
    """Clean up a Kalshi market title for display."""
    clean = _strip_md(title).strip()
    for prefix in ("Will ", "will "):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    clean = clean.rstrip("?").strip()
    if clean:
        clean = clean[0].upper() + clean[1:]
    return clean


def _position_desc(side: str, subtitle: str) -> str:
    """Build a clear position description from side + subtitle.

    Example: side="no", subtitle="49° or below"
      → "NO  49° or below will NOT happen"
    """
    s = subtitle.strip()
    if side.lower() == "yes":
        if s:
            return f"YES — {s}"
        return "YES — this will happen"
    else:
        if s:
            return f"NO — not {s}"
        return "NO — this will NOT happen"


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Text widget helpers
# ---------------------------------------------------------------------------
def _make_text(parent: tk.Widget, font: tuple = FONT) -> tk.Text:
    t = tk.Text(
        parent, bg=BG_PANEL, fg=FG, font=font, wrap="word",
        borderwidth=0, highlightthickness=0, padx=14, pady=10,
        cursor="arrow", insertbackground=BG_PANEL,
        selectbackground="#2a2a40", relief="flat", spacing1=2, spacing3=2,
    )
    t.config(state="disabled")
    for tag, kw in {
        "h2":           dict(font=FONT_BOLD, foreground=ACCENT, spacing1=8, spacing3=4),
        "dim":          dict(foreground=FG_DIM),
        "subtle":       dict(foreground=FG_SUBTLE),
        "label":        dict(foreground=FG_LABEL),
        "bold":         dict(font=FONT_BOLD, foreground=FG),
        "green":        dict(foreground=GREEN),
        "red":          dict(foreground=RED),
        "yellow":       dict(foreground=YELLOW),
        "cyan":         dict(foreground=CYAN),
        "magenta":      dict(foreground=MAGENTA),
        "accent":       dict(foreground=ACCENT),
        "bold_green":   dict(font=FONT_BOLD, foreground=GREEN),
        "bold_red":     dict(font=FONT_BOLD, foreground=RED),
        "bold_cyan":    dict(font=FONT_BOLD, foreground=CYAN),
        "bold_magenta": dict(font=FONT_BOLD, foreground=MAGENTA),
        "sm":           dict(font=FONT_SM, foreground=FG_SUBTLE),
        "mono":         dict(font=FONT_MONO, foreground=FG),
    }.items():
        t.tag_configure(tag, **kw)
    return t


def _w(widget: tk.Text, text: str, *tags: str) -> None:
    widget.config(state="normal")
    widget.insert("end", text, tags)
    widget.config(state="disabled")


def _clear(widget: tk.Text) -> None:
    widget.config(state="normal")
    widget.delete("1.0", "end")
    widget.config(state="disabled")


# ---------------------------------------------------------------------------
# Panel frame builder
# ---------------------------------------------------------------------------
def _panel(parent: tk.Widget, title: str, text_font: tuple = FONT) -> tk.Text:
    """Build a panel with a header bar and scrollable text area. Returns the Text."""
    frame = tk.Frame(parent, bg=BG_PANEL)
    frame.pack(fill="both", expand=True)
    frame.rowconfigure(1, weight=1)
    frame.columnconfigure(0, weight=1)

    hdr = tk.Frame(frame, bg=BG_HEADER, height=26)
    hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
    hdr.grid_propagate(False)
    tk.Label(hdr, text=f"  {title}", bg=BG_HEADER, fg=ACCENT,
             font=FONT_TITLE, anchor="w").pack(fill="both", expand=True, padx=4)

    txt = _make_text(frame, font=text_font)
    sb = ttk.Scrollbar(frame, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    txt.grid(row=1, column=0, sticky="nsew")
    sb.grid(row=1, column=1, sticky="ns")
    return txt


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
class TradingBotGUI:

    def __init__(self, bot: "TradingBot", state: "BotState") -> None:
        self._bot = bot
        self._state = state
        self._log_line_count = 0
        self._pos_fetch_running = False  # guard against thread pile-up

        # ── Root ────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Kalshi Trading Bot")
        self.root.configure(bg=BG)
        self.root.geometry("1500x850")
        self.root.minsize(950, 500)

        # ── ttk style ───────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar",
                        background=BG_PANEL, troughcolor=BG,
                        borderwidth=0, arrowsize=0, relief="flat")
        style.map("Vertical.TScrollbar", background=[("active", SASH)])

        # ── Status bar (top, fixed) ─────────────────────────────────
        status = tk.Frame(self.root, bg=BG_STATUS, height=38)
        status.pack(fill="x", side="top")
        status.pack_propagate(False)
        self._status_inner = tk.Frame(status, bg=BG_STATUS)
        self._status_inner.pack(fill="both", expand=True, padx=16)
        self._status_labels: dict[str, tk.Label] = {}
        self._status_separators: list[tk.Label] = []

        # ── Vertical split: top panels / bottom bar ─────────────────
        vpane = tk.PanedWindow(
            self.root, orient="vertical", bg=BG,
            sashwidth=5, sashrelief="flat", borderwidth=0,
            opaqueresize=True,
        )
        vpane.pack(fill="both", expand=True)

        # ── Top pane: Research | Positions (horizontal split) ───────
        hpane_top = tk.PanedWindow(
            vpane, orient="horizontal", bg=BG,
            sashwidth=5, sashrelief="flat", borderwidth=0,
            opaqueresize=True,
        )

        # Research (left)
        research_frame = tk.Frame(hpane_top, bg=BG_PANEL,
                                  highlightbackground=BORDER, highlightthickness=1)
        self._research_text = _panel(research_frame, "RESEARCH")
        hpane_top.add(research_frame, minsize=180, stretch="never")

        # Positions (right — gets all extra space)
        positions_frame = tk.Frame(hpane_top, bg=BG_PANEL,
                                   highlightbackground=BORDER, highlightthickness=1)
        self._positions_text = _panel(positions_frame, "POSITIONS")
        hpane_top.add(positions_frame, minsize=400, stretch="always")

        vpane.add(hpane_top, minsize=200, stretch="always")

        # ── Bottom pane: Learning | Log (horizontal split) ──────────
        hpane_bot = tk.PanedWindow(
            vpane, orient="horizontal", bg=BG,
            sashwidth=5, sashrelief="flat", borderwidth=0,
            opaqueresize=True,
        )

        # Learning (left, small)
        learning_frame = tk.Frame(hpane_bot, bg=BG_PANEL,
                                  highlightbackground=BORDER, highlightthickness=1)
        self._postmortem_text = _panel(learning_frame, "LEARNING")
        hpane_bot.add(learning_frame, minsize=200, stretch="never")

        # Log (right, gets extra space)
        log_frame = tk.Frame(hpane_bot, bg=BG_PANEL,
                             highlightbackground=BORDER, highlightthickness=1)
        self._log_text = _panel(log_frame, "LOG", text_font=FONT_LOG)
        hpane_bot.add(log_frame, minsize=300, stretch="always")

        vpane.add(hpane_bot, minsize=120, stretch="never")

        # ── Set initial sash positions after window renders ─────────
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        # Top: research gets ~22% width
        hpane_top.sash_place(0, int(w * 0.22), 0)
        # Bottom: learning gets ~22% width
        hpane_bot.sash_place(0, int(w * 0.22), 0)
        # Vertical: bottom panel gets ~20% height
        vpane.sash_place(0, 0, int(h * 0.75))

        # ── Configure static card tags once ─────────────────────────
        self._card_bg = "#161625"
        pt = self._positions_text
        pt.tag_configure("card_title", font=FONT_BOLD, foreground=FG,
                         lmargin1=14, spacing1=2, spacing3=0)
        pt.tag_configure("card_border_top", foreground="#2a2a45",
                         font=("Consolas", 1), spacing1=6, spacing3=0)
        pt.tag_configure("card_border_bot", foreground="#2a2a45",
                         font=("Consolas", 1), spacing1=0, spacing3=6)
        pt.tag_configure("cr", background=self._card_bg, lmargin1=16,
                         lmargin2=16, rmargin=12, spacing1=1, spacing3=1)
        pt.tag_configure("cr_side_yes", foreground=GREEN, font=FONT_BOLD,
                         background=self._card_bg)
        pt.tag_configure("cr_side_no", foreground=RED, font=FONT_BOLD,
                         background=self._card_bg)
        pt.tag_configure("cr_sub", foreground=FG, background=self._card_bg)
        pt.tag_configure("cr_lbl", foreground=FG_LABEL, font=FONT_SM,
                         background=self._card_bg)
        pt.tag_configure("cr_val", foreground=FG, font=FONT_MONO,
                         background=self._card_bg)
        pt.tag_configure("cr_dim", foreground=FG_SUBTLE, background=self._card_bg)
        pt.tag_configure("cr_grn", foreground=GREEN, font=FONT_BOLD,
                         background=self._card_bg)
        pt.tag_configure("cr_red", foreground=RED, font=FONT_BOLD,
                         background=self._card_bg)
        pt.tag_configure("cr_payout", foreground=FG, font=FONT_MONO,
                         background=self._card_bg)
        pt.tag_configure("cr_sep", foreground="#2a2a45", background=self._card_bg,
                         font=("Consolas", 7))

        # ── Schedule refreshes ──────────────────────────────────────
        self.root.after(200, self._refresh_status)
        self.root.after(200, self._refresh_research)
        self.root.after(200, self._refresh_positions)
        self.root.after(200, self._refresh_postmortem)
        self.root.after(500, self._refresh_log)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _refresh_status(self) -> None:
        try:
            phase = self._state.get_phase().upper()
            cycle = self._state.cycle_count
            balance = self._state.get_balance()
            error = self._state.get_error()

            try:
                stats = self._bot.db.get_trade_stats()
            except Exception:
                stats = {}

            mode = "LIVE" if self._bot.cfg.is_live else "PAPER"
            pnl = stats.get("total_pnl", 0) or 0
            wr = (stats.get("win_rate", 0) or 0) * 100
            wins = stats.get("realized_wins", 0) or 0
            losses = stats.get("realized_losses", 0) or 0
            open_count = stats.get("open_trades", 0) or 0
            bal_str = f"${balance:.2f}" if balance else "---"

            items = [
                ("mode", mode, BLUE if mode == "LIVE" else GREEN_SOFT, True),
                ("bal", f"Balance  {bal_str}", FG, False),
                ("cycle", f"Cycle {cycle}", FG_SUBTLE, False),
                ("phase", phase, CYAN, True),
                ("pnl", f"P&L  ${pnl:+.2f}", GREEN if pnl >= 0 else RED, True),
                ("record", f"{wins}W / {losses}L  ({wr:.0f}%)", FG_LABEL, False),
                ("open", f"{open_count} Open", FG_SUBTLE, False),
            ]
            if error:
                items.append(("error", f"ERR  {_trunc(error, 40)}", RED, True))

            if len(self._status_labels) != len(items):
                for lbl in self._status_labels.values():
                    lbl.destroy()
                for sep in self._status_separators:
                    sep.destroy()
                self._status_labels.clear()
                self._status_separators.clear()

                for i, (key, text, fg, bold) in enumerate(items):
                    lbl = tk.Label(
                        self._status_inner, text=text, bg=BG_STATUS, fg=fg,
                        font=FONT_STATUS_B if bold else FONT_STATUS, padx=6,
                    )
                    lbl.pack(side="left", pady=8)
                    self._status_labels[key] = lbl
                    if i < len(items) - 1:
                        sep = tk.Label(self._status_inner, text="\u2502",
                                       bg=BG_STATUS, fg=SASH, font=FONT_STATUS)
                        sep.pack(side="left", padx=2)
                        self._status_separators.append(sep)
            else:
                for key, text, fg, _ in items:
                    if key in self._status_labels:
                        self._status_labels[key].config(text=text, fg=fg)
        except Exception:
            pass
        finally:
            self.root.after(5000, self._refresh_status)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------
    def _refresh_research(self) -> None:
        try:
            candidates, total = self._state.get_scan_results()
            decisions = self._state.get_risk_decisions()
            decision_map = {d["ticker"]: d for d in decisions}

            _clear(self._research_text)

            if not candidates:
                _w(self._research_text, "\nWaiting for first scan\u2026\n", "subtle")
                return

            _w(self._research_text, f"{len(candidates)} candidates", "label")
            if total:
                _w(self._research_text, f"  /  {total} markets\n", "dim")
            else:
                _w(self._research_text, "\n")
            _w(self._research_text, "\n")

            for cand in candidates[:25]:
                ticker = cand.market.ticker
                title = _strip_md(cand.market.title or ticker)
                display = _trunc(title, 42)
                mid = cand.market.mid_price
                signals = ", ".join(s.split("(")[0] for s in cand.signals[:3])
                decision = decision_map.get(ticker)

                _w(self._research_text, f"{display}\n", "bold")
                _w(self._research_text, f"  ${mid:.2f}", "subtle")
                if signals:
                    _w(self._research_text, f"  {signals}", "dim")
                _w(self._research_text, "\n")

                if decision:
                    if decision["approved"]:
                        _w(self._research_text,
                           f"  \u2713 {decision['side'].upper()} "
                           f"edge {decision['edge']:+.3f}  "
                           f"${decision['size']:.2f}\n",
                           "bold_green")
                    else:
                        reason = _trunc(decision["reasoning"], 48)
                        _w(self._research_text, f"  \u2717 {reason}\n", "red")

                _w(self._research_text, "\n")

        except Exception:
            logger.debug("Research panel refresh error", exc_info=True)
        finally:
            self.root.after(10000, self._refresh_research)

    # ------------------------------------------------------------------
    # Positions (threaded API fetch)
    # ------------------------------------------------------------------
    def _refresh_positions(self) -> None:
        if self._pos_fetch_running:
            # Previous fetch still in flight — skip this cycle
            self.root.after(20000, self._refresh_positions)
            return
        self._pos_fetch_running = True
        threading.Thread(
            target=self._fetch_positions, daemon=True, name="pos-refresh"
        ).start()

    def _fetch_positions(self) -> None:
        try:
            open_trades = self._bot.db.get_open_trades()
        except Exception:
            self._pos_fetch_running = False
            self.root.after(20000, self._refresh_positions)
            return

        enriched: list[dict[str, Any]] = []
        for trade in open_trades:
            if trade.get("status") != "open":
                continue
            ticker = trade["ticker"]
            try:
                market = self._bot.kalshi.get_market(ticker)
                side = trade.get("side", "yes")
                if side == "yes":
                    live_price = market.yes_bid if market.yes_bid > 0 else market.last_price
                else:
                    live_price = market.no_bid if market.no_bid > 0 else (1.0 - market.last_price)
                trade["_live_price"] = live_price
                trade["_live_status"] = market.status
                trade["_title"] = _strip_md(market.title or ticker)
                trade["_subtitle"] = _strip_md(market.subtitle or "")
                trade["_event_ticker"] = market.event_ticker or ""
            except Exception:
                trade["_live_price"] = None
                trade["_live_status"] = "unknown"
                trade["_title"] = ticker
                trade["_subtitle"] = ""
                trade["_event_ticker"] = ""
            enriched.append(trade)

        # Group trades by event (market) so multiple positions in the same
        # event render together under one header, matching Kalshi's UI.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for trade in enriched:
            key = trade.get("_event_ticker") or trade["ticker"]
            grouped.setdefault(key, []).append(trade)

        self._pos_fetch_running = False
        self.root.after(0, self._render_positions, grouped)

    def _toggle_manual(self, trade_id: int) -> None:
        """Toggle manual flag for a trade and refresh."""
        try:
            new_val = self._bot.db.toggle_trade_manual(trade_id)
            label = "MANUAL" if new_val else "AUTO"
            logger.info("Trade #%d set to %s", trade_id, label)
            # Trigger immediate refresh
            self._refresh_positions()
        except Exception:
            logger.debug("Failed to toggle manual for trade #%d", trade_id, exc_info=True)

    def _render_positions(self, grouped: dict[str, list[dict[str, Any]]]) -> None:
        try:
            _clear(self._positions_text)
            pt = self._positions_text

            # Remove old toggle tag bindings
            for tag in list(pt.tag_names()):
                if tag.startswith("toggle_"):
                    pt.tag_unbind(tag, "<Button-1>")

            total_count = sum(len(v) for v in grouped.values())
            if total_count == 0:
                _w(pt, "\nNo open positions.\n", "subtle")
                self.root.after(20000, self._refresh_positions)
                return

            total_pnl = 0.0
            total_positions = 0

            for event_key, trades in grouped.items():
                first = trades[0]
                raw_title = first.get("_title", first["ticker"])
                market_title = _clean_title(raw_title)

                # ── Box top border ──────────────────────────────────
                _w(pt, "  \u250c" + "\u2500" * 80 + "\u2510\n", "card_border_top")

                # ── Market title row ────────────────────────────────
                _w(pt, f"  \u2502  {market_title}\n", "card_title")

                # ── Separator ───────────────────────────────────────
                _w(pt, "  \u251c" + "\u2500" * 80 + "\u2524\n", "cr_sep")

                for row_idx, trade in enumerate(trades):
                    tid = trade["id"]
                    subtitle = trade.get("_subtitle", "")
                    side = trade.get("side", "?").lower()
                    side_upper = side.upper()
                    entry = float(trade.get("entry_price", 0))
                    contracts = int(trade.get("size_contracts", 0) or 0)
                    live = trade.get("_live_price")
                    is_manual = bool(trade.get("manual"))

                    cost = round(entry * contracts, 2)
                    market_value = round(live * contracts, 2) if live is not None else cost
                    max_payout = round(1.0 * contracts, 2)

                    pnl = 0.0
                    pnl_pct = 0.0
                    if live is not None and entry > 0:
                        pnl = (live - entry) * contracts
                        pnl_pct = ((live - entry) / entry) * 100
                        total_pnl += pnl
                    total_positions += 1

                    up = pnl >= 0
                    now_pct = int(live * 100) if live is not None else 0
                    bought_pct = int(entry * 100)

                    # ── Line 1: Side · Subtitle  |  Bought  |  Now  |  Payout  |  Toggle
                    side_tag = "cr_side_yes" if side == "yes" else "cr_side_no"
                    _w(pt, f"  {side_upper}", side_tag)
                    sub_display = f" \u00b7 {subtitle}" if subtitle else ""
                    _w(pt, sub_display, "cr_sub")

                    col_used = len(side_upper) + len(sub_display) + 2
                    pad = max(2, 32 - col_used)
                    _w(pt, " " * pad, "cr")

                    _w(pt, "Bought ", "cr_lbl")
                    _w(pt, f"{bought_pct}%", "cr_val")

                    _w(pt, "     Now ", "cr_lbl")
                    now_tag = "cr_grn" if up else "cr_red"
                    arrow = " \u25b2" if (live is not None and live > entry) else (" \u25bc" if (live is not None and live < entry) else "")
                    _w(pt, f"{now_pct}%{arrow}", now_tag)

                    _w(pt, "     Payout ", "cr_lbl")
                    _w(pt, f"${max_payout:.0f}", "cr_payout")

                    # Clickable MANUAL/AUTO toggle
                    _w(pt, "   ", "cr")
                    toggle_tag = f"toggle_{tid}"
                    pt.tag_configure(
                        toggle_tag,
                        foreground=YELLOW if is_manual else GREEN_SOFT,
                        font=FONT_BOLD, underline=True, background=self._card_bg,
                    )
                    _w(pt, "MANUAL" if is_manual else "AUTO", toggle_tag)
                    pt.tag_bind(
                        toggle_tag, "<Button-1>",
                        lambda e, t=tid: self._toggle_manual(t),
                    )
                    _w(pt, "\n", "cr")

                    # ── Line 2: P&L  |  Cost  |  Mkt Value  |  Entry  |  Held
                    pnl_str = f"${pnl:+.2f}" if live is not None else "---"
                    pct_str = f" ({pnl_pct:+.1f}%)" if live is not None else ""
                    _w(pt, "  P&L ", "cr_lbl")
                    _w(pt, f"{pnl_str}{pct_str}", "cr_grn" if up else "cr_red")

                    _w(pt, "     Cost ", "cr_lbl")
                    _w(pt, f"${cost:.2f}", "cr_val")

                    _w(pt, "     Mkt Value ", "cr_lbl")
                    _w(pt, f"${market_value:.2f}", "cr_val")

                    _w(pt, "     Entry ", "cr_lbl")
                    _w(pt, f"${entry:.2f}", "cr_val")

                    held_str = ""
                    opened_at = trade.get("opened_at")
                    if opened_at:
                        try:
                            t_open = datetime.fromisoformat(str(opened_at).replace("Z", "+00:00"))
                            delta = datetime.now(timezone.utc) - t_open
                            hours = delta.total_seconds() / 3600
                            if hours < 1:
                                held_str = f"{int(delta.total_seconds() / 60)}m"
                            elif hours < 24:
                                held_str = f"{hours:.1f}h"
                            else:
                                held_str = f"{hours / 24:.1f}d"
                        except (ValueError, TypeError):
                            pass
                    if held_str:
                        _w(pt, f"     {held_str}", "cr_dim")

                    _w(pt, "\n", "cr")

                    # Separator between positions in the same card
                    if row_idx < len(trades) - 1:
                        _w(pt, "  \u251c" + "\u2500" * 80 + "\u2524\n", "cr_sep")

                # ── Box bottom border ───────────────────────────────
                _w(pt, "  \u2514" + "\u2500" * 80 + "\u2518\n", "card_border_bot")

            # ── Summary footer ──────────────────────────────────────
            _w(pt, "\n")
            total_tag = "bold_green" if total_pnl >= 0 else "bold_red"
            _w(pt, "  Unrealized  ", "label")
            _w(pt, f"${total_pnl:+.2f}", total_tag)
            _w(pt, f"      {total_positions} position{'s' if total_positions != 1 else ''}"
               f" in {len(grouped)} market{'s' if len(grouped) != 1 else ''}\n",
               "subtle")

        except Exception:
            logger.debug("Positions panel render error", exc_info=True)
        finally:
            self.root.after(20000, self._refresh_positions)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _refresh_postmortem(self) -> None:
        try:
            _clear(self._postmortem_text)

            try:
                h = self._bot.db.get_all_heuristics()
            except Exception:
                h = {}

            kelly = h.get("learned_kelly_fraction", "---")
            streak = h.get("loss_streak", "0")
            agg = h.get("order_aggression", "---")

            _w(self._postmortem_text, "Kelly ", "label")
            _w(self._postmortem_text, f"{kelly}", "mono")
            _w(self._postmortem_text, "   Streak ", "label")
            _w(self._postmortem_text, f"{streak}", "mono")
            _w(self._postmortem_text, "   Aggr ", "label")
            _w(self._postmortem_text, f"{agg}\n", "mono")

            # Category performance
            try:
                cat_stats = self._bot.db.get_category_stats()
            except Exception:
                cat_stats = {}

            if cat_stats:
                _w(self._postmortem_text, "\n")
                _w(self._postmortem_text, "Categories\n", "h2")
                for cat, stats in sorted(cat_stats.items()):
                    wins = stats.get("wins", 0) or 0
                    losses = stats.get("losses", 0) or 0
                    pnl = stats.get("total_pnl", 0) or 0
                    total = wins + losses
                    if total == 0:
                        continue
                    pnl_tag = "green" if pnl >= 0 else "red"
                    _w(self._postmortem_text, f" {cat[:14]:<14} ", "label")
                    _w(self._postmortem_text, f"{wins}W/{losses}L ", "subtle")
                    _w(self._postmortem_text, f"${pnl:+.2f}\n", pnl_tag)

            # Calibration
            cal_c = h.get("cal_model_correct", "0")
            cal_i = h.get("cal_model_incorrect", "0")
            try:
                cc, ci = int(cal_c), int(cal_i)
                ct = cc + ci
                if ct > 0:
                    _w(self._postmortem_text, "\n")
                    _w(self._postmortem_text, "Calibration\n", "h2")
                    _w(self._postmortem_text,
                       f" {cc}/{ct} correct ({cc / ct * 100:.0f}%)\n", "mono")
            except (ValueError, TypeError):
                pass

            # Postmortems
            try:
                pms = self._bot.db.get_recent_postmortems_with_trades(limit=5)
            except Exception:
                pms = []

            if pms:
                _w(self._postmortem_text, "\n")
                _w(self._postmortem_text, "Postmortems\n", "h2")
                for pm in pms:
                    ticker = pm.get("ticker", "?")
                    short = _trunc(ticker, 22)
                    reason = pm.get("failure_reason", "")
                    if len(reason) > 38:
                        reason = reason[:35] + "\u2026"
                    agreement = pm.get("edge_agreement")
                    tag = ""
                    if agreement is True:
                        tag = "unlucky"
                    elif agreement is False:
                        tag = "bad read"

                    _w(self._postmortem_text, f" {short}", "bold")
                    if tag:
                        _w(self._postmortem_text, f"  {tag}",
                           "yellow" if tag == "bad read" else "cyan")
                    _w(self._postmortem_text, "\n")
                    if reason:
                        _w(self._postmortem_text, f"   {reason}\n", "dim")

        except Exception:
            logger.debug("Postmortem panel refresh error", exc_info=True)
        finally:
            self.root.after(30000, self._refresh_postmortem)

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    def _refresh_log(self) -> None:
        try:
            # Drain new lines since last check using a monotonic sequence id
            # to avoid the deque-rollover desync bug.
            all_lines = self._state.get_log_lines(500)
            total_len = len(all_lines)

            if self._log_line_count > total_len:
                # Deque rolled over — reset and show tail
                self._log_line_count = max(0, total_len - 50)

            new_lines = all_lines[self._log_line_count:]
            self._log_line_count = total_len

            if new_lines:
                self._log_text.config(state="normal")
                # Batch insert for efficiency
                chunk = "\n".join(new_lines) + "\n"
                self._log_text.insert("end", chunk)
                # Trim to 300 lines max to keep widget responsive
                n = int(self._log_text.index("end-1c").split(".")[0])
                if n > 300:
                    self._log_text.delete("1.0", f"{n - 300}.0")
                self._log_text.see("end")
                self._log_text.config(state="disabled")
        except Exception:
            pass
        finally:
            self.root.after(3000, self._refresh_log)

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()
