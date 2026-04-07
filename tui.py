"""
Entry point for the Kalshi Trading Bot GUI.

Usage:
    python tui.py
"""

from __future__ import annotations

import logging
import threading

from config import Config
from main import TradingBot, set_shutdown
from tui_state import BotState, TUILogHandler
from tui_app import TradingBotGUI
from utils import setup_logging


def main() -> None:
    cfg = Config()

    # ── Logging setup ───────────────────────────────────────────────
    root_logger = setup_logging(
        level=cfg.log_level,
        log_file=cfg.log_file,
        quiet_research_console=cfg.quiet_research_console,
    )

    # Mute the console handler — logs go to the GUI log panel instead
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(logging.CRITICAL + 1)

    # Wire GUI log handler (INFO+ into the log panel)
    state = BotState()
    tui_handler = TUILogHandler(state)
    tui_handler.setLevel(logging.INFO)
    tui_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(tui_handler)

    # ── Bot ─────────────────────────────────────────────────────────
    bot = TradingBot(cfg, state=state)

    def _run_bot() -> None:
        try:
            bot.run_loop()
        except Exception:
            logging.getLogger("kalshi_bot.tui").exception("Bot thread crashed")

    bot_thread = threading.Thread(target=_run_bot, name="bot-main", daemon=True)
    bot_thread.start()

    # ── GUI ─────────────────────────────────────────────────────────
    app = TradingBotGUI(bot, state)
    app.run()

    # ── Shutdown ────────────────────────────────────────────────────
    set_shutdown()
    bot_thread.join(timeout=10)


if __name__ == "__main__":
    main()
