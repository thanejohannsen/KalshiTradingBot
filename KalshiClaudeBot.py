"""
KalshiClaudeBot — Multi-agent prediction market trading bot.

Entry point.  All logic lives in main.py and the agents/ directory.
Run this file to start the bot:

    python KalshiClaudeBot.py

Configuration is loaded from .env (copy .env.example to get started).
"""

from main import main

if __name__ == "__main__":
    main()
