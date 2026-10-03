# Predator: Algorithmic Crypto Trading Bot (Bybit)

Advanced algorithmic trading bot for Bybit API, built with Python. Designed for automated execution using multi-timeframe analysis and order book density tracking.

## Core Features
* **Multi-Strategy Engine**: Integrates TrendPRO, InstitutionalSMCPRO (False Breakouts), and MomentumBreakoutPRO algorithms.
* **Advanced Risk Management**: Dynamic R-based trailing stops, strict margin usage ceilings, and cost-guard protection (spread and fee awareness).
* **Smart Execution**: PostOnly limit orders for maker fee optimization, auto-cancellation of stale orders, and true break-even calculation.
* **Live Telemetry & Alerts**: Real-time Telegram API integration for execution alerts, PnL tracking, and structural exit notifications.

*Note: API keys and sensitive environment variables are managed securely via a local secrets file and are excluded from this public repository.*
