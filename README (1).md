# Biotech Pro

A Python tool that analyzes a stock ticker with technical indicators and a machine learning price forecast. It runs three ways: a command line app, a desktop app, and a web dashboard. Results can also be posted to a Discord channel.

Built October 2025.

> This is a personal learning project. It is not financial advice.

## What it does

Type in a ticker (for example `NVDA`) and run one of these:

| Command | What you get |
|---|---|
| **LONG** | Long-term trend from 1 year of daily data. Shows SMA 20/50, RSI, MACD, and a Bollinger Band based "potential buy price." |
| **DAY** | Day-trade snapshot from intraday data. Shows EMA 5/10, RSI, MACD, Bollinger Bands, and a BUY / SELL / HOLD suggestion. |
| **ML** | Price forecasts for 30 minutes, 1 hour, 1 day, 2 days, and 1 week, each with a direction, percent change, and confidence score. |
| **BT FAST** | A quick walk-forward backtest of the ML model. Reports samples, hit rate, average return, volatility, and an approximate Sharpe ratio. |

## How the ML forecast works

- **Data:** price history from Yahoo Finance (`yfinance`), 5 minute bars with fallbacks to other intervals. Only regular market hours are used (9:35 AM to 3:55 PM ET).
- **Market context:** returns from SPY, QQQ, and SOXX are added as features so the model sees what the broader market is doing.
- **Features:** about 30 candidates built with TA-Lib and pandas (EMA, RSI, ADX, ATR, Stochastic, CCI, OBV, rate of change, rolling volatility, VWAP slope, RSI divergence, volume z-score). The top 25 by importance are kept.
- **Model:** one `GradientBoostingRegressor` (scikit-learn) per forecast horizon, predicting the percent change.
- **Validation:** a time ordered train/test split with an embargo gap between them, to reduce leakage from overlapping bars.
- **Confidence score:** based on how well predictions correlate with real returns on held-out data, then scaled to a 70% to 97% range. It is a rough quality signal, **not** a probability that the forecast is right.
- **Weak signal warning:** if the predicted move is under 0.40% or confidence is under 82%, the app flags it as noise.

## Ways to run it

- **Command line:** `python Biotech.py`
- **Desktop app:** `python Biotech_GUI.py` (built with CustomTkinter, dark and light themes, forecast table and confidence bar). `Biotech Pro.spec` packages it into a Mac app with PyInstaller.
- **Web dashboard:** `streamlit run Bioetch_Dashboard.py` (Streamlit version with the same buttons, tables, and confidence bar)

The GUI and dashboard both call the same engine in `Biotech.py`.

## Setup

1. Install Python 3.10 or newer.
2. Install the TA-Lib C library first (for example `brew install ta-lib` on Mac). The Python `TA-Lib` package needs it.
3. Install the packages:

```bash
pip install -r requirements.txt
pip install TA-Lib customtkinter pytz
```

4. Optional, for Discord posts: copy `.env.template` to `.env` and add your webhook:

```
DISCORD_WEBHOOK_URL=your_webhook_url_here
```

Never commit your real `.env` file or any tokens.

## Tech stack

Python, pandas, NumPy, scikit-learn, yfinance, TA-Lib, Streamlit, CustomTkinter, PyInstaller, Discord webhooks

## Project files

| File | Purpose |
|---|---|
| `Biotech.py` | The analysis and ML engine, plus the command line app |
| `Biotech_GUI.py` | Desktop app |
| `Bioetch_Dashboard.py` | Streamlit web dashboard |
| `Biotech Pro.spec` | PyInstaller build settings |
| `requirements.txt` | Python packages |
| `.env.template` | Example environment file |

## What I'd improve next

- Add a screenshot of the app to this README.
- Add unit tests for the feature builder and the output parser.
- Add GitHub Actions to run the tests on every push.
- Run it in Docker so setup does not depend on the TA-Lib install.
- Store forecasts in a database so past predictions can be checked against what happened.
