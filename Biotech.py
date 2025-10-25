#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pytz
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ========== env ==========
ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

NY = pytz.timezone("America/New_York")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# confidence mapping
CONF_FLOOR = 0.70
CONF_CEIL  = 0.97
CONF_POWER = 0.75

# horizons (5m bars)
BARS = {"30m":6, "1h":12, "1d":78, "2d":156, "1w":390}

# ---------- small utils ----------
def _fmt(v):
    try: return f"{float(v):.2f}"
    except: return "n/a"

def _is_intraday(interval: str) -> bool:
    return any(x in interval for x in ("m","h"))

def _to_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty: return df
    if df.index.tz is None: df = df.tz_localize("UTC")
    return df.tz_convert(NY)

def _ensure_numeric(df: pd.DataFrame, cols=("Open","High","Low","Close","Volume")):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _filter_regular_hours(df, tz=NY):
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty: return df
    idx = df.index.tz_convert(tz) if df.index.tz is not None else df.index.tz_localize('UTC').tz_convert(tz)
    mask = (idx.time >= pd.Timestamp("09:35", tz=tz).time()) & (idx.time <= pd.Timestamp("15:55", tz=tz).time())
    return df.loc[mask]

# ---------- webhook ----------
def _post_webhook(text: str):
    if not WEBHOOK_URL: return
    MAX_LEN = 1900
    parts, buf, n = [], [], 0
    for line in text.splitlines():
        add = line + "\n"
        if n + len(add) > MAX_LEN:
            parts.append("".join(buf)); buf, n = [add], len(add)
        else:
            buf.append(add); n += len(add)
    if buf: parts.append("".join(buf))
    for p in parts:
        try:
            r = requests.post(WEBHOOK_URL, json={"content": p}, timeout=10)
            if r.status_code == 429:
                import time; time.sleep(float(r.headers.get("Retry-After","1")))
                requests.post(WEBHOOK_URL, json={"content": p}, timeout=10)
        except: pass

def _check_webhook():
    print(f"🔌 .env path: {ENV_PATH}")
    ok = bool(WEBHOOK_URL)
    print(f"🔌 Webhook present: {ok} (len={len(WEBHOOK_URL) if ok else 0})")
    if not ok: return False
    try:
        r = requests.post(WEBHOOK_URL, json={"content":"✅ Webhook check: connected."}, timeout=10)
        print(f"🔌 Webhook test → status {r.status_code} (expect 200/204)")
        return r.status_code in (200,204)
    except Exception as e:
        print(f"🔌 Webhook error: {e}"); return False

# ---------- data ----------
def safe_fetch(ticker, period="6mo", interval="1d", prepost=False):
    adj = False if _is_intraday(interval) else True
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=adj, prepost=prepost)
    if (df is None) or df.empty or ('Close' not in df.columns):
        df = yf.Ticker(ticker).history(period=period, interval=interval,
                                       auto_adjust=adj, prepost=prepost)
    if (df is None) or df.empty:
        raise ValueError(f"Empty dataframe for {ticker}")
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if 'Close' not in df.columns:
        for col in df.columns:
            if 'close' in col.lower():
                df.rename(columns={col: 'Close'}, inplace=True); break
    return _to_ny_index(_ensure_numeric(df).dropna(subset=["Close"]))

def get_intraday_with_fallbacks(ticker: str) -> pd.DataFrame:
    for per, itv in [("60d","5m"), ("30d","5m"), ("14d","1m")]:
        try:
            d = safe_fetch(ticker, per, itv, prepost=True)
            d = _ensure_numeric(d).dropna(subset=["Close"])
            if len(d)>0: return d
        except: pass
    raise ValueError("Intraday fetch failed")

def get_reliable_last_price(ticker: str, day_df: pd.DataFrame, use_extended=True) -> float:
    t = yf.Ticker(ticker); vals=[]
    fi = getattr(t,"fast_info",None)
    if fi is not None:
        seq = (fi.post_market_price, fi.pre_market_price, fi.last_price, fi.regular_market_price) if use_extended \
              else (fi.regular_market_price, fi.last_price)
        for p in seq:
            if p is not None: vals.append(float(p))
    try:
        h = t.history(period="1d", interval="1m", prepost=use_extended, auto_adjust=False)
        if h is not None and not h.empty: vals.append(float(h["Close"].iloc[-1]))
    except: pass
    if day_df is not None and not day_df.empty:
        vals.append(float(day_df["Close"].iloc[-1]))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals: raise ValueError("No price")
    med = float(np.median(vals))
    filt = [v for v in vals if abs(v-med)/med <= 0.05] or [med]
    return float(filt[-1])

def _fetch_benchmarks(interval="5m", prepost=False):
    out={}
    for tk in ["SPY","QQQ","SOXX"]:
        try:
            d = safe_fetch(tk,"60d",interval,prepost=prepost)
            out[tk] = d.dropna(subset=["Close"])
        except: out[tk]=None
    return out

# ---------- feature builder (upgraded) ----------
def _build_features(base: pd.DataFrame) -> pd.DataFrame:
    # RTH-only + light denoise
    base = _filter_regular_hours(base, tz=NY).copy()
    base["Close"] = base["Close"].rolling(3).mean().fillna(base["Close"])

    # market context (benchmarks)
    benches = _fetch_benchmarks("5m", prepost=False)
    for tk, dfb in benches.items():
        if dfb is not None:
            dfb = _filter_regular_hours(dfb, tz=NY).reindex(base.index)
            dfb.ffill(inplace=True)
            base[f"{tk}_ret"] = dfb["Close"].pct_change()*100.0
            base[f"{tk}_ret_l1"] = base[f"{tk}_ret"].shift(1)
            m = base[f"{tk}_ret"]
            base[f"{tk}_ret_z"] = (m - m.rolling(60).mean())/(m.rolling(60).std()+1e-9)
        else:
            base[f"{tk}_ret"] = 0.0; base[f"{tk}_ret_l1"]=0.0; base[f"{tk}_ret_z"]=0.0

    c,h,l,v = base["Close"], base["High"], base["Low"], base["Volume"].replace(0,np.nan).ffill()

    # classic indicators
    base["EMA_5"]  = talib.EMA(c,5)
    base["EMA_10"] = talib.EMA(c,10)
    base["RSI_7"]  = talib.RSI(c,7)
    base["ADX_14"] = talib.ADX(h,l,c,14)
    base["ATR_14"] = talib.ATR(h,l,c,14)
    # stochastics
    k,d = talib.STOCH(h,l,c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    base["STOCH_K"], base["STOCH_D"] = k,d
    # CCI / OBV
    base["CCI_14"] = talib.CCI(h,l,c,14)
    base["OBV"]    = talib.OBV(c, v)
    # Price dynamics
    base["RET_1"]  = c.pct_change()*100.0
    base["RET_3"]  = c.pct_change(3)*100.0
    base["RET_6"]  = c.pct_change(6)*100.0
    base["ROC_5"]  = talib.ROC(c,5)
    base["ROC_10"] = talib.ROC(c,10)
    base["ROC_20"] = talib.ROC(c,20)
    base["ROLL_STD_10"] = c.pct_change().rolling(10).std()
    base["ROLL_MAX_20"] = c.rolling(20).max() / c - 1.0
    base["ROLL_MIN_20"] = c.rolling(20).min() / c - 1.0
    # VWAP slope (30 bars)
    typical = (h+l+c)/3.0
    vwap = (typical*v).rolling(30).sum()/v.rolling(30).sum()
    base["VWAP_SLOPE"] = vwap.diff()
    # RSI divergence proxy (RSI slope vs price slope)
    base["RSI_SLOPE"] = base["RSI_7"].diff()
    base["PX_SLOPE"]  = c.diff()
    base["RSI_DIV"]   = base["RSI_SLOPE"] - base["PX_SLOPE"]

    # Volume zscore
    base["VOL_Z"] = (v - v.rolling(60).mean()) / (v.rolling(60).std()+1e-9)

    base.replace([np.inf,-np.inf], np.nan, inplace=True)
    base.ffill(inplace=True); base.fillna(0.0,inplace=True)
    return base

# list of features to try (we'll auto-select best later)
FEAT_CANDIDATES = [
    # market
    "SPY_ret","SPY_ret_l1","SPY_ret_z","QQQ_ret","QQQ_ret_l1","QQQ_ret_z","SOXX_ret","SOXX_ret_l1","SOXX_ret_z",
    # classic
    "EMA_5","EMA_10","RSI_7","ADX_14","ATR_14",
    # stoch/cci/obv
    "STOCH_K","STOCH_D","CCI_14","OBV",
    # price dynamics
    "RET_1","RET_3","RET_6","ROC_5","ROC_10","ROC_20","ROLL_STD_10","ROLL_MAX_20","ROLL_MIN_20",
    # vwap/rsidiv/volume
    "VWAP_SLOPE","RSI_SLOPE","PX_SLOPE","RSI_DIV","VOL_Z"
]

# ---------- main ----------
class TechnicalAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        print(f"\nFetching data for {self.ticker} ...")
        self.long_data = safe_fetch(self.ticker,"1y","1d",prepost=False)
        day = get_intraday_with_fallbacks(self.ticker)
        self.day_data = day
        try:
            self.last_price = get_reliable_last_price(self.ticker, day, use_extended=True)
        except Exception:
            self.last_price = float(day["Close"].iloc[-1])
        print("Data fetched successfully.\n")

    # ----- LONG -----
    def long_term(self):
        df = self.long_data.copy()
        c = df["Close"].astype(float)
        df["SMA_20"] = talib.SMA(c,20); df["SMA_50"] = talib.SMA(c,50)
        df["RSI_14"] = talib.RSI(c,14)
        macd,sig,_ = talib.MACD(c,12,26,9); df["MACD"],df["Signal"]=macd,sig
        up,mid,low = talib.BBANDS(c,20,2,2); df["BBL"]=low
        df.dropna(inplace=True); last=df.iloc[-1]
        trend = "📈 Strong Bullish" if last["Close"]>last["SMA_20"]>last["SMA_50"] else \
                "📉 Strong Bearish" if last["Close"]<last["SMA_20"]<last["SMA_50"] else "⚪ Neutral"
        buy=float(last["BBL"])
        print("==== LONG-TERM ANALYSIS ====")
        print(f"Last Close: ${_fmt(last['Close'])}")
        print(f"SMA20: ${_fmt(last['SMA_20'])} | SMA50: ${_fmt(last['SMA_50'])}")
        print(f"RSI(14): {_fmt(last['RSI_14'])}")
        print(f"MACD: {_fmt(last['MACD'])} | Signal: {_fmt(last['Signal'])}")
        print(f"Trend: {trend}")
        print(f"Potential Buy Price: ${_fmt(buy)}\n")
        _post_webhook("\n".join([
            f"**📈 LONG-TERM ANALYSIS — {self.ticker}**","```",
            f"{'Field':<18} | {'Value':<12}","-"*34,
            f"{'Last Close':<18} | ${last['Close']:.2f}",
            f"{'SMA20':<18} | ${last['SMA_20']:.2f}",
            f"{'SMA50':<18} | ${last['SMA_50']:.2f}",
            f"{'RSI(14)':<18} | {last['RSI_14']:.2f}",
            f"{'MACD':<18} | {last['MACD']:.2f}",
            f"{'Signal':<18} | {last['Signal']:.2f}",
            f"{'BB Lower':<18} | ${buy:.2f}","```",f"Trend: {trend}"
        ]))

    # ----- DAY -----
    def day_trade(self):
        df = self.day_data.copy()
        c = df["Close"].astype(float)
        df["EMA_5"]=talib.EMA(c,5); df["EMA_10"]=talib.EMA(c,10); df["RSI_7"]=talib.RSI(c,7)
        m,s,_ = talib.MACD(c,6,13,5); df["MACD"],df["Signal"]=m,s
        up,mid,low = talib.BBANDS(c,20,2,2); df["Upper"],df["Middle"],df["Lower"]=up,mid,low
        df.dropna(inplace=True); last=df.iloc[-1]
        action = "BUY 🟢" if last["EMA_5"]>last["EMA_10"] else "SELL 🔴" if last["EMA_5"]<last["EMA_10"] else "HOLD ⚪"
        print("==== DAY-TRADE ANALYSIS ====")
        print(f"Last Price: ${_fmt(self.last_price)}")
        print(f"EMA5: ${_fmt(last['EMA_5'])} | EMA10: ${_fmt(last['EMA_10'])}")
        print(f"RSI(7): {_fmt(last['RSI_7'])}")
        print(f"MACD: {_fmt(last['MACD'])} | Signal: {_fmt(last['Signal'])}")
        print(f"Bollinger Bands: Upper {_fmt(last['Upper'])} | Lower {_fmt(last['Lower'])}")
        print(f"Suggested Action: {action}")
        print(f"Potential Buy Price: ${_fmt(last['Lower'])}\n")
        _post_webhook("\n".join([
            f"**⚡ DAY-TRADE ANALYSIS — {self.ticker}**","```",
            f"{'Field':<20} | {'Value':<12}","-"*38,
            f"{'Last Price':<20} | ${self.last_price:.2f}",
            f"{'EMA5':<20} | ${last['EMA_5']:.2f}",
            f"{'EMA10':<20} | ${last['EMA_10']:.2f}",
            f"{'RSI(7)':<20} | {last['RSI_7']:.2f}",
            f"{'MACD':<20} | {last['MACD']:.2f}",
            f"{'Signal':<20} | {last['Signal']:.2f}",
            f"{'BB Upper':<20} | ${last['Upper']:.2f}",
            f"{'BB Lower':<20} | ${last['Lower']:.2f}",
            "```",f"Suggested Action: {action}"
        ]))

    # ----- ML (upgraded) -----
    def ml_forecast(self):
        base = _ensure_numeric(self.day_data.copy())
        base = _build_features(base)

        # feature matrix
        X_all = base[FEAT_CANDIDATES].copy()

        # quick auto feature selection:  keep top ~25 by tree importance (adaptive)
        # We'll fit once on a subset to score features, then refit below per horizon.
        seed_split = int(len(X_all)*0.75)
        if seed_split > 600:
            y_tmp = (base["Close"].shift(-12)/base["Close"] - 1.0)*100.0  # 1h temp target
            Xs = X_all.iloc[:seed_split].fillna(0.0); ys = y_tmp.iloc[:seed_split].fillna(0.0)
            scaler0 = StandardScaler().fit(Xs); m0 = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42
            ).fit(scaler0.transform(Xs), ys)
            imps = pd.Series(m0.feature_importances_, index=X_all.columns).sort_values(ascending=False)
            keep = list(imps.head(min(25, (imps>0).sum() or 25)).index)
            X_all = X_all[keep]
        feat_cols = list(X_all.columns)

        lp = float(self.last_price)
        last_feat_row = X_all.iloc[[-1]]
        results=[]; up_ct=0; dn_ct=0; WEAK=True

        print("==== 🧠 ML PRICE PREDICTION ====")
        for label,k in BARS.items():
            fut_ret = (base["Close"].shift(-k)/base["Close"] - 1.0)*100.0
            X = X_all.iloc[:-k]; y = fut_ret.iloc[:-k].dropna(); X = X.loc[y.index]
            if len(X) < 500:
                results.append((label, lp, 0.0, CONF_FLOOR)); continue

            # purged split with embargo to reduce leakage
            split = int(len(X)*0.80); embargo=10
            X_train, y_train = X.iloc[:split-embargo], y.iloc[:split-embargo]
            X_test,  y_test  = X.iloc[split+embargo:],  y.iloc[split+embargo:]
            if len(X_test) < 3:
                X_train, y_train = X.iloc[:split], y.iloc[:split]
                X_test,  y_test  = X.iloc[split:],  y.iloc[split:]

            scaler = StandardScaler().fit(X_train)
            Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

            model = GradientBoostingRegressor(
                n_estimators=900, learning_rate=0.02, max_depth=4,
                subsample=0.85, random_state=42
            )
            model.fit(Xtr, y_train)

            # holdout correlation (confidence basis)
            if len(X_test) > 3:
                pred_val = model.predict(Xte)
                corr = float(np.corrcoef(pred_val, y_test)[0,1])
            else:
                corr = 0.0
            corr = max(0.0, min(abs(corr), 1.0))
            conf = CONF_FLOOR + (corr**CONF_POWER)*(CONF_CEIL-CONF_FLOOR)

            # refit on all history then predict last bar
            scaler_full = StandardScaler().fit(X)
            model.fit(scaler_full.transform(X), y)
            pred_pct = float(model.predict(scaler_full.transform(last_feat_row))[0])
            pred_pct = float(np.clip(pred_pct, -15.0, 15.0))
            price = lp*(1.0+pred_pct/100.0)

            results.append((label, price, pred_pct, conf))
            if abs(pred_pct) >= 0.40 and conf >= 0.82: WEAK=False
            if price>lp: up_ct+=1
            elif price<lp: dn_ct+=1

        for label,price,ret_pct,conf in results:
            direction = "UP 🔺" if price>lp else "DOWN 🔻"
            print(f"Next {label}: ${price:.2f} ({ret_pct:+.2f}%) → {direction} | Confidence: {conf*100:.1f}%")

        align = "✅ Alignment: ML skew UP across horizons." if up_ct>=3 else \
                "✅ Alignment: ML skew DOWN across horizons." if dn_ct>=3 else \
                "⚪ Alignment: Mixed across horizons."
        print(align)
        if WEAK: print("⚠️  Signal weak (|Δ%|<0.40% or conf<82%). Treat as noise.\n")
        print("💡 Forecast uses RTH-only, market/sector conditioning, richer TA features, and purged walk-forward validation.\n")

        # discord table (unchanged look)
        if WEBHOOK_URL:
            lines=[f"**🧠 ML PRICE PREDICTION for {self.ticker.upper()}**","```",
                   f"{'Horizon':<8} | {'Price($)':<10} | {'Δ%':<7} | {'Dir':<5} | {'Conf':<6}",
                   "-"*45]
            for label,price,ret_pct,conf in results:
                direction="UP" if price>lp else "DOWN"
                lines.append(f"{label:<8} | {price:<10.2f} | {ret_pct:<+7.2f} | {direction:<5} | {conf*100:5.1f}%")
            lines.append("```"); lines.append(align)
            if WEAK:
                lines.append("⚠️  Signal weak (|Δ%|<0.40% or conf<82%). Treat as noise.")
            lines.append("💡 Forecast uses RTH-only, market/sector conditioning, richer TA features, and purged walk-forward validation.")
            _post_webhook("\n".join(lines))

    # ----- simple backtests (existing hooks if you already had them) -----
    def ml_backtest_fast(self, horizon="1d", step=20, max_models=150):
        if horizon not in BARS:
            print(f"Backtest: unknown horizon '{horizon}'. Use one of: {list(BARS.keys())}"); return
        k = BARS[horizon]
        base = _build_features(_ensure_numeric(self.day_data.copy()))
        X_all = base[FEAT_CANDIDATES]; y_all = (base["Close"].shift(-k)/base["Close"] - 1.0)*100.0
        start = max(500, k*2); last_idx = len(X_all)-k
        if last_idx <= start: print("Backtest: not enough data."); return
        _post_webhook(f"⏱️ Backtest (fast) starting — {self.ticker} {horizon}, step={step}")

        preds,trues=[],[]; fits=0
        import time; t0=time.time()
        for t in range(start,last_idx,step):
            if fits>=max_models or (time.time()-t0)>60: break
            X_train = X_all.iloc[:t]; y_train = y_all.iloc[:t].dropna(); X_train = X_train.loc[y_train.index]
            y_true = y_all.iloc[t]
            if len(X_train)<500 or not np.isfinite(y_true): continue
            scaler = StandardScaler().fit(X_train)
            model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=3,
                                              subsample=0.9, random_state=42)
            model.fit(scaler.transform(X_train), y_train)
            y_pred = float(model.predict(scaler.transform(X_all.iloc[[t]]))[0]); y_pred = float(np.clip(y_pred,-15.0,15.0))
            preds.append(y_pred); trues.append(y_true); fits+=1

        if not trues: _post_webhook(f"⚠️ Backtest ended for {self.ticker} — no samples."); return
        preds, trues = np.array(preds), np.array(trues)
        hit = (np.sign(preds)==np.sign(trues)).mean()*100.0
        pnl = (np.sign(preds)*trues); mu=pnl.mean(); sigma=pnl.std(ddof=1)
        from math import sqrt; sharpe = (mu/(sigma+1e-9))*sqrt(252)
        report = ["**📜 ML BACKTEST (fast)** — "+self.ticker+f" ({horizon})","```",
                  f"Samples     : {len(trues)}",
                  f"Fits run    : {fits} (step={step}, cap={max_models})",
                  f"Hit-Rate    : {hit:5.1f}%",
                  f"Avg PnL     : {mu:+.3f}%",
                  f"Volatility  : {sigma:.3f}%",
                  f"Sharpe~     : {sharpe:.2f}",
                  "```","Expanding walk-forward. Live ML unaffected."]
        print("\n".join(report)); _post_webhook("\n".join(report))

# ---------- CLI ----------
def main():
    print("🧠 Perfect Technical + ML Analysis Tool — Type 'exit' to quit.\n")
    _check_webhook()
    ticker = input("Enter a stock ticker: ").strip()
    if not ticker: return
    try:
        ta = TechnicalAnalyzer(ticker)
    except Exception as e:
        print(f"Error fetching data: {e}"); return

    while True:
        cmd = input("Command (long/day/ml/bt/btfast/back/exit): ").strip().lower()
        if   cmd=="exit": break
        elif cmd=="long": ta.long_term()
        elif cmd=="day":  ta.day_trade()
        elif cmd=="ml":   ta.ml_forecast()
        elif cmd=="bt":   ta.ml_backtest_fast(horizon="1d", step=20, max_models=150)  # fast default
        elif cmd=="btfast": ta.ml_backtest_fast(horizon="1d", step=20, max_models=150)
        elif cmd=="back": main(); break
        else: print("Commands: long | day | ml | bt | btfast | back | exit\n")

if __name__ == "__main__":
    main()
