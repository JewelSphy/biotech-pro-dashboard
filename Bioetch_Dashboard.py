# Biotech_Dashboard.py
# Streamlit web dashboard wrapping your existing Biotech.py engine.
# Runs locally (or can be deployed). Presents neat tables + confidence gauge.
# Keeps your working logic: imports TechnicalAnalyzer and _post_webhook.

import io
import ta
import re
import time
from contextlib import redirect_stdout
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import your engine (DO NOT modify Biotech.py)
from Biotech import TechnicalAnalyzer, _post_webhook

# ---- setup ----
st.set_page_config(page_title="Biotech Pro — TA + ML", page_icon="🧠", layout="wide")
load_dotenv(dotenv_path=Path(".env"), override=True)

PRIMARY = "#2563eb"
GREEN = "#16a34a"
YELLOW = "#f59e0b"
RED = "#ef4444"
MUTED = "#93a1b3"

# ---- utils ----
def capture_stdout(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()

def parse_ml_output(txt: str):
    """
    Parse your console ML lines into a structured table + notes.
    Expected lines like:
    Next 1h: $123.45 (+0.67%) → UP 🔺 | Confidence: 86.2%
    """
    rows = []
    confmax = 0.0
    pat = re.compile(
        r"Next\s+([0-9a-zA-Z]+):\s*\$([0-9]+\.[0-9]+)\s*\(([+\-]?[0-9]+\.[0-9]+)%\)"
        r"\s*→\s*(UP|DOWN).+?Confidence:\s*([0-9]+\.[0-9])%",
        re.IGNORECASE
    )
    for m in pat.finditer(txt):
        horizon = m.group(1)
        price   = float(m.group(2))
        pct     = float(m.group(3))
        dirn    = m.group(4).upper()
        conf    = float(m.group(5))
        rows.append({
            "Horizon": horizon,
            "Price ($)": f"{price:.2f}",
            "Δ%": f"{pct:+.2f}%",
            "Direction": dirn,
            "Confidence": f"{conf:.1f}%"
        })
        confmax = max(confmax, conf)

    align, weak = "", ""
    for line in txt.splitlines():
        if "Alignment:" in line:
            align = line.strip().replace("✅ ", "").replace("⚪ ", "")
        if "Signal weak" in line:
            weak = "Signal weak (|Δ%|<0.40% or conf<82%)."
    return pd.DataFrame(rows), align, weak, confmax/100.0

def color_badge(text, color):
    return f"""<span style="
        background:{color};color:white;padding:6px 10px;border-radius:14px;
        font-weight:600;font-size:12px;">{text}</span>"""

def conf_bar_html(v: float):
    v = max(0.0, min(1.0, float(v)))
    pct = int(round(v*100))
    color = RED if v < 0.82 else YELLOW if v < 0.87 else GREEN
    return f"""
    <div style="width:100%;background:#0f172a;border-radius:10px;height:16px;">
        <div style="width:{pct}%;background:{color};height:16px;border-radius:10px;"></div>
    </div>
    <div style="text-align:right;color:#e5e7eb;margin-top:4px;font-size:12px;">{pct}%</div>
    """

# ---- sidebar ----
with st.sidebar:
    st.title("🧠 Biotech Pro")
    st.caption("TA + ML web dashboard (uses your Biotech.py backend)")
    ticker = st.text_input("Ticker", value="NVDA").upper().strip()
    post_to_discord = st.checkbox("Post to Discord", value=True)
    st.divider()
    run_all = st.button("Run All (LONG → DAY → ML)", use_container_width=True, type="primary")

# ---- header ----
st.markdown(
    f"""
    <h2 style="margin-bottom:0">Ticker: <span style="color:{PRIMARY}">{ticker or '—'}</span></h2>
    <div style="color:{MUTED};margin-bottom:14px;">Clean web UI wrapping your existing engine.</div>
    """,
    unsafe_allow_html=True
)

colA, colB, colC, colD = st.columns([1,1,1,1])

# storage for sections
if "last_conf" not in st.session_state:
    st.session_state.last_conf = 0.0

# ---- actions ----
def do_long():
    if not ticker: return None
    ta = TechnicalAnalyzer(ticker)
    out = capture_stdout(ta.long_term)
    # extract friendly lines
    trend, buy = "Trend: —", ""
    for ln in out.splitlines():
        if "Trend:" in ln: trend = ln.strip()
        if "Potential Buy Price:" in ln: buy = ln.strip()
    if post_to_discord:
        _post_webhook(f"**Web UI** • LONG for `{ticker}`")
    return trend.replace("Trend:", "").strip(), buy

def do_day():
    if not ticker: return None
    ta = TechnicalAnalyzer(ticker)
    out = capture_stdout(ta.day_trade)
    action, bb = "", ""
    for ln in out.splitlines():
        if "Suggested Action:" in ln: action = ln.strip()
        if "Bollinger Bands" in ln or "BB Upper" in ln: bb = ln.strip()
    if post_to_discord:
        _post_webhook(f"**Web UI** • DAY for `{ticker}`")
    return action.replace("Suggested Action: ","").strip(), bb

def do_ml():
    if not ticker: return None
    ta = TechnicalAnalyzer(ticker)
    out = capture_stdout(ta.ml_forecast)
    df, align, weak, confmax = parse_ml_output(out)
    st.session_state.last_conf = confmax
    if post_to_discord:
        _post_webhook(f"**Web UI** • ML for `{ticker}`")
    return df, align, weak, confmax, out

def do_btfast():
    if not ticker: return None
    ta = TechnicalAnalyzer(ticker)
    out = capture_stdout(lambda: ta.ml_backtest_fast(horizon="1d", step=20, max_models=150))
    if post_to_discord:
        _post_webhook(f"**Web UI** • BTFAST for `{ticker}`")
    return out

# ---- top buttons row ----
with colA:
    if st.button("LONG", use_container_width=True):
        res = do_long()
        st.session_state._long = res
with colB:
    if st.button("DAY", use_container_width=True):
        res = do_day()
        st.session_state._day = res
with colC:
    if st.button("ML", use_container_width=True, type="primary"):
        res = do_ml()
        st.session_state._ml = res
with colD:
    if st.button("BT FAST", use_container_width=True):
        res = do_btfast()
        st.session_state._bt = res

if run_all:
    st.session_state._long = do_long()
    st.session_state._day  = do_day()
    st.session_state._ml   = do_ml()

st.divider()

# ---- cards / layout ----
left, right = st.columns([1,1])

with left:
    st.subheader("📈 Long-Term Trend")
    if st.session_state.get("_long"):
        trend, buy = st.session_state._long
        st.markdown(f"- **{trend}**")
        if buy: st.markdown(f"- {buy}")
    else:
        st.caption("Click **LONG** to compute.")

    st.markdown("---")
    st.subheader("⚡ Day-Trade Snapshot")
    if st.session_state.get("_day"):
        action, bb = st.session_state._day
        st.markdown(f"- **{action}**")
        if bb: st.markdown(f"- {bb}")
    else:
        st.caption("Click **DAY** to compute.")

with right:
    st.subheader("🧠 ML Forecast — Price Targets")
    if st.session_state.get("_ml"):
        df, align, weak, confmax, raw_out = st.session_state._ml
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No ML rows parsed. Run **ML** once more or try another ticker.")

        # Confidence gauge
        st.markdown("#### Confidence")
        st.markdown(conf_bar_html(confmax), unsafe_allow_html=True)

        # Badges (alignment + weak)
        badges = []
        if align:
            color = GREEN if "UP" in align.upper() else RED if "DOWN" in align.upper() else "#334155"
            badges.append(color_badge(align, color))
        if weak:
            badges.append(color_badge(weak, YELLOW))
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)

        # Optional: raw console output (collapsed)
        with st.expander("Raw ML console output"):
            st.code(raw_out, language="text")
    else:
        st.caption("Click **ML** to compute.")

st.divider()
st.subheader("📜 Backtest (Fast)")
if st.session_state.get("_bt"):
    st.code(st.session_state._bt, language="text")
else:
    st.caption("Click **BT FAST** to run a quick expanding backtest (≈ 30–90s).")
#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git push -u origin main

