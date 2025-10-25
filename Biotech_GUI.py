# Biotech_GUI_Styled.py — Aesthetic GUI with real table layout (uses your existing Biotech.py)
# Requires: customtkinter (installed from GitHub), tkinter (built-in)

import io
import re
import threading
from contextlib import redirect_stdout

import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

# Import your engine (do not modify Biotech.py)
from Biotech import TechnicalAnalyzer, _post_webhook

APP_TITLE = "🧠 Biotech Pro — TA + ML"
DEFAULT_TICKER = "NVDA"

# ---------------- Colors ----------------
COL_BG      = "#0b1220"   # cards
COL_PANEL   = "#0a0f1a"   # window bg
COL_ACCENT  = "#2563eb"   # blue
COL_GREEN   = "#16a34a"
COL_YELLOW  = "#f59e0b"
COL_RED     = "#ef4444"
COL_TEXT    = "#e5e7eb"
COL_MUTE    = "#93a1b3"

# ------------- helpers ------------------
def capture_stdout(fn):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()

def run_bg(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t

def parse_ml_output(txt: str):
    """
    Parse your ML console text into rows + notes.
    Returns: rows(list of tuples), alignment(str or ''), weak(str or ''), conf_max(float 0..1)
    """
    rows = []
    confmax = 0.0
    # lines like: Next 1h: $123.45 (+0.67%) → UP 🔺 | Confidence: 86.2%
    pat = re.compile(
        r"Next\s+([0-9a-zA-Z]+):\s*\$([0-9]+\.[0-9]+)\s*\(([+\-]?[0-9]+\.[0-9]+)%\)\s*→\s*(UP|DOWN).+?Confidence:\s*([0-9]+\.[0-9])%",
        re.IGNORECASE
    )
    for m in pat.finditer(txt):
        horizon = m.group(1)
        price   = float(m.group(2))
        pct     = float(m.group(3))
        dirn    = m.group(4).upper()
        conf    = float(m.group(5))
        rows.append((horizon, f"{price:.2f}", f"{pct:+.2f}%", dirn, f"{conf:.1f}%"))
        confmax = max(confmax, conf)

    align = ""
    weak  = ""
    for line in txt.splitlines():
        if "Alignment:" in line:
            align = line.strip().replace("✅ ", "").replace("⚪ ", "")
        if "Signal weak" in line:
            weak = "Signal weak (|Δ%|<0.40% or conf<82%)."

    return rows, align, weak, confmax/100.0

# ------------- main app -----------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=COL_PANEL)
        self.title(APP_TITLE)
        self.geometry("1180x780")
        self.minsize(1080, 680)

        # fonts
        self.f_title = ctk.CTkFont(size=18, weight="bold")
        self.f_sub   = ctk.CTkFont(size=13)
        self.f_badge = ctk.CTkFont(size=12, weight="bold")

        # header
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color=COL_PANEL)
        hdr.pack(fill="x", pady=(4,0))
        ctk.CTkLabel(hdr, text="🧠 Biotech Pro", font=self.f_title).pack(side="left", padx=16, pady=8)
        self.theme_var = tk.StringVar(value="dark")
        theme = ctk.CTkSegmentedButton(hdr, values=["dark", "light"], variable=self.theme_var, command=self._toggle_theme)
        theme.pack(side="right", padx=14, pady=8)

        # toolbar
        bar = ctk.CTkFrame(self, corner_radius=12, fg_color=COL_BG)
        bar.pack(fill="x", padx=16, pady=(12,8))
        ctk.CTkLabel(bar, text="Ticker", font=self.f_sub).pack(side="left", padx=(12,6), pady=10)
        self.ticker_var = tk.StringVar(value=DEFAULT_TICKER)
        self.inp_ticker = ctk.CTkEntry(bar, width=120, textvariable=self.ticker_var)
        self.inp_ticker.pack(side="left", padx=(0,10), pady=10)

        self.post_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="Post to Discord", variable=self.post_var).pack(side="left", padx=6)

        # segmented actions
        act = ctk.CTkFrame(bar, fg_color="transparent")
        act.pack(side="left", padx=14)
        self._mk_btn(act, "Run All", self.on_all, fg=COL_ACCENT)
        self._mk_btn(act, "LONG",   self.on_long, fg="#334155")
        self._mk_btn(act, "DAY",    self.on_day,  fg="#334155")
        self._mk_btn(act, "ML",     self.on_ml,   fg=COL_GREEN)
        self._mk_btn(act, "BT FAST",self.on_bt,   fg=COL_YELLOW)

        # status chips
        chips = ctk.CTkFrame(bar, fg_color="transparent")
        chips.pack(side="right", padx=8)
        self.chip_status = self._chip(chips, "Idle", "#64748b")
        self.chip_status.pack(side="right", padx=8, pady=8)

        # body layout (two columns)
        body = ctk.CTkFrame(self, corner_radius=12, fg_color=COL_PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=(4,16))

        # left — cards for LONG & DAY summaries (plain language)
        left = ctk.CTkFrame(body, corner_radius=12, fg_color=COL_PANEL)
        left.pack(side="left", fill="both", expand=True, padx=(10,8), pady=10)

        self.card_long = self._card(left, "Long-Term Trend")
        self.lbl_long  = ctk.CTkLabel(self.card_long, text="—", wraplength=440, justify="left")
        self.lbl_long.pack(anchor="w", padx=14, pady=(6,14))

        self.card_day = self._card(left, "Day-Trade Snapshot")
        self.lbl_day  = ctk.CTkLabel(self.card_day, text="—", wraplength=440, justify="left")
        self.lbl_day.pack(anchor="w", padx=14, pady=(6,14))

        # right — ML results + confidence + badges
        right = ctk.CTkFrame(body, corner_radius=12, fg_color=COL_PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(8,10), pady=10)

        # confidence gauge
        self.card_g = self._card(right, "Confidence")
        self.bar = ctk.CTkProgressBar(self.card_g, height=16, corner_radius=10)
        self.bar.pack(fill="x", padx=14, pady=(8,6))
        self.lbl_conf = ctk.CTkLabel(self.card_g, text="—", font=self.f_sub)
        self.lbl_conf.pack(anchor="e", padx=14, pady=(0,12))
        self._set_conf(0.0)

        # table card
        self.card_tbl = self._card(right, "ML Forecast — Price Targets")
        self._build_table(self.card_tbl)

        # badges
        self.card_badge = self._card(right, "Signal Status")
        self.badge_align = self._badge(self.card_badge, "—", bg="#334155")
        self.badge_align.pack(side="left", padx=10, pady=10)
        self.badge_weak  = self._badge(self.card_badge, "", bg=COL_YELLOW)
        self.badge_weak.pack(side="left", padx=10, pady=10)

        # footer
        self.footer = ctk.CTkLabel(self, text="Ready.", anchor="w")
        self.footer.pack(fill="x", padx=18, pady=(0,10))

    # ---------- UI building blocks ----------
    def _mk_btn(self, parent, text, cmd, fg=COL_ACCENT):
        b = ctk.CTkButton(parent, text=text, width=110, height=36, fg_color=fg, hover_color="#1f2937",
                          command=cmd)
        b.pack(side="left", padx=6, pady=8)

    def _chip(self, parent, text, color):
        frame = ctk.CTkFrame(parent, corner_radius=18, fg_color=COL_BG)
        dot = ctk.CTkLabel(frame, text="●", text_color=color, font=ctk.CTkFont(size=14, weight="bold"))
        dot.pack(side="left", padx=(10,6), pady=8)
        lab = ctk.CTkLabel(frame, text=text, font=self.f_sub); lab.pack(side="left", padx=(0,12))
        frame._dot = dot; frame._lab = lab
        return frame

    def _card(self, parent, title):
        c = ctk.CTkFrame(parent, corner_radius=14, fg_color=COL_BG)
        head = ctk.CTkFrame(c, fg_color=COL_BG); head.pack(fill="x", padx=12, pady=(10,0))
        ctk.CTkLabel(head, text=title, font=self.f_title).pack(side="left")
        c.pack(fill="x", padx=8, pady=8)
        return c

    def _badge(self, parent, text, bg=COL_GREEN):
        f = ctk.CTkFrame(parent, corner_radius=16, fg_color=bg)
        ctk.CTkLabel(f, text=text, font=self.f_badge).pack(padx=14, pady=6)
        f._set = lambda t, b=bg: (f.configure(fg_color=b), f.children[list(f.children)[0]].configure(text=t))
        return f

    def _build_table(self, parent):
        # style Treeview for dark mode
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=COL_BG, fieldbackground=COL_BG, foreground=COL_TEXT,
                        rowheight=28, borderwidth=0, relief="flat")
        style.map("Dark.Treeview", background=[("selected", "#1f2937")])
        style.configure("Dark.Treeview.Heading",
                        background="#0e1629", foreground=COL_TEXT, relief="flat")
        style.map("Dark.Treeview.Heading", background=[("active", "#0e1629")])

        cols = ("Horizon", "Price ($)", "Δ%", "Direction", "Confidence")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", style="Dark.Treeview", height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            w = 110 if c != "Horizon" else 90
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", padx=12, pady=12)

    def _set_table(self, rows):
        for it in self.tree.get_children():
            self.tree.delete(it)
        for r in rows:
            # r = (horizon, price, pct, dir, conf)
            self.tree.insert("", "end", values=r)

    def _set_conf(self, v):
        v = max(0.0, min(1.0, float(v)))
        self.bar.set(v)
        pct = int(round(v*100))
        self.lbl_conf.configure(text=f"{pct}%")
        color = COL_RED if v < 0.82 else COL_YELLOW if v < 0.87 else COL_GREEN
        self.bar.configure(progress_color=color)

    # ---------- theme ----------
    def _toggle_theme(self, *_):
        ctk.set_appearance_mode(self.theme_var.get())

    # ---------- engine calls (threaded) ----------
    def _maybe_post(self, msg):
        if self.post_var.get():
            _post_webhook(msg)

    def on_all(self):
        self.on_long(run_next=lambda: self.on_day(run_next=lambda: self.on_ml()))

    def on_long(self, run_next=None):
        tkc = self.ticker_var.get().strip().upper()
        if not tkc: return
        self._status("Running LONG…", "#38bdf8")
        def work():
            out = capture_stdout(lambda: TechnicalAnalyzer(tkc).long_term())
            # extract a friendly sentence
            trend = "Trend: —"
            buy   = ""
            for ln in out.splitlines():
                if "Trend:" in ln: trend = ln.strip()
                if "Potential Buy Price:" in ln: buy = ln.strip()
            txt = f"{trend.replace('Trend:','').strip()}\n{buy}"
            self.after(0, lambda: (self.lbl_long.configure(text=txt),
                                   self._status("Done", COL_GREEN),
                                   self._maybe_post(f"**GUI** LONG {tkc}"),
                                   run_next() if run_next else None))
        run_bg(work)

    def on_day(self, run_next=None):
        tkc = self.ticker_var.get().strip().upper()
        if not tkc: return
        self._status("Running DAY…", "#38bdf8")
        def work():
            out = capture_stdout(lambda: TechnicalAnalyzer(tkc).day_trade())
            # friendly summary
            action = ""
            bb = ""
            for ln in out.splitlines():
                if "Suggested Action:" in ln: action = ln.strip()
                if "Bollinger Bands" in ln or "BB Upper" in ln: bb = ln.strip()
            txt = f"{action}\n{bb}"
            self.after(0, lambda: (self.lbl_day.configure(text=txt),
                                   self._status("Done", COL_GREEN),
                                   self._maybe_post(f"**GUI** DAY {tkc}"),
                                   run_next() if run_next else None))
        run_bg(work)

    def on_ml(self):
        tkc = self.ticker_var.get().strip().upper()
        if not tkc: return
        self._status("Running ML…", "#a855f7")
        def work():
            out = capture_stdout(lambda: TechnicalAnalyzer(tkc).ml_forecast())
            rows, align, weak, confmax = parse_ml_output(out)
            def ui():
                self._set_table(rows)
                self._set_conf(confmax)
                # badges
                if align:
                    col = COL_GREEN if "UP" in align.upper() else (COL_RED if "DOWN" in align.upper() else "#334155")
                    self.badge_align._set(align, col)
                else:
                    self.badge_align._set("—", "#334155")
                self.badge_weak._set(weak, COL_YELLOW if weak else "#334155")
                self._status("Done", COL_GREEN)
                self._maybe_post(f"**GUI** ML {tkc}")
            self.after(0, ui)
        run_bg(work)

    def on_bt(self):
        tkc = self.ticker_var.get().strip().upper()
        if not tkc: return
        self._status("Backtest (fast)…", COL_YELLOW)
        def work():
            out = capture_stdout(lambda: TechnicalAnalyzer(tkc).ml_backtest_fast(horizon="1d", step=20, max_models=150))
            # Put the backtest text into Long card so it shows somewhere readable
            self.after(0, lambda: (self.lbl_long.configure(text=out or "Backtest done."),
                                   self._status("Done", COL_GREEN),
                                   self._maybe_post(f"**GUI** BTFAST {tkc}")))
        run_bg(work)

    # ---------- status ----------
    def _status(self, text, color):
        self.chip_status._lab.configure(text=text)
        self.chip_status._dot.configure(text="●", text_color=color)


if __name__ == "__main__":
    app = App()
    app.mainloop()
