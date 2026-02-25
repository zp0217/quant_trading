"""
most part of app was created by AI
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import numpy as np
import pandas as pd

import dash
from dash import html, dcc, dash_table, Input, Output, State, ctx, ALL
import dash.exceptions
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetcher   import fetch_stock_data, get_stock_info
from strategies.momentum import run_momentum_strategy
from strategies.bollinger import run_bollinger_strategy
from strategies.rsi       import run_rsi_strategy
from utils.metrics        import compare_strategies
from models.lstm_model    import train_lstm
from models.gru_model     import train_gru
from models.arima_model   import run_arima
from portfolio.optimizer  import build_portfolio, efficient_frontier

# ─────────────────────────────────────────────────────────────────
# Colors / Constants
# ─────────────────────────────────────────────────────────────────
F = "'Courier New', monospace"

C = dict(
    bg="#0d0d0d", panel="#111", sidebar="#0a0a0a", header="#0f0f0f",
    tabbar="#0a0a0a", border="#1e1e1e", border2="#2a2a2a", grid="#161616",
    white="#e8e8e8", gray="#666", dim="#333",
    green="#00ff88", green2="#00aa55", red="#ff3333", red2="#aa2222",
    yellow="#ffcc00", cyan="#00ccff", orange="#ff8800", purple="#9966ff",
    mom="#00ff88", boll="#ff8800", rsi_c="#cc66ff",
    lstm_c="#00ccff", gru_c="#ff66aa", arima_c="#ffcc00",
)

TABS    = ["OVERVIEW", "CHART", "FORECAST", "UNIVERSE", "RISK", "PORTFOLIO", "LIVE LOG"]
PERIODS = ["1y", "2y", "3y", "5y"]   # Minimum 1 year

# ─────────────────────────────────────────────────────────────────
# Dash App
# ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="algorithm trading demo",
)

app.index_string = """<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#d0d0d0;font-family:'Courier New',monospace;overflow-x:hidden}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#0a0a0a}
::-webkit-scrollbar-thumb{background:#2a2a2a}
/* Dropdown */
.dd .Select-control{background:#111!important;border:1px solid #252525!important;
  color:#00ccff!important;font-family:'Courier New',monospace!important;font-size:11px!important;
  min-height:26px!important;border-radius:0!important}
.dd .Select-value-label,.dd .Select-placeholder{color:#00ccff!important;line-height:24px!important}
.dd .Select-menu-outer{background:#111!important;border:1px solid #252525!important;
  border-radius:0!important;z-index:9999!important}
.dd .Select-option{background:#111!important;color:#aaa!important;
  font-family:monospace!important;font-size:11px!important;padding:4px 8px!important}
.dd .Select-option:hover,.dd .Select-option.is-focused{background:#1a1a1a!important;color:#00ccff!important}
.dd .Select-arrow{border-top-color:#444!important}
.dd .Select-input input{color:#00ccff!important;font-family:monospace!important}
input[type=number]{background:#111;border:1px solid #252525;color:#ffcc00;
  font-family:'Courier New',monospace;font-size:11px;padding:2px 6px;width:100%}
.sym-pill{display:inline-flex;align-items:center;padding:2px 8px;background:#111;
  border:1px solid #252525;color:#00ff88;font-family:monospace;font-size:10px;letter-spacing:1px}
.sym-pill:hover{border-color:#555}
.rm-x{color:#444;font-size:11px;cursor:pointer;margin-left:5px;
  font-family:monospace;user-select:none;transition:color .15s}
.rm-x:hover{color:#ff4444!important}
/* Buttons */
.run-btn{background:#00ff88!important;color:#000!important;font-weight:bold!important;
  font-family:monospace!important;font-size:11px!important;letter-spacing:2px!important;
  border:none!important;padding:5px 16px!important;cursor:pointer!important;transition:.15s}
.run-btn:hover{background:#00cc66!important}
.run-btn:disabled{background:#003322!important;color:#006644!important;cursor:not-allowed!important}
.add-btn{background:transparent!important;border:1px solid #252525!important;color:#aaa!important;
  font-family:monospace!important;font-size:10px!important;padding:4px 10px!important;cursor:pointer!important}
.add-btn:hover{border-color:#555!important;color:#fff!important}
/* Toast notification */
#toast{position:fixed;top:18px;right:18px;z-index:999999;
  min-width:280px;pointer-events:none;transition:opacity .3s}
#toast.hidden{opacity:0}
#toast.visible{opacity:1}
.toast-box{background:#0f0f0f;border:1px solid #ffcc00;
  border-left:3px solid #ffcc00;padding:10px 14px;
  font-family:'Courier New',monospace;font-size:11px;
  color:#ffcc00;letter-spacing:.5px;
  box-shadow:0 0 20px rgba(255,204,0,.15)}
/* Loading overlay */
#ld-overlay{position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,0.88);backdrop-filter:blur(3px);
  z-index:99999;display:flex;align-items:center;justify-content:center}
#ld-overlay.hidden{display:none!important}
.ld-box{border:1px solid #00ff88;background:#0a0a0a;padding:36px 52px;
  text-align:center;min-width:380px;box-shadow:0 0 60px rgba(0,255,136,.1)}
.ld-title{color:#00ff88;font-size:14px;letter-spacing:5px;margin-bottom:20px}
.ld-msg{color:#888;font-size:11px;letter-spacing:1px;min-height:18px;margin-bottom:22px}
.ld-track{width:100%;height:2px;background:#1a1a1a;margin-bottom:18px;overflow:hidden}
.ld-bar{height:2px;background:#00ff88;animation:ldr 1.8s ease-in-out infinite}
@keyframes ldr{0%{width:0%;margin-left:0%}60%{width:50%;margin-left:25%}100%{width:0%;margin-left:100%}}
.ld-dots{display:flex;justify-content:center;gap:6px;margin-bottom:14px}
.ld-dot{width:6px;height:6px;border-radius:50%;background:#00ff88;opacity:.15;
  animation:ldd 1.2s ease-in-out infinite}
.ld-dot:nth-child(2){animation-delay:.2s}
.ld-dot:nth-child(3){animation-delay:.4s}
@keyframes ldd{0%,80%,100%{opacity:.15;transform:scale(1)}40%{opacity:1;transform:scale(1.4)}}
.ld-hint{color:#2a2a2a;font-size:9px;letter-spacing:1px}
</style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────
def _s(v, d=0.0):
    """None/NaN/inf → safe numeric conversion"""
    if v is None: return d
    try:
        f = float(v)
        return d if (f != f or abs(f) == float("inf")) else f
    except Exception:
        return d

def _rgb(h):
    h = h.lstrip("#")
    if len(h) == 3: h = "".join(c*2 for c in h)
    return f"{int(h[:2],16)},{int(h[2:4],16)},{int(h[4:],16)}"

def _ts(): return datetime.now().strftime("%H:%M:%S")

def _fv(v):
    v = _s(v)
    if v >= 1e12: return f"{v/1e12:.1f}T"
    if v >= 1e9:  return f"{v/1e9:.1f}B"
    if v >= 1e6:  return f"{v/1e6:.1f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}K"
    return str(int(v))

def _ax():
    return dict(gridcolor=C["grid"], gridwidth=1, zeroline=False,
                tickfont=dict(family=F, color=C["gray"], size=9),
                linecolor=C["border2"])

def _tl(fig, title="", h=300, pad=None):
    p = pad or dict(l=44, r=10, t=28, b=22)
    fig.update_layout(
        paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
        height=h, margin=p,
        font=dict(color=C["white"], family=F, size=10),
        title=dict(text=title, x=0.01, xanchor="left",
                   font=dict(color=C["gray"], size=10, family=F)),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(family=F, size=9, color=C["gray"]),
                    orientation="h", y=1.05, x=0.5, xanchor="center"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#111", font=dict(family=F, size=10)),
    )
    fig.update_xaxes(**_ax())
    fig.update_yaxes(**_ax())

def _card(children, tc=None, style=None):
    s = {"backgroundColor": C["panel"], "border": f"1px solid {C['border']}",
         "padding": "10px", "marginBottom": "8px"}
    if tc: s["borderTop"] = f"2px solid {tc}"
    if style: s.update(style)
    return html.Div(children, style=s)

def _lbl(t):
    return html.Span(t, style={"color": C["gray"], "fontSize": "9px",
                                "letterSpacing": "1px", "fontFamily": F})
def _val(t, c=None):
    return html.Span(str(t), style={"color": c or C["white"],
                                     "fontSize": "13px", "fontWeight": "bold", "fontFamily": F})


# ─────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────
def _sb_item(label, value, color=None):
    return html.Div([
        html.Div(label, style={"color": C["gray"], "fontSize": "9px",
                               "letterSpacing": ".8px", "marginBottom": "1px"}),
        html.Div(str(value), style={"color": color or C["green"], "fontSize": "14px",
                                     "fontWeight": "bold", "marginBottom": "8px",
                                     "fontFamily": F}),
    ])

app.layout = html.Div([
    html.Div([

        # ── Sidebar ─────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("trading", style={"color": C["green"], "fontSize": "11px",
                    "fontWeight": "bold", "letterSpacing": "2px"}),
                html.Div("TERMINAL v5", style={"color": C["gray"], "fontSize": "9px"}),
            ], style={"padding": "10px 10px 8px",
                      "borderBottom": f"1px solid {C['border']}", "marginBottom": "10px"}),

            html.Div("ML MODEL", style={"color": C["gray"], "fontSize": "9px",
                "letterSpacing": "1px", "marginBottom": "4px", "padding": "0 10px"}),
            html.Div(dcc.Dropdown(id="model-dd", className="dd",
                options=[{"label": l, "value": v} for l, v in [
                    ("LSTM","lstm"),("GRU","gru"),("ARIMA","arima"),("ALL","all")]],
                value="arima", clearable=False),
                style={"padding": "0 10px", "marginBottom": "10px"}),

            html.Div("PERIOD", style={"color": C["gray"], "fontSize": "9px",
                "letterSpacing": "1px", "marginBottom": "4px", "padding": "0 10px"}),
            html.Div(dcc.Dropdown(id="period-dd", className="dd",
                options=[{"label": p, "value": p} for p in PERIODS],
                value="2y", clearable=False),
                style={"padding": "0 10px", "marginBottom": "10px"}),

            html.Div("DL EPOCHS", style={"color": C["gray"], "fontSize": "9px",
                "letterSpacing": "1px", "marginBottom": "4px", "padding": "0 10px"}),
            html.Div(dcc.Input(id="epochs-in", type="number", value=20, min=5, max=100, step=5,
                style={"backgroundColor": "#111", "border": f"1px solid {C['border2']}",
                       "color": C["yellow"], "fontFamily": F, "fontSize": "11px",
                       "padding": "3px 8px", "width": "100%"}),
                style={"padding": "0 10px", "marginBottom": "14px"}),

            html.Hr(style={"borderColor": C["border"], "margin": "0 10px 10px"}),

            html.Div([
                html.Div("STRATEGIES", style={"color": C["gray"], "fontSize": "9px",
                    "letterSpacing": "1px", "marginBottom": "6px"}),
                *[html.Div(f"▶ {n}", style={"color": c, "fontSize": "9px", "padding": "2px 0"})
                  for n, c in [("Momentum",C["mom"]),("Bollinger",C["boll"]),("RSI Rev.",C["rsi_c"])]],
            ], style={"padding": "0 10px", "marginBottom": "10px"}),

            html.Hr(style={"borderColor": C["border"], "margin": "0 10px 10px"}),

            html.Div("PORTFOLIO", style={"color": C["gray"], "fontSize": "9px",
                "letterSpacing": "2px", "padding": "0 10px", "marginBottom": "6px"}),
            html.Div(id="sb-stats", style={"padding": "0 10px"}),

            html.Div(id="sb-time", style={"color": C["dim"], "fontSize": "9px",
                "padding": "10px", "position": "absolute", "bottom": "8px"}),
        ], style={"width": "160px", "flexShrink": "0", "backgroundColor": C["sidebar"],
                  "borderRight": f"1px solid {C['border']}", "height": "100vh",
                  "overflowY": "auto", "position": "relative"}),

        # ── Main ─────────────────────────────────────────────
        html.Div([
            # Header
            html.Div([
                html.Div(id="active-tickers",
                    style={"display": "flex", "alignItems": "center", "gap": "4px", "flex": "1"}),
                html.Div([
                    html.Span("TICKER", style={"color": C["gray"], "fontSize": "10px",
                        "letterSpacing": "1px", "marginRight": "6px"}),
                    dcc.Input(id="ticker-in", type="text", placeholder="AAPL",
                        style={"backgroundColor": "#0a0a0a", "border": f"1px solid {C['border2']}",
                               "color": C["green"], "fontFamily": F, "fontSize": "11px",
                               "padding": "3px 8px", "width": "110px", "outline": "none"}),
                    html.Button("ADD",   id="add-btn",   n_clicks=0, className="add-btn"),
                    html.Button("▶  RUN", id="run-btn", n_clicks=0,
                        className="run-btn", disabled=False),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
            ], style={"backgroundColor": C["header"], "borderBottom": f"1px solid {C['border']}",
                      "padding": "6px 14px", "display": "flex",
                      "alignItems": "center", "justifyContent": "space-between"}),

            # Symbol row
            html.Div([
                html.Span("SYMBOL : ", style={"color": C["gray"], "fontSize": "10px",
                    "letterSpacing": "1px"}),
                html.Div(id="symbol-pills", style={"display": "flex", "gap": "4px"}),
            ], style={"backgroundColor": C["bg"], "borderBottom": f"1px solid {C['border']}",
                      "padding": "5px 14px", "display": "flex", "alignItems": "center", "gap": "8px"}),

            # Tab bar
            html.Div([
                html.Div(t, id={"type": "tab-btn", "index": t}, n_clicks=0,
                    style={"padding": "7px 14px", "cursor": "pointer", "fontSize": "10px",
                           "letterSpacing": "1.5px", "color": C["gray"], "fontFamily": F,
                           "borderBottom": "2px solid transparent", "userSelect": "none"})
                for t in TABS
            ], style={"backgroundColor": C["tabbar"], "borderBottom": f"1px solid {C['border']}",
                      "display": "flex", "alignItems": "stretch"}),

            # Content
            html.Div(id="content-area",
                style={"padding": "10px 14px", "overflowY": "auto",
                       "height": "calc(100vh - 107px)"}),
        ], style={"flex": "1", "overflow": "hidden", "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "height": "100vh", "overflow": "hidden"}),

    # Store — results-store is memory (clears on refresh so button state is consistent)
    # On refresh: store=None → welcome screen, run-btn enabled → normal operation
    dcc.Store(id="tickers-store", storage_type="session", data=[]),
    dcc.Store(id="results-store", storage_type="memory",  data=None),
    dcc.Store(id="log-store",     storage_type="memory",  data=[]),
    dcc.Store(id="active-tab",    storage_type="session", data="OVERVIEW"),
    dcc.Interval(id="clk-iv", interval=1000, n_intervals=0),

    # Toast notification (ticker limit exceeded, etc.)
    html.Div(
        html.Div(id="toast-msg", className="toast-box"),
        id="toast", className="hidden"
    ),

    # Loading overlay
    html.Div([
        html.Div([
            html.Div("◈  ANALYZING", className="ld-title"),
            html.Div(id="ld-msg", className="ld-msg", children="Initializing..."),
            html.Div(className="ld-track", children=html.Div(className="ld-bar")),
            html.Div(className="ld-dots", children=[
                html.Div(className="ld-dot"),
                html.Div(className="ld-dot"),
                html.Div(className="ld-dot"),
            ]),
            html.Div("Will close automatically when analysis is complete", className="ld-hint"),
        ], className="ld-box"),
    ], id="ld-overlay", className="hidden"),

], style={"backgroundColor": C["bg"]})


# ─────────────────────────────────────────────────────────────────
# Callback: Clock
# ─────────────────────────────────────────────────────────────────
@app.callback(Output("sb-time", "children"), Input("clk-iv", "n_intervals"))
def _clk(n):
    return datetime.now().strftime("%Y.%m.%d  %H:%M:%S")


MAX_TICKERS = 5   # Maximum number of tickers

# ─────────────────────────────────────────────────────────────────
# Callback: Add ticker (ADD button / Enter)
# ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("tickers-store", "data", allow_duplicate=True),
    Output("ticker-in",     "value"),
    Output("toast-msg",     "children"),
    Output("toast",         "className"),
    Input("add-btn",        "n_clicks"),
    Input("ticker-in",      "n_submit"),
    State("ticker-in",      "value"),
    State("tickers-store",  "data"),
    prevent_initial_call=True,
)
def add_ticker(add, sub, inp, tickers):
    tickers = tickers or []
    if inp and inp.strip():
        t = inp.strip().upper()
        if t in tickers:
            return tickers, "", f"⚠  {t} is already added.", "visible"
        if len(tickers) >= MAX_TICKERS:
            return tickers, inp, f"⚠  Maximum {MAX_TICKERS} tickers allowed.", "visible"
        return tickers + [t], "", "", "hidden"
    return tickers, (inp or ""), "", "hidden"


# ─────────────────────────────────────────────────────────────────
# Callback: Remove individual ticker (× button on each pill)
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("tickers-store", "data", allow_duplicate=True),
    Input({"type": "remove-ticker", "index": ALL}, "n_clicks"),
    State("tickers-store", "data"),
    prevent_initial_call=True,
)
def remove_ticker(n_clicks_list, tickers):
    tickers = tickers or []
    if not ctx.triggered_id or not any(n for n in (n_clicks_list or []) if n):
        raise dash.exceptions.PreventUpdate
    # Remove _pill suffix from index to get ticker name
    raw = ctx.triggered_id["index"]
    ticker_to_remove = raw.replace("_pill", "")
    return [t for t in tickers if t != ticker_to_remove]


@app.callback(
    Output("active-tickers", "children"),
    Output("symbol-pills",   "children"),
    Input("tickers-store", "data"),
)
def update_pills(tickers):
    tickers = tickers or []

    # Header ticker tags (with X button)
    hdr = []
    for t in tickers:
        hdr.append(html.Div([
            html.Span(t, style={"fontSize": "10px", "fontFamily": F,
                "letterSpacing": "1px", "color": C["green"], "marginRight": "3px"}),
            html.Span("×", id={"type": "remove-ticker", "index": t},
                n_clicks=0, className="rm-x"),
        ], style={"backgroundColor": "#001a0d", "border": f"1px solid {C['border2']}",
                  "fontSize": "10px", "padding": "2px 6px", "fontFamily": F,
                  "display": "inline-flex", "alignItems": "center"}))

    # Symbol row pills (with X button)
    pills = []
    for t in tickers:
        pills.append(html.Div([
            html.Span(t, style={"fontSize": "10px", "fontFamily": F,
                "letterSpacing": "1px", "color": C["green"]}),
            html.Span("×", id={"type": "remove-ticker", "index": t + "_pill"},
                n_clicks=0, className="rm-x"),
        ], className="sym-pill"))

    return hdr, pills


# ─────────────────────────────────────────────────────────────────
# Callback: Tab switching
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("active-tab", "data"),
    Input({"type": "tab-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_tab(nc):
    return ctx.triggered_id["index"] if ctx.triggered_id else "OVERVIEW"


@app.callback(
    Output({"type": "tab-btn", "index": ALL}, "style"),
    Input("active-tab", "data"),
)
def style_tabs(active):
    out = []
    for t in TABS:
        base = {"padding": "7px 14px", "cursor": "pointer", "fontSize": "10px",
                "letterSpacing": "1.5px", "fontFamily": F, "userSelect": "none"}
        if t == active:
            out.append({**base, "color": C["green"],
                        "borderBottom": f"2px solid {C['green']}",
                        "backgroundColor": C["tabbar"]})
        else:
            out.append({**base, "color": C["gray"], "borderBottom": "2px solid transparent"})
    return out


# ─────────────────────────────────────────────────────────────────
# Clientside: Show overlay immediately on RUN click
# ─────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(n) {
        if (!n) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        var o = document.getElementById('ld-overlay');
        if (o) o.className = '';
        var b = document.getElementById('run-btn');
        if (b) b.disabled = true;
        return ['', true];
    }
    """,
    Output("ld-overlay", "className", allow_duplicate=True),
    Output("run-btn",    "disabled",  allow_duplicate=True),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)

# ─────────────────────────────────────────────────────────────────
# Clientside: Auto-hide toast after 3 seconds
# ─────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(cls) {
        if (cls === 'visible') {
            setTimeout(function() {
                var el = document.getElementById('toast');
                if (el) el.className = 'hidden';
            }, 3000);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("toast", "className", allow_duplicate=True),
    Input("toast",  "className"),
    prevent_initial_call=True,
)


# ─────────────────────────────────────────────────────────────────
# Callback: Loading message update
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("ld-msg", "children"),
    Input("log-store", "data"),
    prevent_initial_call=True,
)
def update_ld_msg(logs):
    if not logs: return "Preparing..."
    last = logs[-1]
    msg  = last.split("] ", 1)[-1] if "] " in last else last
    for kw, ic in [("FETCHING","📡 "),("strategy","📊 "),("LSTM","🧠 "),
                   ("GRU","🧠 "),("ARIMA","📈 "),("complete","✅ "),
                   ("DONE","✅ "),("ERROR","⚠  ")]:
        if kw in msg:
            return ic + msg
    return msg


# ─────────────────────────────────────────────────────────────────
# Callback: Main analysis
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("results-store", "data"),
    Output("log-store",     "data"),
    Output("sb-stats",      "children"),
    Output("ld-overlay",    "className"),
    Output("run-btn",       "disabled"),
    Input("run-btn", "n_clicks"),
    State("tickers-store", "data"),
    State("period-dd",     "value"),
    State("model-dd",      "value"),
    State("epochs-in",     "value"),
    prevent_initial_call=True,
)
def run_analysis(n, tickers, period, model_sel, epochs):
    DONE = ("hidden", False)  # (overlay_class, btn_disabled)

    if not tickers:
        return None, [], _sb_empty(), *DONE

    epochs = int(epochs or 20)
    logs, store = [f"[{_ts()}] ▶ RUN  {','.join(tickers)}  period={period}"], {}

    for tk in tickers:
        try:
            logs.append(f"[{_ts()}] FETCHING {tk}...")
            df = fetch_stock_data(tk, period=period)
            if df is None or df.empty or len(df) < 20:
                logs.append(f"[{_ts()}] ERROR {tk}: Insufficient data ({len(df) if df is not None else 0} rows)")
                continue

            info = get_stock_info(tk)
            logs.append(f"[{_ts()}] {tk}: {len(df)} rows received")

            # Strategies
            logs.append(f"[{_ts()}] {tk}: Running strategies...")
            mom   = _safe_strat(run_momentum_strategy, df, "Momentum")
            boll  = _safe_strat(run_bollinger_strategy, df, "Bollinger")
            rsi_s = _safe_strat(run_rsi_strategy, df, "RSI")

            # ML
            ml_raw = {}
            run_list = ["lstm","gru","arima"] if model_sel == "all" else [model_sel]
            for m in run_list:
                logs.append(f"[{_ts()}] {tk}: Running {m.upper()}...")
                try:
                    if m == "lstm":
                        ml_raw["LSTM"] = train_lstm(df, epochs=epochs)
                    elif m == "gru":
                        ml_raw["GRU"]  = train_gru(df,  epochs=epochs)
                    elif m == "arima":
                        ml_raw["ARIMA"] = run_arima(df)
                    logs.append(f"[{_ts()}] {tk}: {m.upper()} complete")
                except Exception as e:
                    logs.append(f"[{_ts()}] {tk}: {m.upper()} failed — {e}")

            store[tk] = _pack(tk, df, info, mom, boll, rsi_s, [mom,boll,rsi_s], ml_raw)
            logs.append(f"[{_ts()}] {tk}: ✓ complete")

        except Exception as e:
            logs.append(f"[{_ts()}] ERROR {tk}: {type(e).__name__}: {e}")
            continue

    logs.append(f"[{_ts()}] ■ DONE — {len(store)} tickers")
    return store or None, logs, _build_sb_stats(store), *DONE


def _safe_strat(fn, df, name):
    try:
        r = fn(df)
        if r and "data" in r and len(r["data"]) > 0:
            return r
    except Exception:
        pass
    empty = pd.DataFrame({col: pd.Series(dtype=t) for col, t in [
        ("Signal","int"), ("Cumulative_Strategy","float"),
        ("Cumulative_BuyHold","float"), ("Strategy_Return","float")]})
    return {"name": name, "data": empty, "buy_signals": pd.DatetimeIndex([]),
            "sell_signals": pd.DatetimeIndex([]), "metrics": {}}


def _pack(tk, df, info, mom, boll, rsi_s, strats, ml_raw):
    """Convert all results to JSON-serializable dict"""
    close = df["Close"]
    last  = _s(close.iloc[-1] if len(close) else 0)
    prev  = _s(close.iloc[-2] if len(close)>1 else last)
    chg   = (last-prev)/prev*100 if prev else 0.0

    def _sig(r):
        try:
            buys  = [str(x) for x in (r.get("buy_signals")  or pd.DatetimeIndex([])).tolist()]
            sells = [str(x) for x in (r.get("sell_signals") or pd.DatetimeIndex([])).tolist()]
            d = r.get("data", pd.DataFrame())
            cur = int(d["Signal"].iloc[-1]) if (not d.empty and "Signal" in d.columns) else 0
        except Exception:
            buys, sells, cur = [], [], 0
        return {"buys": buys, "sells": sells, "cur": cur}

    def _cum(r):
        try:
            d = r.get("data", pd.DataFrame())
            if d.empty or "Cumulative_Strategy" not in d.columns:
                return {"dates":[], "strat":[], "bh":[]}
            return {
                "dates": [str(x) for x in d.index.tolist()],
                "strat": (d["Cumulative_Strategy"]-1).mul(100).round(3).tolist(),
                "bh":    (d["Cumulative_BuyHold"] -1).mul(100).round(3).tolist(),
            }
        except Exception:
            return {"dates":[], "strat":[], "bh":[]}

    ohlcv = {
        "dates":  [str(x) for x in df.index.tolist()],
        "open":   df["Open"].round(4).tolist(),
        "high":   df["High"].round(4).tolist(),
        "low":    df["Low"].round(4).tolist(),
        "close":  close.round(4).tolist(),
        "volume": df["Volume"].tolist(),
        "ma20":   df["MA20"].round(4).tolist()    if "MA20"     in df.columns else [],
        "ma50":   df["MA50"].round(4).tolist()    if "MA50"     in df.columns else [],
        "ma200":  df["MA200"].round(4).tolist()   if "MA200"    in df.columns else [],
        "bb_u":   df["BB_Upper"].round(4).tolist()if "BB_Upper" in df.columns else [],
        "bb_l":   df["BB_Lower"].round(4).tolist()if "BB_Lower" in df.columns else [],
        "rsi":    df["RSI"].round(2).tolist()     if "RSI"      in df.columns else [],
        "vol_ma": df["Vol_MA20"].round(0).tolist()if "Vol_MA20" in df.columns else [],
    }

    # ML serialization
    ml_out = {}
    for k, r in ml_raw.items():
        if not r or "error" in r:
            ml_out[k] = {"_err": (r or {}).get("error","Execution failed")}
            continue
        try:
            if k in ("LSTM","GRU"):
                pred   = r.get("pred_prices",   np.array([]))
                actual = r.get("actual_prices", np.array([]))
                tidx   = r.get("test_index",    pd.DatetimeIndex([]))
                fidx   = r.get("fc_index",      pd.DatetimeIndex([]))
                ml_out[k] = {
                    "hist_dates": [str(x) for x in tidx.tolist()],
                    "hist_actual":[round(_s(x),4) for x in actual.tolist()],
                    "hist_pred":  [round(_s(x),4) for x in pred.tolist()],
                    "fc_dates":   [str(x) for x in fidx.tolist()],
                    "fc":   [round(_s(x),4) for x in r.get("fc_mean",  np.array([])).tolist()],
                    "ci_l": [round(_s(x),4) for x in r.get("ci_lower", np.array([])).tolist()],
                    "ci_u": [round(_s(x),4) for x in r.get("ci_upper", np.array([])).tolist()],
                    "next":   _s(r.get("next_day_pred")),
                    "fc_30d": _s(r.get("forecast_30d")),
                    "ret30":  _s(r.get("ret30")),
                    "dir":    r.get("direction","?"),
                    "mape":   _s(r.get("metrics",{}).get("MAPE")),
                }
            elif k == "ARIMA":
                fc  = r.get("forecast", pd.Series(dtype=float))
                cil = r.get("ci_lower", pd.Series(dtype=float))
                ciu = r.get("ci_upper", pd.Series(dtype=float))
                hist = df["Close"].tail(90)
                # fc_30d is forecast last value, next is forecast first value (tomorrow)
                fc_list  = [round(_s(x),4) for x in fc.tolist()]  if len(fc)  else []
                cil_list = [round(_s(x),4) for x in cil.tolist()] if len(cil) else []
                ciu_list = [round(_s(x),4) for x in ciu.tolist()] if len(ciu) else []
                ml_out[k] = {
                    "hist_dates": [str(x) for x in hist.index.tolist()],
                    "hist_actual":[round(_s(x),4) for x in hist.tolist()],
                    "fc_dates":   [str(x) for x in fc.index.tolist()],
                    "fc":         fc_list,
                    # ★ ci_l first, ci_u second — order guaranteed for fill=tonexty
                    "ci_l":       cil_list,
                    "ci_u":       ciu_list,
                    "next":   _s(fc_list[0] if fc_list else None),   # Tomorrow's forecast = first day
                    "fc_30d": _s(r.get("forecast_30d")),              # 30-day forecast
                    "ret30":  _s(r.get("expected_return_30d")),
                    "dir":    r.get("direction","?"),
                    "mape":   _s(r.get("metrics",{}).get("MAPE")),
                    "order":  str(r.get("order","?")),
                }
        except Exception as e:
            ml_out[k] = {"_err": f"Serialization failed: {e}"}

    comp = []
    try:
        cdf = compare_strategies(strats)
        if not cdf.empty:
            comp = cdf.reset_index().fillna(0).to_dict("records")
    except Exception:
        pass

    return {
        "info":  info, "last": round(last,4), "chg": round(chg,2),
        "ohlcv": ohlcv,
        "sig":   {"mom": _sig(mom), "boll": _sig(boll), "rsi": _sig(rsi_s)},
        "met":   {"mom": mom.get("metrics",{}), "boll": boll.get("metrics",{}),
                  "rsi": rsi_s.get("metrics",{})},
        "cum":   {"mom": _cum(mom), "boll": _cum(boll), "rsi": _cum(rsi_s)},
        "comp":  comp,
        "ml":    ml_out,
    }


def _build_sb_stats(store):
    if not store:
        return _sb_empty()
    vals = {"ret":[],"sh":[],"dd":[],"wr":[],"pf":[]}
    for d in store.values():
        for sk in ["mom","boll","rsi"]:
            m = d.get("met",{}).get(sk,{})
            if m:
                vals["ret"].append(_s(m.get("total_return")))
                vals["sh"].append( _s(m.get("sharpe_ratio")))
                vals["dd"].append( _s(m.get("mdd")))
                vals["wr"].append( _s(m.get("win_rate")))
                vals["pf"].append( _s(m.get("profit_factor")))
    def avg(l): return sum(l)/len(l) if l else 0
    r = avg(vals["ret"])
    return html.Div([
        _sb_item("TOT RET",   f"{r:+.1f}%",          C["green"] if r>=0 else C["red"]),
        _sb_item("SHARPE",    f"{avg(vals['sh']):.3f}", C["cyan"]),
        _sb_item("MAX DD",    f"{avg(vals['dd']):.1f}%",C["red"]),
        _sb_item("WIN RATE",  f"{avg(vals['wr']):.1f}%",C["yellow"]),
        _sb_item("PROF.F",    f"{avg(vals['pf']):.2f}", C["orange"]),
    ])

def _sb_empty():
    return html.Div([
        _sb_item(l,v,c) for l,v,c in [
            ("TOT RET","+0.0%",C["green"]),("SHARPE","0.000",C["cyan"]),
            ("MAX DD","0.0%",C["red"]),("WIN RATE","0.0%",C["yellow"]),
            ("PROF.F","0.00",C["orange"])]
    ])


# ─────────────────────────────────────────────────────────────────
# Callback: Content routing
# ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("content-area", "children"),
    Input("active-tab",    "data"),
    Input("results-store", "data"),
    State("log-store",     "data"),
    prevent_initial_call=False,
)
def route(tab, store, logs):
    try:
        if not store or not isinstance(store, dict) or not store:
            return _welcome()
        tickers = [k for k in store if store.get(k)]
        if not tickers:
            return _welcome()
        tab = tab or "OVERVIEW"
        if tab == "OVERVIEW":  return _tab_overview(store, tickers)
        if tab == "CHART":     return _tab_chart(store, tickers)
        if tab == "FORECAST":  return _tab_forecast(store, tickers)
        if tab == "UNIVERSE":  return _tab_universe(store, tickers)
        if tab == "RISK":      return _tab_risk(store, tickers)
        if tab == "PORTFOLIO": return _tab_portfolio(store, tickers)
        if tab == "LIVE LOG":  return _tab_log(logs or [])
        return _tab_overview(store, tickers)
    except Exception as e:
        return html.Div([
            html.Div("⚠  Rendering Error", style={"color":C["yellow"],"fontSize":"13px",
                "letterSpacing":"2px","marginBottom":"8px"}),
            html.Div(f"{type(e).__name__}: {e}",
                style={"color":C["red"],"fontSize":"10px","fontFamily":F}),
            html.Div("Please press RUN again to retry.",
                style={"color":C["gray"],"fontSize":"10px","marginTop":"8px"}),
        ], style={"padding":"40px 20px","textAlign":"center"})


def _welcome():
    return html.Div([
        html.Div("trading", style={"color":C["green"],"fontSize":"18px",
            "letterSpacing":"4px","marginBottom":"12px"}),
        html.Div("Enter a ticker and press ▶ RUN to start analysis.",
            style={"color":C["gray"],"fontSize":"11px","marginBottom":"6px"}),
        html.Div("Examples:  AAPL  ·  MSFT  ·  005930.KS  ·  BTC-USD",
            style={"color":C["dim"],"fontSize":"10px","marginBottom":"16px"}),
        html.Div([
            html.Span("⚠", style={"color":C["yellow"],"marginRight":"6px"}),
            html.Span(f"You can add up to {MAX_TICKERS} tickers.",
                style={"color":C["yellow"]}),
        ], style={"display":"inline-flex","alignItems":"center",
                  "border":f"1px solid {C['border2']}","padding":"6px 16px",
                  "fontSize":"10px","fontFamily":F,"letterSpacing":"1px"}),
    ], style={"padding":"60px 20px","textAlign":"center"})


# ─────────────────────────────────────────────────────────────────
# OVERVIEW Tab
# ─────────────────────────────────────────────────────────────────
def _tab_overview(store, tickers):
    blocks = []
    for tk in tickers:
        d    = store[tk]
        info = d.get("info",{})
        last = _s(d.get("last"))
        chg  = _s(d.get("chg"))
        cc   = C["green"] if chg>=0 else C["red"]

        sig_rows = []
        for sk, sn, sc in [("mom","MOMENTUM",C["mom"]),("boll","BOLLINGER",C["boll"]),
                            ("rsi","RSI REV.",C["rsi_c"])]:
            cur = d.get("sig",{}).get(sk,{}).get("cur",0)
            st  = "▲ BUY" if cur==1 else ("▼ SELL" if cur==-1 else "─ HOLD")
            stc = C["green"] if cur==1 else (C["red"] if cur==-1 else C["gray"])
            m   = d.get("met",{}).get(sk,{})
            sig_rows.append(html.Div([
                html.Span(sn, style={"color":sc,"fontSize":"10px","width":"80px","display":"inline-block"}),
                html.Span(st, style={"color":stc,"fontSize":"10px","fontWeight":"bold",
                    "width":"60px","display":"inline-block"}),
                html.Span(f"Ret {_s(m.get('total_return')):+.1f}%",
                    style={"color":C["gray"],"fontSize":"9px","marginRight":"8px"}),
                html.Span(f"Sh {_s(m.get('sharpe_ratio')):.2f}",
                    style={"color":C["gray"],"fontSize":"9px","marginRight":"8px"}),
                html.Span(f"MDD {_s(m.get('mdd')):.1f}%",
                    style={"color":C["gray"],"fontSize":"9px"}),
            ], style={"display":"flex","alignItems":"center","padding":"4px 8px",
                      "marginBottom":"2px","backgroundColor":C["bg"],
                      "border":f"1px solid {C['border2']}","borderLeft":f"3px solid {sc}"}))

        ml_rows = []
        for mk, mc in [("LSTM",C["lstm_c"]),("GRU",C["gru_c"]),("ARIMA",C["arima_c"])]:
            mr = d.get("ml",{}).get(mk)
            if not mr or "_err" in mr: continue
            r30 = _s(mr.get("ret30"))
            dc  = C["green"] if r30>=0 else C["red"]
            ml_rows.append(html.Div([
                html.Span(mk, style={"color":mc,"fontSize":"10px","width":"50px","display":"inline-block"}),
                html.Span(f"Tomorrow {_s(mr.get('next')):,.2f}",
                    style={"color":C["white"],"fontSize":"10px","marginRight":"10px"}),
                html.Span(f"30D {r30:+.1f}%",
                    style={"color":dc,"fontSize":"10px","fontWeight":"bold","marginRight":"10px"}),
                html.Span(f"MAPE {_s(mr.get('mape')):.1f}%",
                    style={"color":C["gray"],"fontSize":"9px"}),
            ], style={"display":"flex","alignItems":"center","padding":"4px 8px",
                      "marginBottom":"2px","backgroundColor":C["bg"],
                      "border":f"1px solid {C['border2']}","borderLeft":f"3px solid {mc}"}))

        blocks.append(_card([
            html.Div([
                html.Span(f"{tk}  ",style={"color":C["green"],"fontSize":"16px","fontWeight":"bold"}),
                html.Span(info.get("name","")[:40],style={"color":C["gray"],"fontSize":"11px"}),
            ], style={"marginBottom":"8px"}),
            html.Div([
                html.Div([_lbl("LAST"),  html.Br(), _val(f"{last:,.4f}", cc)],  style={"marginRight":"20px"}),
                html.Div([_lbl("CHG"),   html.Br(), _val(f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%", cc)], style={"marginRight":"20px"}),
                html.Div([_lbl("SECTOR"),html.Br(), _val(info.get("sector","N/A")[:14], C["gray"])],  style={"marginRight":"20px"}),
                html.Div([_lbl("MCAP"),  html.Br(), _val(_fv(info.get("market_cap",0)),  C["cyan"])]),
            ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),
            html.Div([
                html.Div([
                    html.Div("STRATEGY SIGNALS",style={"color":C["gray"],"fontSize":"9px",
                        "letterSpacing":"2px","marginBottom":"6px"}),
                    *sig_rows,
                ], style={"flex":"1","minWidth":0,"marginRight":"8px"}),
                html.Div([
                    html.Div("ML FORECAST",style={"color":C["gray"],"fontSize":"9px",
                        "letterSpacing":"2px","marginBottom":"6px"}),
                    *(ml_rows if ml_rows else [html.Div("No ML — select a model and RUN",
                        style={"color":C["dim"],"fontSize":"9px"})]),
                ], style={"flex":"1","minWidth":0}),
            ], style={"display":"flex"}),
        ], tc=C["green"]))

    return html.Div(blocks)


# ─────────────────────────────────────────────────────────────────
# CHART Tab
# ─────────────────────────────────────────────────────────────────
def _tab_chart(store, tickers):
    blocks = []
    for tk in tickers:
        d     = store[tk]
        ohlcv = d["ohlcv"]
        dates = ohlcv["dates"]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.60,0.20,0.20], vertical_spacing=0.02)

        fig.add_trace(go.Candlestick(
            x=dates, open=ohlcv["open"], high=ohlcv["high"],
            low=ohlcv["low"],  close=ohlcv["close"], name="OHLC",
            increasing=dict(line=dict(color=C["green"],width=1),fillcolor="#001a0d"),
            decreasing=dict(line=dict(color=C["red"],  width=1),fillcolor="#1a0000"),
        ), row=1, col=1)

        for key,lbl,clr in [("ma20","MA20","#ffcc00"),("ma50","MA50","#ff8800"),
                             ("ma200","MA200","#9966ff")]:
            if ohlcv.get(key):
                fig.add_trace(go.Scatter(x=dates,y=ohlcv[key],name=lbl,
                    line=dict(color=clr,width=1),opacity=0.8),row=1,col=1)

        if ohlcv.get("bb_u") and ohlcv.get("bb_l"):
            fig.add_trace(go.Scatter(x=dates,y=ohlcv["bb_u"],name="BB+",
                line=dict(color=C["boll"],width=0.7,dash="dot"),opacity=0.4),row=1,col=1)
            fig.add_trace(go.Scatter(x=dates,y=ohlcv["bb_l"],name="BB-",
                line=dict(color=C["boll"],width=0.7,dash="dot"),opacity=0.4,
                fill="tonexty",fillcolor="rgba(255,136,0,0.04)"),row=1,col=1)

        cmap = {str(dt):cl for dt,cl in zip(dates,ohlcv["close"])}
        for sk,sym,clr,sz,lbl in [("mom","triangle-up",C["mom"],10,"MOM"),
                                   ("boll","diamond",C["boll"],8,"BOLL"),
                                   ("rsi","circle",C["rsi_c"],8,"RSI")]:
            buys = d["sig"][sk]["buys"]
            if buys:
                pts = [(b, cmap.get(str(b))) for b in buys]
                pts = [(b,p*0.990) for b,p in pts if p]
                if pts:
                    bd,pl = zip(*pts)
                    fig.add_trace(go.Scatter(x=list(bd),y=list(pl),mode="markers",name=lbl+" BUY",
                        marker=dict(symbol=sym,color=clr,size=sz,
                                    line=dict(color=C["bg"],width=0.5))),row=1,col=1)

        vcols = [C["green2"] if ohlcv["close"][i]>=ohlcv["open"][i] else C["red2"]
                 for i in range(len(dates))]
        fig.add_trace(go.Bar(x=dates,y=ohlcv["volume"],name="VOL",
            marker_color=vcols,opacity=0.5),row=2,col=1)
        if ohlcv.get("vol_ma"):
            fig.add_trace(go.Scatter(x=dates,y=ohlcv["vol_ma"],name="VOL MA",
                line=dict(color=C["yellow"],width=1)),row=2,col=1)

        if ohlcv.get("rsi"):
            fig.add_trace(go.Scatter(x=dates,y=ohlcv["rsi"],name="RSI",
                line=dict(color=C["rsi_c"],width=1.3)),row=3,col=1)
            for lv,cl in [(70,C["red"]),(50,C["dim"]),(30,C["green"])]:
                fig.add_hline(y=lv,line_dash="dot",line_color=cl,opacity=0.4,row=3,col=1)

        _tl(fig, f"{tk}  ·  ALL STRATEGIES + INDICATORS", h=560)
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Cumulative return comparison
        fig_c = go.Figure()
        for sk,sn,sc in [("mom","MOMENTUM",C["mom"]),("boll","BOLLINGER",C["boll"]),
                         ("rsi","RSI REV.",C["rsi_c"])]:
            c = d["cum"][sk]
            if c["dates"]:
                fig_c.add_trace(go.Scatter(x=c["dates"],y=c["strat"],name=sn,
                    line=dict(color=sc,width=1.5)))
        if d["cum"]["mom"]["dates"]:
            fig_c.add_trace(go.Scatter(x=d["cum"]["mom"]["dates"],y=d["cum"]["mom"]["bh"],
                name="BUY&HOLD",line=dict(color=C["gray"],width=1,dash="dot")))
        fig_c.add_hline(y=0,line_color=C["border2"],line_dash="dot")
        _tl(fig_c,"Cumulative Return by Strategy (%)",240)

        # Comparison table
        tbl = html.Div()
        if d.get("comp"):
            cdf = pd.DataFrame(d["comp"])
            rn  = {"strategy":"Strategy","total_return":"Return%","cagr":"CAGR%",
                   "sharpe_ratio":"SHARPE","mdd":"MDD%","win_rate":"Win Rate%","profit_factor":"Prof.Factor"}
            cdf.rename(columns=rn, inplace=True)
            show = [c for c in rn.values() if c in cdf.columns]
            tbl = dash_table.DataTable(
                data=cdf[show].to_dict("records"),
                columns=[{"name":c,"id":c} for c in show],
                style_header={"backgroundColor":C["bg"],"color":C["cyan"],"fontFamily":F,
                    "fontSize":"9px","fontWeight":"bold","border":f"1px solid {C['border2']}",
                    "textAlign":"center"},
                style_cell={"backgroundColor":C["panel"],"color":C["white"],"fontFamily":F,
                    "fontSize":"10px","border":f"1px solid {C['border']}",
                    "textAlign":"center","padding":"5px 6px"},
            )

        blocks.append(html.Div([
            html.Div(f"[ {tk} ]", style={"color":C["green"],"fontSize":"11px",
                "letterSpacing":"2px","marginBottom":"6px","fontWeight":"bold"}),
            _card(dcc.Graph(figure=fig, config={"displayModeBar":False}), style={"padding":"0"}),
            html.Div([
                html.Div([_card(dcc.Graph(figure=fig_c,config={"displayModeBar":False}),
                    style={"padding":"0"})],style={"flex":"1.3","minWidth":0}),
                html.Div([_card(tbl,style={"padding":"8px"})],style={"flex":"1","minWidth":0}),
            ], style={"display":"flex","gap":"8px"}),
        ], style={"marginBottom":"16px"}))

    return html.Div(blocks)


# ─────────────────────────────────────────────────────────────────
# FORECAST Tab ★ Key fix: 60-day history + TODAY boundary + 30-day future + 95% CI
# ─────────────────────────────────────────────────────────────────
def _tab_forecast(store, tickers):
    """
    Design principles:
    - Past (gray): Last 60 days actual price — provides context
    - TODAY vertical line: Clear past/future separation (cyan dashed)
    - Future (colored): 30-day forecast center line + 95% CI band (fill=tonexty)
    - ARIMA: ci_l trace → ci_u trace (fill=tonexty) order strictly guaranteed
    - Right summary panel: Numeric forecast values
    """
    blocks = []
    for tk in tickers:
        d    = store[tk]
        ml   = d.get("ml", {})
        last = _s(d.get("last"))

        blocks.append(html.Div([
            html.Span(f"[ {tk} ]", style={"color":C["green"],"fontSize":"13px",
                "fontWeight":"bold","marginRight":"12px"}),
            html.Span("ML PRICE FORECAST",
                style={"color":C["cyan"],"fontSize":"11px","letterSpacing":"3px"}),
            html.Span("  ·  FUTURE 30D  +  95% CI",
                style={"color":C["gray"],"fontSize":"9px"}),
        ], style={"marginBottom":"10px","borderBottom":f"1px solid {C['border']}",
                  "paddingBottom":"6px"}))

        if not ml:
            blocks.append(html.Div("⚠  No ML model — select a model and press RUN.",
                style={"color":C["yellow"],"fontSize":"11px","padding":"20px 0"}))
            continue

        model_cards, table_rows = [], []

        for mk, mc in [("LSTM",C["lstm_c"]),("GRU",C["gru_c"]),("ARIMA",C["arima_c"])]:
            if mk not in ml:
                continue
            mr = ml[mk]

            # Error card
            if "_err" in mr:
                model_cards.append(html.Div([
                    html.Div(mk, style={"color":mc,"fontSize":"11px",
                        "fontWeight":"bold","marginBottom":"8px"}),
                    html.Div("⚠  Forecast unavailable",
                        style={"color":C["yellow"],"fontSize":"12px","marginBottom":"4px"}),
                    html.Div(mr["_err"],
                        style={"color":C["gray"],"fontSize":"10px","fontFamily":F}),
                ], style={"flex":"1","minWidth":0,"padding":"20px",
                          "backgroundColor":C["panel"],"border":f"1px solid {C['border']}",
                          "borderTop":f"2px solid {mc}"}))
                continue

            # Data
            hist_dates  = mr.get("hist_dates",  [])
            hist_actual = mr.get("hist_actual", [])
            hist_pred   = mr.get("hist_pred",   [])
            fc_dates    = mr.get("fc_dates",    [])
            fc          = mr.get("fc",          [])
            ci_l        = mr.get("ci_l",        [])
            ci_u        = mr.get("ci_u",        [])
            ret30       = _s(mr.get("ret30"))
            fc_30d      = _s(mr.get("fc_30d"))
            nxt         = _s(mr.get("next"))
            mape        = _s(mr.get("mape"))
            dc          = C["green"] if ret30>=0 else C["red"]
            ds          = "▲" if ret30>=0 else "▼"

            # Last 60 days as historical context
            N = 60
            ctx_d = hist_dates[-N:]  if len(hist_dates)>N  else hist_dates
            ctx_a = hist_actual[-N:] if len(hist_actual)>N else hist_actual
            ctx_p = hist_pred[-N:]   if len(hist_pred)>N   else hist_pred

            # y range — completely remove None/NaN
            vy = [v for v in ctx_a + fc + ci_l + ci_u
                  if v is not None and isinstance(v,(int,float)) and v==v]
            ymin = min(vy)*0.994 if vy else 0
            ymax = max(vy)*1.006 if vy else 1
            if ymin >= ymax: ymin, ymax = ymin*0.99, ymax*1.01

            fig = go.Figure()

            # ① Historical actual price (gray)
            if ctx_d and ctx_a:
                fig.add_trace(go.Scatter(
                    x=ctx_d, y=ctx_a, name="Actual (60D)",
                    mode="lines", line=dict(color="#777",width=1.5)
                ))

            # ② Historical fit (LSTM/GRU only)
            if mk in ("LSTM","GRU") and ctx_d and ctx_p:
                fig.add_trace(go.Scatter(
                    x=ctx_d, y=ctx_p, name=f"{mk} FIT",
                    mode="lines", line=dict(color=mc,width=1,dash="dot"), opacity=0.5
                ))

            # ③ TODAY boundary line
            today_x = ctx_d[-1] if ctx_d else None
            if today_x:
                fig.add_shape(type="line",
                    x0=today_x, x1=today_x, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color=C["cyan"],width=1.5,dash="dot"))
                fig.add_annotation(
                    x=today_x, y=0.97, xref="x", yref="paper",
                    text="◀ Past  │  Future ▶",
                    showarrow=False, xanchor="center",
                    font=dict(color=C["cyan"],size=8,family=F),
                    bgcolor="rgba(0,0,0,0.85)", borderpad=3)

            # ④ 95% CI band — ci_l first, ci_u with fill=tonexty
            #    Same logic for ARIMA/LSTM/GRU
            if fc_dates and ci_l and ci_u and len(ci_l)==len(fc_dates) and len(ci_u)==len(fc_dates):
                fig.add_trace(go.Scatter(
                    x=fc_dates, y=ci_l,
                    name="CI Lower", mode="lines",
                    line=dict(color=mc,width=0),
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=fc_dates, y=ci_u,
                    name="95% CI", mode="lines",
                    fill="tonexty",
                    fillcolor=f"rgba({_rgb(mc)},0.14)",
                    line=dict(color=mc,width=0.8,dash="dot"),
                ))

            # ⑤ Forecast center line
            if fc_dates and fc:
                fig.add_trace(go.Scatter(
                    x=fc_dates, y=fc,
                    name=f"{mk} 30D Forecast",
                    mode="lines+markers",
                    line=dict(color=mc,width=2.5),
                    marker=dict(size=3,color=mc),
                ))
                # Final value annotation
                last_fc = [v for v in fc if v is not None]
                if last_fc:
                    fig.add_annotation(
                        x=fc_dates[-1], y=_s(last_fc[-1]),
                        xref="x", yref="y",
                        text=f"  {_s(last_fc[-1]):,.2f}",
                        showarrow=False, xanchor="left",
                        font=dict(color=dc,size=11,family=F),
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor=mc, borderwidth=1)

            # Layout
            ml_lbl = f"ARIMA{mr.get('order','')}" if mk=="ARIMA" else mk
            all_x  = list(ctx_d) + list(fc_dates)
            _tl(fig, f"{ml_lbl}  ·  30D FORECAST  +  95% CI", h=320)
            fig.update_layout(
                yaxis=dict(range=[ymin,ymax]),
                xaxis=dict(range=[all_x[0],all_x[-1]] if all_x else None),
                annotations=list(fig.layout.annotations) + [dict(
                    x=0.01, y=0.04, xref="paper", yref="paper", showarrow=False,
                    text=(f"Tomorrow <b>{nxt:,.2f}</b>"
                          f"  │  30D <b>{fc_30d:,.2f}</b>"
                          f"  <span style='color:{dc}'><b>{ds}{abs(ret30):.2f}%</b></span>"
                          f"  │  MAPE <b>{mape:.1f}%</b>"),
                    font=dict(color=C["gray"],size=10,family=F),
                    bgcolor=C["bg"], bordercolor=mc, borderwidth=1, xanchor="left"
                )],
            )

            # Right summary panel
            ci_end_l = _s(ci_l[-1] if ci_l else 0)
            ci_end_u = _s(ci_u[-1] if ci_u else 0)
            spanel = html.Div([
                html.Div([
                    html.Div(lb, style={"color":C["gray"],"fontSize":"9px",
                        "letterSpacing":"1px","marginBottom":"1px"}),
                    html.Div(vl, style={"color":cl,"fontSize":"12px",
                        "fontWeight":"bold","marginBottom":"8px","fontFamily":F}),
                ]) for lb,vl,cl in [
                    ("MODEL",          ml_lbl,          mc),
                    ("CURRENT",        f"{last:,.2f}",   C["white"]),
                    ("TOMORROW",       f"{nxt:,.2f}",    dc),
                    ("30D FORECAST",   f"{fc_30d:,.2f}", dc),
                    ("30D RETURN",     f"{ds}{abs(ret30):.2f}%", dc),
                    ("CI LOWER",       f"{ci_end_l:,.2f}", C["gray"]),
                    ("CI UPPER",       f"{ci_end_u:,.2f}", C["gray"]),
                    ("MAPE",           f"{mape:.1f}%",    C["yellow"]),
                ]
            ], style={"width":"115px","flexShrink":"0",
                      "backgroundColor":C["bg"],"border":f"1px solid {C['border']}",
                      "borderTop":f"3px solid {mc}","padding":"10px 8px"})

            model_cards.append(html.Div([
                html.Div(dcc.Graph(figure=fig,config={"displayModeBar":False}),
                    style={"flex":"1","minWidth":0}),
                spanel,
            ], style={"display":"flex","flex":"1","minWidth":0,
                      "backgroundColor":C["panel"],"border":f"1px solid {C['border']}",
                      "borderTop":f"2px solid {mc}"}))

            table_rows.append({
                "MODEL":       ml_lbl,
                "Current":     f"{last:,.2f}",
                "Tomorrow":    f"{nxt:,.2f}",
                "30D Forecast":f"{fc_30d:,.2f}",
                "30D Return":  f"{ret30:+.2f}%",
                "CI Lower":    f"{ci_end_l:,.2f}",
                "CI Upper":    f"{ci_end_u:,.2f}",
                "MAPE":        f"{mape:.1f}%",
                "Direction":   "▲ BUY" if ret30>=0 else "▼ SELL",
            })

        if not model_cards:
            blocks.append(html.Div("No data for selected models.",
                style={"color":C["gray"],"fontSize":"11px","padding":"20px 0"}))
        else:
            for i in range(0, len(model_cards), 2):
                blocks.append(html.Div(model_cards[i:i+2],
                    style={"display":"flex","gap":"8px","marginBottom":"8px"}))

        if table_rows:
            fdf = pd.DataFrame(table_rows)
            blocks.append(html.Div([
                html.Div("FORECAST SUMMARY", style={"color":C["gray"],"fontSize":"9px",
                    "letterSpacing":"2px","marginBottom":"8px"}),
                dash_table.DataTable(
                    data=fdf.to_dict("records"),
                    columns=[{"name":c,"id":c} for c in fdf.columns],
                    style_header={"backgroundColor":"#0a0a0a","color":C["cyan"],"fontFamily":F,
                        "fontSize":"9px","fontWeight":"bold","border":f"1px solid {C['border2']}",
                        "textAlign":"center"},
                    style_cell={"backgroundColor":C["panel"],"color":C["white"],"fontFamily":F,
                        "fontSize":"11px","border":f"1px solid {C['border']}",
                        "textAlign":"center","padding":"7px"},
                    style_data_conditional=[
                        {"if":{"filter_query":"{Direction} contains '▲'"},"color":C["green"],"fontWeight":"bold"},
                        {"if":{"filter_query":"{Direction} contains '▼'"},"color":C["red"],"fontWeight":"bold"},
                        {"if":{"column_id":"30D Return","filter_query":"{30D Return} contains '+'"},"color":C["green"]},
                        {"if":{"column_id":"30D Return","filter_query":"{30D Return} contains '-'"},"color":C["red"]},
                    ],
                ),
            ], style={"backgroundColor":C["panel"],"border":f"1px solid {C['border']}",
                      "padding":"10px","marginBottom":"8px"}))

    return html.Div(blocks)


# ─────────────────────────────────────────────────────────────────
# UNIVERSE Tab
# ─────────────────────────────────────────────────────────────────
def _tab_universe(store, tickers):
    rows = []
    for tk in tickers:
        d    = store[tk]
        info = d.get("info",{})
        last = _s(d.get("last"))
        chg  = _s(d.get("chg"))

        best_s, best_sh = "─", 0.0
        for sk in ["mom","boll","rsi"]:
            sh = _s(d.get("met",{}).get(sk,{}).get("sharpe_ratio"))
            if sh > best_sh: best_sh, best_s = sh, sk.upper()

        ml_dirs = [mr.get("dir","?") for mr in d.get("ml",{}).values()
                   if mr and "_err" not in mr]
        up = ml_dirs.count("UP")
        consensus = ("▲ BULLISH" if up>len(ml_dirs)//2 else
                     "▼ BEARISH" if up<len(ml_dirs)//2 else "─ NEUTRAL") if ml_dirs else "─"

        rows.append({"SYMBOL":tk,"Name":info.get("name",tk)[:25],
            "Sector":info.get("sector","N/A")[:18],
            "Price":f"{last:,.4f}","Change":f"{chg:+.2f}%",
            "ML Direction":consensus,"Best Strategy":best_s,"SHARPE":f"{best_sh:.3f}"})

    if not rows:
        return html.Div("No data", style={"color":C["gray"],"padding":"20px"})

    return html.Div([
        html.Div("MARKET UNIVERSE",style={"color":C["gray"],"fontSize":"9px",
            "letterSpacing":"2px","marginBottom":"10px"}),
        dash_table.DataTable(
            data=rows, columns=[{"name":c,"id":c} for c in rows[0]],
            sort_action="native",
            style_table={"overflowX":"auto"},
            style_header={"backgroundColor":C["bg"],"color":C["cyan"],"fontFamily":F,
                "fontSize":"9px","fontWeight":"bold","border":f"1px solid {C['border2']}",
                "textAlign":"center","letterSpacing":"1px"},
            style_cell={"backgroundColor":C["panel"],"color":C["white"],"fontFamily":F,
                "fontSize":"11px","border":f"1px solid {C['border']}",
                "textAlign":"center","padding":"8px"},
            style_data_conditional=[
                {"if":{"column_id":"Change","filter_query":"{Change} contains '+'"},"color":C["green"]},
                {"if":{"column_id":"Change","filter_query":"{Change} contains '-'"},"color":C["red"]},
                {"if":{"column_id":"ML Direction","filter_query":"{ML Direction} contains '▲'"},"color":C["green"],"fontWeight":"bold"},
                {"if":{"column_id":"ML Direction","filter_query":"{ML Direction} contains '▼'"},"color":C["red"],"fontWeight":"bold"},
            ],
        ),
    ], style={"backgroundColor":C["panel"],"border":f"1px solid {C['border']}","padding":"12px"})


# ─────────────────────────────────────────────────────────────────
# RISK Tab
# ─────────────────────────────────────────────────────────────────
def _tab_risk(store, tickers):
    blocks = []
    for tk in tickers:
        d     = store[tk]
        ohlcv = d["ohlcv"]
        rets  = pd.Series(ohlcv["close"], index=ohlcv["dates"]).pct_change().dropna()

        cum     = (1+rets).cumprod()
        rollmax = cum.cummax()
        dd      = (cum - rollmax) / rollmax * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index,y=dd.values,name="Drawdown",
            fill="tozeroy",fillcolor="rgba(255,51,51,0.15)",
            line=dict(color=C["red"],width=1)))
        fig_dd.add_hline(y=0,line_color=C["gray"],opacity=0.3)
        _tl(fig_dd, f"{tk}  ·  DRAWDOWN (%)", 220)

        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=rets.values*100,nbinsx=60,
            name="Daily Ret",marker_color=C["cyan"],opacity=0.7))
        fig_h.add_vline(x=0,line_color=C["gray"],opacity=0.5)
        _tl(fig_h, f"{tk}  ·  RETURN DISTRIBUTION", 220)

        rsh = rets.rolling(60).apply(
            lambda x: (x.mean()*252)/(x.std()*np.sqrt(252)+1e-9), raw=True)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=rsh.index,y=rsh.values,name="Rolling Sharpe(60D)",
            line=dict(color=C["yellow"],width=1.3)))
        fig_rs.add_hline(y=1,line_color=C["green"],line_dash="dot",opacity=0.5)
        fig_rs.add_hline(y=0,line_color=C["gray"],opacity=0.3)
        _tl(fig_rs, f"{tk}  ·  ROLLING SHARPE (60D)", 220)

        var95  = float(np.percentile(rets,5))*100
        cvar95 = float(rets[rets<=np.percentile(rets,5)].mean())*100
        vol    = float(rets.std()*np.sqrt(252))*100

        items = [
            ("VaR 95%", f"{var95:.2f}%", C["red"]),
            ("CVaR 95%",f"{cvar95:.2f}%",C["red"]),
            ("ANN.VOL",  f"{vol:.2f}%",  C["yellow"]),
            ("MAX DD",   f"{float(dd.min()):.2f}%",C["red"]),
            ("SKEWNESS", f"{float(rets.skew()):.3f}",C["cyan"]),
            ("KURTOSIS", f"{float(rets.kurtosis()):.3f}",C["purple"]),
        ]
        blocks.append(html.Div([
            html.Div(f"[ {tk} ]  RISK ANALYTICS",style={"color":C["cyan"],"fontSize":"11px",
                "letterSpacing":"2px","marginBottom":"8px",
                "borderBottom":f"1px solid {C['border']}","paddingBottom":"5px"}),
            html.Div([html.Div([_lbl(l),html.Br(),_val(v,c)],
                style={"textAlign":"center","padding":"8px 12px","flex":"1",
                       "backgroundColor":C["panel"],"border":f"1px solid {C['border2']}"})
                for l,v,c in items],
                style={"display":"flex","gap":"4px","marginBottom":"8px"}),
            html.Div([
                html.Div([_card(dcc.Graph(figure=fig_dd,config={"displayModeBar":False}),
                    style={"padding":"0"})],style={"flex":"1","minWidth":0}),
                html.Div([_card(dcc.Graph(figure=fig_h, config={"displayModeBar":False}),
                    style={"padding":"0"})],style={"flex":"1","minWidth":0}),
                html.Div([_card(dcc.Graph(figure=fig_rs,config={"displayModeBar":False}),
                    style={"padding":"0"})],style={"flex":"1","minWidth":0}),
            ],style={"display":"flex","gap":"8px"}),
        ],style={"marginBottom":"16px"}))

    return html.Div(blocks)


# ─────────────────────────────────────────────────────────────────
# PORTFOLIO Tab
# ─────────────────────────────────────────────────────────────────
def _tab_portfolio(store, tickers):
    """
    PORTFOLIO Tab — 3 sections
      1. Correlation heatmap    ← Understand diversification effects between tickers
      2. Efficient frontier     ← Visualize optimal risk-return combinations
      3. Weight optimization    ← Compare Equal / Min Vol / Max Sharpe 3 methods
    """

    # ── Common data: daily returns per ticker ────────────────
    returns_dict = {}
    for tk in tickers:
        ohlcv = store[tk].get("ohlcv", {})
        closes = ohlcv.get("close", [])
        dates  = ohlcv.get("dates", [])
        if len(closes) < 30:
            continue
        s = pd.Series(closes, index=pd.to_datetime(dates))
        returns_dict[tk] = s.pct_change().dropna()

    valid_tickers = list(returns_dict.keys())
    multi = len(valid_tickers) >= 2

    # ── Header ───────────────────────────────────────────────
    header = html.Div([
        html.Span("PORTFOLIO OPTIMIZER",
            style={"color": C["cyan"], "fontSize": "11px",
                   "letterSpacing": "3px", "marginRight": "12px"}),
        html.Span(f"  {len(valid_tickers)} tickers",
            style={"color": C["gray"], "fontSize": "9px"}),
    ], style={"marginBottom": "12px",
              "borderBottom": f"1px solid {C['border']}",
              "paddingBottom": "6px"})

    blocks = [header]

    # ════════════════════════════════════════════════════════
    # SECTION 1 — Correlation Matrix
    # ════════════════════════════════════════════════════════
    blocks.append(html.Div("① CORRELATION MATRIX",
        style={"color": C["gray"], "fontSize": "9px",
               "letterSpacing": "2px", "marginBottom": "6px"}))

    if not multi:
        blocks.append(html.Div(
            "⚠  Correlation analysis requires at least 2 tickers. Please add more.",
            style={"color": C["yellow"], "fontSize": "10px",
                   "padding": "16px 0", "marginBottom": "12px"}))
    else:
        ret_df   = pd.DataFrame(returns_dict).dropna()
        corr     = ret_df.corr()
        tks      = corr.columns.tolist()
        z        = corr.values.tolist()
        z_text   = [[f"{v:.2f}" for v in row] for row in z]

        # Colors: -1(red) → 0(black) → +1(green)
        fig_corr = go.Figure(go.Heatmap(
            z=z, x=tks, y=tks, text=z_text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#aa2222"],
                [0.5, "#111111"],
                [1.0, "#00aa55"],
            ],
            zmin=-1, zmax=1,
            hoverongaps=False,
            hovertemplate="%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        ))
        _tl(fig_corr, "Return Correlation Between Tickers (1Y Daily)", h=max(260, 80 * len(tks)))
        fig_corr.update_layout(
            margin=dict(l=80, r=20, t=36, b=60),
        )
        fig_corr.update_xaxes(tickfont=dict(size=11, color=C["white"]))
        fig_corr.update_yaxes(tickfont=dict(size=11, color=C["white"]))

        # Diversification summary text
        low_pairs = []
        for i in range(len(tks)):
            for j in range(i + 1, len(tks)):
                v = corr.iloc[i, j]
                if v < 0.3:
                    emoji = "🟢" if v < 0 else "🟡"
                    low_pairs.append(f"{emoji} {tks[i]}×{tks[j]}  ({v:+.2f})")
        pair_msg = ("  │  ".join(low_pairs)
                    if low_pairs else "No high-diversification pairs found")

        blocks.append(html.Div([
            html.Div(
                dcc.Graph(figure=fig_corr,
                          config={"displayModeBar": False}),
                style={"flex": "1", "minWidth": 0}),
            html.Div([
                html.Div("Best Diversification Pairs", style={"color": C["gray"],
                    "fontSize": "9px", "letterSpacing": "1px",
                    "marginBottom": "6px"}),
                *([html.Div(p, style={"color": C["green"], "fontSize": "10px",
                      "fontFamily": F, "marginBottom": "4px"})
                   for p in (low_pairs or [html.Span("─", style={"color": C["dim"]})])])
            ], style={"width": "180px", "flexShrink": "0",
                      "padding": "12px 10px",
                      "backgroundColor": C["bg"],
                      "border": f"1px solid {C['border']}",
                      "borderTop": f"2px solid {C['cyan']}",
                      "alignSelf": "flex-start",
                      "marginTop": "8px"}),
        ], style={"display": "flex", "gap": "8px",
                  "backgroundColor": C["panel"],
                  "border": f"1px solid {C['border']}",
                  "marginBottom": "12px"}))

    # ════════════════════════════════════════════════════════
    # SECTION 2 — Efficient Frontier
    # ════════════════════════════════════════════════════════
    blocks.append(html.Div("② EFFICIENT FRONTIER",
        style={"color": C["gray"], "fontSize": "9px",
               "letterSpacing": "2px", "marginBottom": "6px"}))

    if not multi:
        blocks.append(html.Div(
            "⚠  Efficient frontier requires at least 2 tickers.",
            style={"color": C["yellow"], "fontSize": "10px",
                   "padding": "16px 0", "marginBottom": "12px"}))
    else:
        ef = efficient_frontier(returns_dict, n_portfolios=500)

        if "error" in ef:
            blocks.append(html.Div(f"⚠  {ef['error']}",
                style={"color": C["yellow"], "fontSize": "10px", "padding": "16px 0"}))
        else:
            ms = ef["special"]["max_sharpe"]
            mv = ef["special"]["min_vol"]

            fig_ef = go.Figure()

            # Random portfolio scatter (Sharpe ratio color)
            fig_ef.add_trace(go.Scatter(
                x=ef["vols"], y=ef["rets"],
                mode="markers",
                name="Portfolios",
                marker=dict(
                    size=4,
                    color=ef["sharpes"],
                    colorscale=[
                        [0.0, "#aa2222"],
                        [0.5, "#ffcc00"],
                        [1.0, "#00ff88"],
                    ],
                    colorbar=dict(
                        title="Sharpe", thickness=10,
                        tickfont=dict(family=F, size=8, color=C["gray"]),
                        titlefont=dict(family=F, size=9, color=C["gray"]),
                    ),
                    showscale=True,
                    opacity=0.6,
                ),
                hovertemplate=(
                    "Volatility: %{x:.1f}%<br>"
                    "Expected Return: %{y:.1f}%<br>"
                    "Sharpe: %{marker.color:.2f}<extra></extra>"
                ),
            ))

            # Max Sharpe point
            fig_ef.add_trace(go.Scatter(
                x=[ms["vol"]], y=[ms["ret"]],
                mode="markers+text",
                name=f"Max Sharpe ({ms['sharpe']:.2f})",
                marker=dict(size=14, color=C["green"],
                            symbol="star", line=dict(width=1, color="#fff")),
                text=["Max Sharpe"], textposition="top right",
                textfont=dict(color=C["green"], size=10, family=F),
            ))

            # Min Vol point
            fig_ef.add_trace(go.Scatter(
                x=[mv["vol"]], y=[mv["ret"]],
                mode="markers+text",
                name=f"Min Vol ({mv['vol']:.1f}%)",
                marker=dict(size=14, color=C["cyan"],
                            symbol="diamond", line=dict(width=1, color="#fff")),
                text=["Min Vol"], textposition="top right",
                textfont=dict(color=C["cyan"], size=10, family=F),
            ))

            # Individual ticker points
            ret_df2 = pd.DataFrame(returns_dict).dropna()
            for tk in valid_tickers:
                tk_ret = ret_df2[tk].mean() * 252 * 100
                tk_vol = ret_df2[tk].std() * np.sqrt(252) * 100
                fig_ef.add_trace(go.Scatter(
                    x=[tk_vol], y=[tk_ret],
                    mode="markers+text",
                    name=tk,
                    marker=dict(size=9, symbol="circle-open",
                                line=dict(width=2, color=C["orange"])),
                    text=[tk], textposition="bottom right",
                    textfont=dict(color=C["orange"], size=9, family=F),
                    showlegend=False,
                ))

            _tl(fig_ef, "Risk × Return  ·  Efficient Frontier (Annualized)", h=380)
            fig_ef.update_xaxes(title_text="Annual Volatility (%)",
                                 title_font=dict(size=9, color=C["gray"]))
            fig_ef.update_yaxes(title_text="Expected Return (%)",
                                 title_font=dict(size=9, color=C["gray"]))

            # Optimal weight summary card
            def _weight_rows(pt_name, pt_color, pt):
                return html.Div([
                    html.Div(pt_name, style={"color": pt_color,
                        "fontSize": "9px", "letterSpacing": "1px",
                        "marginBottom": "4px", "fontWeight": "bold"}),
                    *[html.Div([
                        html.Span(tk, style={"color": C["gray"], "fontSize": "9px"}),
                        html.Span(f"{pt['weights'].get(tk, 0):.1f}%",
                                  style={"color": C["white"], "fontSize": "11px",
                                         "fontWeight": "bold", "float": "right",
                                         "fontFamily": F}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "marginBottom": "2px"})
                      for tk in valid_tickers],
                    html.Div([
                        html.Span("Sharpe", style={"color": C["gray"], "fontSize": "9px"}),
                        html.Span(f"{pt['sharpe']:.2f}",
                                  style={"color": pt_color, "fontSize": "11px",
                                         "fontWeight": "bold", "float": "right"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "marginTop": "4px",
                              "borderTop": f"1px solid {C['border']}",
                              "paddingTop": "4px"}),
                ], style={"marginBottom": "10px",
                          "paddingBottom": "8px",
                          "borderBottom": f"1px solid {C['border2']}"})

            side_ef = html.Div([
                _weight_rows("★ MAX SHARPE", C["green"],  ms),
                _weight_rows("◆ MIN VOL",    C["cyan"],   mv),
            ], style={"width": "160px", "flexShrink": "0",
                      "padding": "12px 10px",
                      "backgroundColor": C["bg"],
                      "border": f"1px solid {C['border']}",
                      "borderTop": f"2px solid {C['green']}",
                      "alignSelf": "flex-start",
                      "marginTop": "8px"})

            blocks.append(html.Div([
                html.Div(dcc.Graph(figure=fig_ef,
                                   config={"displayModeBar": False}),
                         style={"flex": "1", "minWidth": 0}),
                side_ef,
            ], style={"display": "flex", "gap": "8px",
                      "backgroundColor": C["panel"],
                      "border": f"1px solid {C['border']}",
                      "marginBottom": "12px"}))

    # ════════════════════════════════════════════════════════
    # SECTION 3 — Weight Optimization 3-Method Comparison
    # ════════════════════════════════════════════════════════
    blocks.append(html.Div("③ WEIGHT OPTIMIZATION  ·  3-METHOD COMPARISON",
        style={"color": C["gray"], "fontSize": "9px",
               "letterSpacing": "2px", "marginBottom": "6px"}))

    if not multi:
        blocks.append(html.Div(
            "⚠  At least 2 tickers are required.",
            style={"color": C["yellow"], "fontSize": "10px", "padding": "16px 0"}))
        return html.Div(blocks, style={"padding": "14px 16px"})

    method_labels = {
        "equal_weight": ("EQUAL WEIGHT", C["gray"]),
        "min_variance": ("MIN VARIANCE", C["cyan"]),
        "max_sharpe":   ("MAX SHARPE",   C["green"]),
    }
    method_results = {}
    for m in ["equal_weight", "min_variance", "max_sharpe"]:
        try:
            method_results[m] = build_portfolio(returns_dict, method=m)
        except Exception as e:
            method_results[m] = {"error": str(e)}

    # ── Weight comparison bar chart ──────────────────────────
    fig_bar = go.Figure()
    bar_colors = [C["lstm_c"], C["gru_c"], C["arima_c"],
                  C["green"], C["orange"], C["purple"],
                  C["cyan"], C["yellow"], C["red"]]

    for i, tk in enumerate(valid_tickers):
        weights_by_method = []
        method_names      = []
        for m, (lbl, _) in method_labels.items():
            r = method_results[m]
            w = r.get("weights", {}).get(tk, 0) if "error" not in r else 0
            weights_by_method.append(w)
            method_names.append(lbl)

        fig_bar.add_trace(go.Bar(
            name=tk,
            x=method_names,
            y=weights_by_method,
            marker_color=bar_colors[i % len(bar_colors)],
            text=[f"{w:.1f}%" for w in weights_by_method],
            textposition="inside",
            textfont=dict(size=9, family=F, color="#000"),
        ))

    fig_bar.update_layout(
        barmode="stack",
        paper_bgcolor=C["panel"], plot_bgcolor=C["bg"],
        height=240,
        margin=dict(l=44, r=10, t=28, b=40),
        font=dict(color=C["white"], family=F, size=10),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(family=F, size=9, color=C["gray"]),
                    orientation="h", y=-0.25, x=0.5, xanchor="center"),
        title=dict(text="Weight Allocation Comparison (3 Methods)",
                   x=0.01, xanchor="left",
                   font=dict(color=C["gray"], size=10, family=F)),
        yaxis=dict(ticksuffix="%", **_ax()),
        xaxis=_ax(),
        hovermode="x unified",
    )

    # ── Portfolio performance comparison table ───────────────
    perf_rows = []
    for m, (lbl, mc) in method_labels.items():
        r = method_results[m]
        if "error" in r:
            continue
        pm = r.get("portfolio_metrics", {})
        perf_rows.append({
            "METHOD":         lbl,
            "Exp.Return(Ann)": f"{_s(pm.get('expected_return')):+.2f}%",
            "Volatility(Ann)": f"{_s(pm.get('volatility')):.2f}%",
            "SHARPE":         f"{_s(pm.get('sharpe_ratio')):.3f}",
            "MDD":            f"{_s(pm.get('mdd')):.2f}%",
        })

    # ── Ticker weight detail table ───────────────────────────
    detail_rows = []
    for tk in valid_tickers:
        row = {"TICKER": tk}
        for m, (lbl, _) in method_labels.items():
            r = method_results[m]
            w = r.get("weights", {}).get(tk, 0) if "error" not in r else 0
            row[lbl] = f"{w:.1f}%"
        detail_rows.append(row)

    tbl_style_hdr = {
        "backgroundColor": C["bg"], "color": C["cyan"], "fontFamily": F,
        "fontSize": "9px", "fontWeight": "bold",
        "border": f"1px solid {C['border2']}",
        "textAlign": "center", "letterSpacing": "1px",
    }
    tbl_style_cell = {
        "backgroundColor": C["panel"], "color": C["white"], "fontFamily": F,
        "fontSize": "11px", "border": f"1px solid {C['border']}",
        "textAlign": "center", "padding": "7px",
    }

    blocks.append(html.Div([
        # Bar chart
        html.Div([
            dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
        ], style={"backgroundColor": C["panel"],
                  "border": f"1px solid {C['border']}",
                  "marginBottom": "8px"}),

        # Performance comparison + Weight detail (side by side)
        html.Div([
            # Performance table
            html.Div([
                html.Div("Portfolio Performance Comparison",
                         style={"color": C["gray"], "fontSize": "9px",
                                "letterSpacing": "1px", "marginBottom": "6px"}),
                dash_table.DataTable(
                    data=perf_rows,
                    columns=[{"name": c, "id": c}
                             for c in ["METHOD", "Exp.Return(Ann)", "Volatility(Ann)", "SHARPE", "MDD"]],
                    style_table={"overflowX": "auto"},
                    style_header=tbl_style_hdr,
                    style_cell=tbl_style_cell,
                    style_data_conditional=[
                        {"if": {"column_id": "METHOD",
                                "filter_query": '{METHOD} = "MAX SHARPE"'},
                         "color": C["green"], "fontWeight": "bold"},
                        {"if": {"column_id": "METHOD",
                                "filter_query": '{METHOD} = "MIN VARIANCE"'},
                         "color": C["cyan"], "fontWeight": "bold"},
                        {"if": {"column_id": "SHARPE"},
                         "color": C["yellow"]},
                        {"if": {"column_id": "Exp.Return(Ann)",
                                "filter_query": '{Exp.Return(Ann)} contains "+"'},
                         "color": C["green"]},
                        {"if": {"column_id": "Exp.Return(Ann)",
                                "filter_query": '{Exp.Return(Ann)} contains "-"'},
                         "color": C["red"]},
                    ],
                ),
            ], style={"flex": "1", "minWidth": 0,
                      "backgroundColor": C["panel"],
                      "border": f"1px solid {C['border']}",
                      "padding": "10px"}),

            # Weight detail table
            html.Div([
                html.Div("Ticker Weight Details",
                         style={"color": C["gray"], "fontSize": "9px",
                                "letterSpacing": "1px", "marginBottom": "6px"}),
                dash_table.DataTable(
                    data=detail_rows,
                    columns=[{"name": c, "id": c}
                             for c in detail_rows[0]] if detail_rows else [],
                    style_table={"overflowX": "auto"},
                    style_header=tbl_style_hdr,
                    style_cell=tbl_style_cell,
                    style_data_conditional=[
                        {"if": {"column_id": "TICKER"},
                         "color": C["orange"], "fontWeight": "bold"},
                    ],
                ),
            ], style={"flex": "1", "minWidth": 0,
                      "backgroundColor": C["panel"],
                      "border": f"1px solid {C['border']}",
                      "padding": "10px"}),
        ], style={"display": "flex", "gap": "8px"}),
    ]))

    return html.Div(blocks, style={"padding": "14px 16px"})


# ─────────────────────────────────────────────────────────────────
# LIVE LOG Tab
# ─────────────────────────────────────────────────────────────────
def _tab_log(logs):
    lines = []
    for line in reversed((logs or [])[-200:]):
        c = (C["red"]    if "ERROR" in line else
             C["green"]  if ("✓" in line or "DONE" in line or "complete" in line) else
             C["yellow"] if "WARN"  in line else C["gray"])
        lines.append(html.Div(line,style={"color":c,"fontSize":"10px","fontFamily":F,
            "padding":"2px 0","borderBottom":f"1px solid {C['border']}"}))
    return html.Div([
        html.Div("LIVE LOG",style={"color":C["gray"],"fontSize":"9px",
            "letterSpacing":"2px","marginBottom":"10px"}),
        html.Div(lines or [html.Div("No logs",style={"color":C["dim"],"fontSize":"10px"})],
            style={"fontFamily":F,"maxHeight":"80vh","overflowY":"auto"}),
    ],style={"backgroundColor":C["panel"],"border":f"1px solid {C['border']}","padding":"12px"})


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" algorithm trading demo")
    print("  http://localhost:8050")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
