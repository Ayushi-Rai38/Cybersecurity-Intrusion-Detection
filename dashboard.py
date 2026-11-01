"""
Cybersecurity Intrusion Detection System — Streamlit Dashboard
==============================================================
Task 372 · UNSW-NB15 · PySpark MLlib Random Forest
Reads saved outputs from outputs/ — does NOT retrain the model.
Run: .venv/Scripts/streamlit run dashboard.py
"""

import os, json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CyberShield IDS · Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PREDS_CSV = os.path.join(BASE_DIR, "outputs", "predictions.csv")
METRICS_F = os.path.join(BASE_DIR, "outputs", "metrics.json")

# ── Colour palette (single source of truth) ───────────────────────────────────
C = {
    "bg_dark":    "#0d1117",
    "bg_card":    "#161b22",
    "bg_card2":   "#1c2230",
    "border":     "#30363d",
    "accent":     "#58a6ff",       # blue accent
    "green":      "#3fb950",
    "red":        "#f85149",
    "orange":     "#d29922",
    "purple":     "#bc8cff",
    "teal":       "#39d0d8",
    "text_main":  "#e6edf3",
    "text_muted": "#8b949e",
    "grad_blue":  "linear-gradient(135deg,#1f6feb 0%,#388bfd 100%)",
    "grad_red":   "linear-gradient(135deg,#da3633 0%,#f85149 100%)",
    "grad_green": "linear-gradient(135deg,#1a7f37 0%,#3fb950 100%)",
    "grad_purple":"linear-gradient(135deg,#6e40c9 0%,#bc8cff 100%)",
    "grad_teal":  "linear-gradient(135deg,#0d7377 0%,#39d0d8 100%)",
    "grad_orange":"linear-gradient(135deg,#9e6a03 0%,#d29922 100%)",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Imports ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset / Base ────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: {C['bg_dark']} !important;
    color: {C['text_main']} !important;
}}
.stApp {{ background-color: {C['bg_dark']}; }}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {C['bg_card']} !important;
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {C['text_main']} !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {C['bg_card2']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['text_main']} !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] span {{
    color: {C['text_main']} !important;
}}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['bg_card']} !important;
    border-radius: 12px !important;
    padding: 4px 6px !important;
    gap: 4px !important;
    border: 1px solid {C['border']} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 8px !important;
    color: {C['text_muted']} !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    border: none !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: {C['accent']} !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(88,166,255,0.35) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 1.5rem !important;
}}

/* ── Dataframe ───────────────────────────────────────────────── */
.stDataFrame, [data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid {C['border']} !important;
}}
.stDataFrame table {{ background: {C['bg_card']}; color: {C['text_main']}; }}
.stDataFrame th {{ background: {C['bg_card2']} !important; color: {C['accent']} !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.05em; }}
.stDataFrame td {{ font-size: 0.84rem !important; }}

/* ── Streamlit components ────────────────────────────────────── */
.stAlert {{ border-radius: 10px !important; }}
div[data-testid="stMetric"] {{
    background: {C['bg_card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}}
div[data-testid="stMetric"] label {{ color: {C['text_muted']} !important; font-size: 0.8rem !important; }}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {C['text_main']} !important; font-size: 1.6rem !important; font-weight: 700 !important; }}

/* ── Pyplot containers ───────────────────────────────────────── */
[data-testid="stImage"], .element-container:has([data-testid="stPyplotAnchored"]) {{
    border-radius: 12px !important;
    overflow: hidden !important;
}}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg_dark']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C['text_muted']}; }}

/* ── Download button ─────────────────────────────────────────── */
.stDownloadButton > button {{
    width: 100% !important;
    background: {C['accent']} !important;
    color: #000 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.55rem 1rem !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}}
.stDownloadButton > button:hover {{
    background: #79b8ff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(88,166,255,0.4) !important;
}}

/* ── Info / warning boxes ────────────────────────────────────── */
.ids-info {{
    background: rgba(88,166,255,0.08);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: {C['text_main']};
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 0.5rem 0 1rem 0;
}}
.ids-warn {{
    background: rgba(210,153,34,0.1);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: {C['text_main']};
    font-size: 0.9rem;
}}

/* ── Section divider ─────────────────────────────────────────── */
.sec-title {{
    display: flex; align-items: center; gap: 10px;
    font-size: 1.05rem; font-weight: 700;
    color: {C['text_main']};
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0.2rem 0 1.2rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {C['border']};
}}
.sec-title .dot {{
    width: 8px; height: 8px;
    background: {C['accent']};
    border-radius: 50%;
    box-shadow: 0 0 8px {C['accent']};
    flex-shrink: 0;
}}

/* ── KPI card ────────────────────────────────────────────────── */
.kpi {{
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    color: #fff;
    position: relative;
    overflow: hidden;
    min-height: 110px;
    display: flex; flex-direction: column; justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}}
.kpi::before {{
    content: "";
    position: absolute; top: -30px; right: -30px;
    width: 100px; height: 100px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}}
.kpi-icon {{ font-size: 1.6rem; line-height: 1; margin-bottom: 0.4rem; }}
.kpi-value {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; }}
.kpi-label {{ font-size: 0.75rem; font-weight: 500; opacity: 0.85; margin-top: 0.35rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.kpi-sub {{ font-size: 0.7rem; opacity: 0.65; margin-top: 0.15rem; }}

/* ── Metric pill ─────────────────────────────────────────────── */
.metric-pill {{
    background: {C['bg_card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.metric-pill::after {{
    content: "";
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 3px;
}}
.mp-blue::after   {{ background: {C['accent']}; }}
.mp-red::after    {{ background: {C['red']}; }}
.mp-green::after  {{ background: {C['green']}; }}
.mp-orange::after {{ background: {C['orange']}; }}
.mp-purple::after {{ background: {C['purple']}; }}
.metric-pill .mp-val {{
    font-size: 1.75rem; font-weight: 800;
    letter-spacing: -0.02em; color: {C['text_main']};
}}
.metric-pill .mp-label {{
    font-size: 0.72rem; color: {C['text_muted']};
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-top: 0.3rem; font-weight: 500;
}}

/* ── Risk badge ──────────────────────────────────────────────── */
.badge-high   {{ display:inline-block; background:rgba(248,81,73,0.15); color:{C['red']};    border:1px solid {C['red']};    border-radius:5px; padding:2px 8px; font-size:0.72rem; font-weight:600; }}
.badge-medium {{ display:inline-block; background:rgba(210,153,34,0.15); color:{C['orange']}; border:1px solid {C['orange']}; border-radius:5px; padding:2px 8px; font-size:0.72rem; font-weight:600; }}
.badge-low    {{ display:inline-block; background:rgba(63,185,80,0.15);  color:{C['green']};  border:1px solid {C['green']};  border-radius:5px; padding:2px 8px; font-size:0.72rem; font-weight:600; }}
.badge-susp   {{ display:inline-block; background:rgba(248,81,73,0.12); color:{C['red']};    border:1px solid rgba(248,81,73,0.4); border-radius:5px; padding:2px 8px; font-size:0.72rem; font-weight:600; }}
.badge-benign {{ display:inline-block; background:rgba(63,185,80,0.12); color:{C['green']};  border:1px solid rgba(63,185,80,0.4);  border-radius:5px; padding:2px 8px; font-size:0.72rem; font-weight:600; }}

/* ── Alert risk summary row ──────────────────────────────────── */
.risk-row {{
    background: {C['bg_card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1rem 1.3rem;
    text-align: center;
}}
.risk-row .rr-val  {{ font-size: 1.8rem; font-weight: 800; }}
.risk-row .rr-lbl  {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: {C['text_muted']}; margin-top: 0.2rem; }}

/* ── Sidebar branding ────────────────────────────────────────── */
.sb-brand {{
    padding: 1.2rem 0.8rem 0.8rem 0.8rem;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 1rem;
}}
.sb-brand h1 {{
    font-size: 1rem !important; font-weight: 800 !important;
    color: {C['accent']} !important;
    letter-spacing: 0.04em; margin: 0; padding: 0;
}}
.sb-brand p {{
    font-size: 0.72rem; color: {C['text_muted']};
    margin: 0.2rem 0 0 0; padding: 0;
}}
.sb-filter-hdr {{
    font-size: 0.7rem; font-weight: 700; color: {C['text_muted']};
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 1rem 0 0.4rem 0;
}}
.sb-stat {{
    background: {C['bg_card2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.4rem;
}}
.sb-stat .sb-s-label {{ font-size: 0.75rem; color: {C['text_muted']}; }}
.sb-stat .sb-s-val   {{ font-size: 0.85rem; font-weight: 700; color: {C['text_main']}; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB global style (dark theme for all charts)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  C["bg_card"],
    "axes.facecolor":    C["bg_card"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text_muted"],
    "axes.titlecolor":   C["text_main"],
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     12,
    "axes.labelsize":    9,
    "xtick.color":       C["text_muted"],
    "ytick.color":       C["text_muted"],
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.7,
    "text.color":        C["text_main"],
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  C["bg_card2"],
    "legend.edgecolor":  C["border"],
    "legend.fontsize":   8,
    "legend.labelcolor": C["text_main"],
})

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    if not os.path.exists(PREDS_CSV):
        return None
    return pd.read_csv(PREDS_CSV)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_F):
        return None
    with open(METRICS_F) as f:
        return json.load(f)

preds   = load_predictions()
metrics = load_metrics()

# ─── Guard: outputs not ready ─────────────────────────────────────────────────
if preds is None or metrics is None:
    st.markdown("""
    <div class="ids-warn">
    <strong>⚠️  Outputs not found</strong><br><br>
    The dashboard needs pre-computed predictions and metrics.<br>
    Run the training script first, then refresh this page:
    <pre style="margin:0.6rem 0 0 0;background:rgba(0,0,0,0.3);padding:0.5rem 0.8rem;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#e6edf3;">.venv\\Scripts\\python main.py</pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
total_records = len(preds)
suspicious    = int((preds["predicted_status"] == "Suspicious").sum())
benign        = int((preds["predicted_status"] == "Benign").sum())
has_bytes     = "sbytes" in preds.columns and "dbytes" in preds.columns

if has_bytes:
    total_bytes = int(preds["sbytes"].fillna(0).sum() + preds["dbytes"].fillna(0).sum())
    bytes_display = f"{total_bytes / 1e9:.2f} GB"
else:
    bytes_display = "N/A"

alert_rate = suspicious / total_records * 100 if total_records else 0
cm_data    = metrics.get("confusion_matrix", {})
tn, fp, fn, tp = (cm_data.get(k, 0) for k in ("TN","FP","FN","TP"))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <h1>🛡️ CyberShield IDS</h1>
        <p>UNSW-NB15 · PySpark MLlib · Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-filter-hdr">🔍 Filters</div>', unsafe_allow_html=True)

    status_opts = ["All"] + sorted(preds["predicted_status"].dropna().unique().tolist())
    sel_status  = st.selectbox("Predicted Status", status_opts, key="f_status")

    risk_order  = ["High", "Medium", "Low"]
    risk_opts   = ["All"] + sorted(
        preds["risk_level"].dropna().unique().tolist(),
        key=lambda x: risk_order.index(x) if x in risk_order else 99
    )
    sel_risk = st.selectbox("Risk Level", risk_opts, key="f_risk")

    if "proto" in preds.columns:
        proto_opts = ["All"] + sorted(preds["proto"].dropna().unique().tolist())
        sel_proto  = st.selectbox("Protocol", proto_opts, key="f_proto")
    else:
        sel_proto = "All"

    # Apply filters
    filtered = preds.copy()
    if sel_status != "All":
        filtered = filtered[filtered["predicted_status"] == sel_status]
    if sel_risk != "All":
        filtered = filtered[filtered["risk_level"] == sel_risk]
    if sel_proto != "All" and "proto" in filtered.columns:
        filtered = filtered[filtered["proto"] == sel_proto]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-filter-hdr">📊 Filter Summary</div>', unsafe_allow_html=True)

    for label, val in [
        ("Showing", f"{len(filtered):,} records"),
        ("Total",   f"{total_records:,} records"),
        ("Match %", f"{len(filtered)/total_records*100:.1f}%"),
    ]:
        st.markdown(f"""
        <div class="sb-stat">
            <span class="sb-s-label">{label}</span>
            <span class="sb-s-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Export Filtered CSV",
        data=filtered.to_csv(index=False),
        file_name="cybershield_predictions.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.68rem;color:{C['text_muted']};padding:0.4rem 0;border-top:1px solid {C['border']};line-height:1.6;">
        Model: Random Forest (100 trees, depth 10)<br>
        Dataset: UNSW-NB15 · 82,332 test records<br>
        Framework: PySpark MLlib 4.x
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;
            padding:1.4rem 0 0.8rem 0;border-bottom:1px solid {C['border']};margin-bottom:1.8rem;">
    <div>
        <div style="font-size:1.7rem;font-weight:800;letter-spacing:-0.02em;color:{C['text_main']};line-height:1.1;">
            🛡️ Cybersecurity Intrusion Detection System
        </div>
        <div style="font-size:0.85rem;color:{C['text_muted']};margin-top:0.45rem;">
            UNSW-NB15 Network Traffic Dataset &nbsp;·&nbsp; PySpark MLlib Random Forest &nbsp;·&nbsp; Real-time Threat Intelligence
        </div>
    </div>
    <div style="text-align:right;flex-shrink:0;padding-left:1.5rem;">
        <div style="background:{'rgba(248,81,73,0.12)' if alert_rate>40 else 'rgba(63,185,80,0.12)'};
                    border:1px solid {'rgba(248,81,73,0.4)' if alert_rate>40 else 'rgba(63,185,80,0.4)'};
                    border-radius:8px;padding:0.5rem 1rem;display:inline-block;">
            <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
                         color:{'C[red]' if alert_rate>40 else C['green']};">Alert Rate</span><br>
            <span style="font-size:1.4rem;font-weight:800;
                         color:{C['red'] if alert_rate>40 else C['green']};">{alert_rate:.1f}%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS (main navigation)
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_alerts, tab_eval, tab_categories, tab_traffic = st.tabs([
    "📊  Overview",
    "🚨  Alerts",
    "🎯  Evaluation",
    "🎭  Attack Categories",
    "📡  Traffic Analysis",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Key Performance Indicators</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpi_defs = [
        (k1, "📦", f"{total_records:,}", "Total Test Records",
         f"{total_records//1000}k connections analysed", C["grad_blue"]),
        (k2, "🚨", f"{suspicious:,}", "Suspicious Alerts",
         f"{alert_rate:.1f}% of total traffic", C["grad_red"]),
        (k3, "✅", f"{benign:,}", "Benign Records",
         f"{100-alert_rate:.1f}% clean traffic", C["grad_green"]),
        (k4, "🌐", bytes_display, "Total Traffic Volume",
         "Source + destination bytes", C["grad_teal"]),
    ]
    for col, icon, val, label, sub, grad in kpi_defs:
        with col:
            st.markdown(f"""
            <div class="kpi" style="background:{grad};">
                <div class="kpi-icon">{icon}</div>
                <div>
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model performance pills ────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Model Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    pill_defs = [
        (m1, metrics.get("accuracy", 0),         "Accuracy",        "mp-blue"),
        (m2, metrics.get("attack_precision", 0), "Attack Precision", "mp-red"),
        (m3, metrics.get("attack_recall", 0),    "Attack Recall",    "mp-green"),
        (m4, metrics.get("attack_f1", 0),        "Attack F1-Score",  "mp-orange"),
        (m5, metrics.get("roc_auc", 0),          "ROC-AUC",          "mp-purple"),
    ]
    for col, val, label, cls in pill_defs:
        with col:
            pct = val * 100
            bar_color = C["green"] if val >= 0.85 else (C["orange"] if val >= 0.7 else C["red"])
            st.markdown(f"""
            <div class="metric-pill {cls}">
                <div class="mp-val">{val:.4f}</div>
                <div class="mp-label">{label}</div>
                <div style="margin-top:0.5rem;background:{C['border']};border-radius:4px;height:4px;overflow:hidden;">
                    <div style="height:4px;width:{pct:.1f}%;background:{bar_color};border-radius:4px;
                                transition:width 0.8s ease;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick overview charts ──────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Traffic Snapshot</div>', unsafe_allow_html=True)

    ov1, ov2, ov3 = st.columns([1, 1, 1])

    # Donut: predicted status split
    with ov1:
        st.markdown(f"""
        <div style="background:{C['bg_card']};border:1px solid {C['border']};
                    border-radius:12px;padding:1rem;text-align:center;margin-bottom:0.5rem;">
            <div style="font-size:0.78rem;color:{C['text_muted']};text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:0.8rem;font-weight:600;">
                Prediction Split
            </div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        sizes  = [benign, suspicious]
        colors = [C["green"], C["red"]]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=["Benign", "Suspicious"],
            colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor=C["bg_card"], linewidth=2),
        )
        for t in texts:     t.set(color=C["text_muted"], fontsize=8)
        for at in autotexts: at.set(color="white", fontsize=8, fontweight="bold")
        centre = plt.Circle((0,0), 0.36, color=C["bg_card"])
        ax.add_patch(centre)
        ax.text(0, 0.06, f"{alert_rate:.1f}%", ha="center", va="center",
                fontsize=14, fontweight="bold", color=C["text_main"])
        ax.text(0, -0.15, "alerts", ha="center", va="center",
                fontsize=8, color=C["text_muted"])
        fig.patch.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Risk level breakdown
    with ov2:
        st.markdown(f"""
        <div style="background:{C['bg_card']};border:1px solid {C['border']};
                    border-radius:12px;padding:1rem;margin-bottom:0.5rem;">
            <div style="font-size:0.78rem;color:{C['text_muted']};text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:0.8rem;font-weight:600;">
                Risk Level Distribution
            </div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        risk_counts = (
            filtered["risk_level"].value_counts()
            .reindex(["High","Medium","Low"]).fillna(0)
        )
        bar_colors = [C["red"], C["orange"], C["green"]]
        bars = ax.barh(risk_counts.index, risk_counts.values,
                       color=bar_colors, height=0.5,
                       edgecolor=C["bg_card"], linewidth=1.5)
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_width() + risk_counts.max()*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{int(val):,}", va="center",
                    color=C["text_muted"], fontsize=8)
        ax.set_xlabel("Records", color=C["text_muted"])
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        fig.patch.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Top attack categories
    with ov3:
        st.markdown(f"""
        <div style="background:{C['bg_card']};border:1px solid {C['border']};
                    border-radius:12px;padding:1rem;margin-bottom:0.5rem;">
            <div style="font-size:0.78rem;color:{C['text_muted']};text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:0.8rem;font-weight:600;">
                Top Attack Categories
            </div>""", unsafe_allow_html=True)
        if "attack_cat" in preds.columns:
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            top_cats = preds["attack_cat"].value_counts().head(7)
            grad_colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top_cats)))
            bars = ax.barh(top_cats.index[::-1], top_cats.values[::-1],
                           color=grad_colors[::-1], height=0.6,
                           edgecolor=C["bg_card"], linewidth=1)
            for bar, val in zip(bars, top_cats.values[::-1]):
                ax.text(bar.get_width() + top_cats.max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{int(val):,}", va="center",
                        color=C["text_muted"], fontsize=7.5)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            ax.xaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=8)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_alerts:

    suspicious_df = (
        filtered[filtered["predicted_status"] == "Suspicious"]
        .sort_values("attack_prob", ascending=False)
    )

    # ── Risk summary row ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Risk Summary (Filtered View)</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)

    risk_cfg = [
        (r1, "High",   C["red"],    "🔴"),
        (r2, "Medium", C["orange"], "🟡"),
        (r3, "Low",    C["green"],  "🟢"),
    ]
    for col, risk, color, icon in risk_cfg:
        count = int((suspicious_df["risk_level"] == risk).sum()) if len(suspicious_df) > 0 else 0
        with col:
            st.markdown(f"""
            <div class="risk-row" style="border-left:3px solid {color};">
                <div class="rr-val" style="color:{color};">{icon} {count:,}</div>
                <div class="rr-lbl">{risk} Risk Alerts</div>
            </div>""", unsafe_allow_html=True)

    with r4:
        avg_prob = suspicious_df["attack_prob"].mean() if len(suspicious_df) > 0 else 0
        st.markdown(f"""
        <div class="risk-row" style="border-left:3px solid {C['purple']};">
            <div class="rr-val" style="color:{C['purple']};">⚡ {avg_prob:.3f}</div>
            <div class="rr-lbl">Avg Attack Probability</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alert probability distribution ────────────────────────────────────────
    if len(suspicious_df) > 0 and "attack_prob" in suspicious_df.columns:
        st.markdown('<div class="sec-title"><span class="dot"></span>Attack Probability Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 2.8))
        probs = suspicious_df["attack_prob"].dropna()
        ax.hist(probs, bins=50, color=C["red"], alpha=0.7, edgecolor=C["bg_card"])
        ax.axvline(0.80, color=C["red"],    lw=1.5, ls="--", alpha=0.9, label="High threshold (0.80)")
        ax.axvline(0.50, color=C["orange"], lw=1.5, ls="--", alpha=0.9, label="Medium threshold (0.50)")
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                         0.80, 1.0, alpha=0.07, color=C["red"])
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                         0.50, 0.80, alpha=0.07, color=C["orange"])
        ax.set_xlabel("Attack Probability Score")
        ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", framealpha=0.5)
        fig.patch.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alert table ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Suspicious Records (Top 500, sorted by Attack Probability)</div>', unsafe_allow_html=True)

    display_cols = [c for c in [
        "risk_level","attack_prob","attack_cat","proto","state",
        "sbytes","dbytes","dur","actual_status","predicted_status"
    ] if c in suspicious_df.columns]

    if len(suspicious_df) > 0:
        show_df = suspicious_df[display_cols].head(500).copy()
        if "attack_prob" in show_df.columns:
            show_df["attack_prob"] = show_df["attack_prob"].map("{:.4f}".format)
        st.dataframe(show_df, use_container_width=True, height=400)
        st.markdown(
            f'<div style="font-size:0.75rem;color:{C["text_muted"]};margin-top:0.3rem;">'
            f'Showing top {min(500,len(suspicious_df)):,} of <strong style="color:{C["text_main"]}">'
            f'{len(suspicious_df):,}</strong> suspicious records</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class="ids-info">ℹ️ No suspicious records match the current filters. 
        Try changing the sidebar filters to broaden the selection.</div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:

    st.markdown('<div class="sec-title"><span class="dot"></span>Confusion Matrix</div>', unsafe_allow_html=True)

    ev1, ev2 = st.columns([1, 1])

    with ev1:
        # Styled confusion matrix
        cm_arr = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cmap = plt.cm.Blues
        cmap.set_bad(C["bg_card"])
        im = ax.imshow(cm_arr, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAttack"], fontsize=9)
        ax.set_yticklabels(["Actual\nNormal", "Actual\nAttack"], fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=9, labelpad=8)
        ax.set_ylabel("Actual Class",    fontsize=9, labelpad=8)
        cell_labels = [["TN","FP"],["FN","TP"]]
        cell_descs  = [["Correctly\nidentified\nnormal","Normal misclassified\nas attack"],
                       ["Attack\nmissed","Correctly\nidentified\nattack"]]
        thresh = cm_arr.max() / 1.7
        for i in range(2):
            for j in range(2):
                clr = "white" if cm_arr[i,j] > thresh else C["text_main"]
                ax.text(j, i - 0.18, cell_labels[i][j],
                        ha="center", va="center", fontsize=15, fontweight="bold", color=clr)
                ax.text(j, i + 0.18, f"{cm_arr[i,j]:,}",
                        ha="center", va="center", fontsize=11, color=clr)
        fig.patch.set_facecolor(C["bg_card"])
        ax.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=1)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with ev2:
        # Metrics breakdown panel
        st.markdown(f"""
        <div style="background:{C['bg_card']};border:1px solid {C['border']};
                    border-radius:12px;padding:1.4rem;height:100%;">
            <div style="font-size:0.78rem;color:{C['text_muted']};text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:1rem;font-weight:600;">
                Evaluation Metrics Detail
            </div>""", unsafe_allow_html=True)

        metric_rows = [
            ("Accuracy",        metrics.get("accuracy", 0),         C["accent"],  "Overall correctness of predictions"),
            ("Attack Precision", metrics.get("attack_precision", 0), C["red"],     "Of flagged alerts, how many are real attacks"),
            ("Attack Recall",    metrics.get("attack_recall", 0),    C["green"],   "Of real attacks, how many were caught"),
            ("Attack F1-Score",  metrics.get("attack_f1", 0),        C["orange"],  "Harmonic mean of precision & recall"),
            ("ROC-AUC",          metrics.get("roc_auc", 0),          C["purple"],  "Discrimination across all thresholds"),
        ]
        for name, val, color, desc in metric_rows:
            bar_w = val * 100
            st.markdown(f"""
            <div style="margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.25rem;">
                    <span style="font-size:0.85rem;font-weight:600;color:{C['text_main']};">{name}</span>
                    <span style="font-size:1rem;font-weight:800;color:{color};">{val:.4f}</span>
                </div>
                <div style="background:{C['border']};border-radius:4px;height:6px;overflow:hidden;">
                    <div style="height:6px;width:{bar_w:.1f}%;background:{color};
                                border-radius:4px;box-shadow:0 0 8px {color}66;"></div>
                </div>
                <div style="font-size:0.7rem;color:{C['text_muted']};margin-top:0.2rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actual vs Predicted bar chart ──────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Actual vs Predicted Counts</div>', unsafe_allow_html=True)

    ev3, ev4 = st.columns([1.2, 0.8])

    with ev3:
        actual_normal  = int((preds["label"] == 0).sum()) if "label" in preds.columns else 0
        actual_attack  = int((preds["label"] == 1).sum()) if "label" in preds.columns else 0

        fig, ax = plt.subplots(figsize=(7, 3.8))
        categories = ["Actual\nNormal", "Actual\nAttack", "Predicted\nNormal", "Predicted\nAttack"]
        values     = [actual_normal, actual_attack, benign, suspicious]
        colors_bar = [C["green"], C["red"], "#388bfd", "#f05151"]

        x = np.arange(len(categories))
        bars = ax.bar(x, values, width=0.55, color=colors_bar,
                      edgecolor=C["bg_card"], linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f"{val:,}", ha="center", fontsize=9, color=C["text_muted"])

        ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Count")
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

        # Divider between actual vs predicted
        ax.axvline(1.5, color=C["border"], lw=1.5, ls="--", alpha=0.7)
        ax.text(0.75, max(values)*0.97, "Actual",    ha="center", fontsize=8,
                color=C["text_muted"], style="italic")
        ax.text(2.25, max(values)*0.97, "Predicted", ha="center", fontsize=8,
                color=C["text_muted"], style="italic")
        fig.patch.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with ev4:
        # Confusion matrix numbers in a clean table layout
        st.markdown(f"""
        <div style="background:{C['bg_card']};border:1px solid {C['border']};
                    border-radius:12px;padding:1.4rem;">
            <div style="font-size:0.78rem;color:{C['text_muted']};text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:1rem;font-weight:600;">
                Confusion Matrix Breakdown
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;">
                <div style="background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);
                            border-radius:10px;padding:1rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:{C['green']};">{tn:,}</div>
                    <div style="font-size:0.7rem;color:{C['text_muted']};margin-top:0.2rem;
                                text-transform:uppercase;letter-spacing:0.04em;">True Negative</div>
                    <div style="font-size:0.65rem;color:{C['text_muted']};margin-top:0.1rem;">Correctly Normal</div>
                </div>
                <div style="background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,0.3);
                            border-radius:10px;padding:1rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:{C['red']};">{fp:,}</div>
                    <div style="font-size:0.7rem;color:{C['text_muted']};margin-top:0.2rem;
                                text-transform:uppercase;letter-spacing:0.04em;">False Positive</div>
                    <div style="font-size:0.65rem;color:{C['text_muted']};margin-top:0.1rem;">Normal → Attack (False Alarm)</div>
                </div>
                <div style="background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);
                            border-radius:10px;padding:1rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:{C['orange']};">{fn:,}</div>
                    <div style="font-size:0.7rem;color:{C['text_muted']};margin-top:0.2rem;
                                text-transform:uppercase;letter-spacing:0.04em;">False Negative</div>
                    <div style="font-size:0.65rem;color:{C['text_muted']};margin-top:0.1rem;">Attack Missed ⚠️</div>
                </div>
                <div style="background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.3);
                            border-radius:10px;padding:1rem;text-align:center;">
                    <div style="font-size:1.5rem;font-weight:800;color:{C['accent']};">{tp:,}</div>
                    <div style="font-size:0.7rem;color:{C['text_muted']};margin-top:0.2rem;
                                text-transform:uppercase;letter-spacing:0.04em;">True Positive</div>
                    <div style="font-size:0.65rem;color:{C['text_muted']};margin-top:0.1rem;">Correctly Detected Attack</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ATTACK CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_categories:

    if "attack_cat" not in preds.columns:
        st.markdown('<div class="ids-info">ℹ️ <code>attack_cat</code> column not available in predictions.</div>', unsafe_allow_html=True)
    else:
        cat_counts  = preds["attack_cat"].value_counts()
        susp_cats   = (preds[preds["predicted_status"] == "Suspicious"]["attack_cat"].value_counts())

        # ── Category summary cards ─────────────────────────────────────────────
        st.markdown('<div class="sec-title"><span class="dot"></span>Attack Category Summary</div>', unsafe_allow_html=True)

        cat_cols = st.columns(min(5, len(cat_counts)))
        palette = [C["red"], C["orange"], C["purple"], C["teal"], C["accent"],
                   C["green"], "#e879f9", "#fbbf24", "#34d399", "#60a5fa"]
        for i, (cat, cnt) in enumerate(cat_counts.items()):
            if i >= len(cat_cols): break
            col = cat_cols[i]
            color = palette[i % len(palette)]
            pct = cnt / total_records * 100
            with col:
                st.markdown(f"""
                <div style="background:{C['bg_card']};border:1px solid {C['border']};
                            border-top:3px solid {color};border-radius:10px;
                            padding:0.9rem;text-align:center;margin-bottom:0.5rem;">
                    <div style="font-size:1.3rem;font-weight:800;color:{color};">{cnt:,}</div>
                    <div style="font-size:0.72rem;font-weight:600;color:{C['text_main']};
                                margin-top:0.2rem;word-break:break-word;">{cat}</div>
                    <div style="font-size:0.65rem;color:{C['text_muted']};margin-top:0.1rem;">{pct:.1f}% of total</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row ─────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title"><span class="dot"></span>Category Distribution</div>', unsafe_allow_html=True)
        ca1, ca2 = st.columns(2)

        with ca1:
            fig, ax = plt.subplots(figsize=(6, 5))
            explode = [0.04] * len(cat_counts)
            wedge_colors = plt.cm.tab10(np.linspace(0, 1, len(cat_counts)))
            wedges, texts, autotexts = ax.pie(
                cat_counts.values,
                labels=cat_counts.index,
                colors=wedge_colors,
                autopct="%1.1f%%",
                startangle=140,
                pctdistance=0.82,
                explode=explode,
                wedgeprops=dict(edgecolor=C["bg_card"], linewidth=1.5),
            )
            for t in texts:     t.set(color=C["text_muted"], fontsize=7.5)
            for at in autotexts: at.set(fontsize=7, fontweight="bold", color="white")
            centre = plt.Circle((0,0), 0.5, color=C["bg_card"])
            ax.add_patch(centre)
            ax.set_title("Full Dataset — Category Distribution", pad=12)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with ca2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = susp_cats.head(10)
            bar_clrs = plt.cm.Reds(np.linspace(0.45, 0.85, len(sc)))
            bars = ax.barh(sc.index[::-1], sc.values[::-1],
                           color=bar_clrs, height=0.6,
                           edgecolor=C["bg_card"], linewidth=1)
            for bar, val in zip(bars, sc.values[::-1]):
                ax.text(bar.get_width() + sc.max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{int(val):,}", va="center",
                        color=C["text_muted"], fontsize=8)
            ax.set_xlabel("Count")
            ax.set_title("Suspicious Predictions — Categories", pad=12)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cross-tab ──────────────────────────────────────────────────────────
        st.markdown('<div class="sec-title"><span class="dot"></span>Category × Predicted Status (Crosstab)</div>', unsafe_allow_html=True)
        cross = pd.crosstab(preds["attack_cat"], preds["predicted_status"], margins=True)
        try:
            cross = cross.sort_values("All", ascending=False)
        except Exception:
            pass
        st.dataframe(cross, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Stacked bar: category vs status ───────────────────────────────────
        st.markdown('<div class="sec-title"><span class="dot"></span>Detection Rate per Attack Category</div>', unsafe_allow_html=True)
        cat_status = (
            preds.groupby(["attack_cat","predicted_status"])
            .size().unstack(fill_value=0)
        )
        # Add detection rate column
        if "Suspicious" in cat_status.columns and "Benign" in cat_status.columns:
            cat_status["detection_rate"] = (
                cat_status["Suspicious"] / (cat_status["Suspicious"] + cat_status["Benign"]) * 100
            ).round(1)
        cat_status_plot = cat_status.drop(columns=["detection_rate"], errors="ignore")

        fig, ax = plt.subplots(figsize=(10, 3.8))
        bar_colors_stack = {"Benign": C["green"], "Suspicious": C["red"]}
        bottoms = np.zeros(len(cat_status_plot))
        for col_name in cat_status_plot.columns:
            vals = cat_status_plot[col_name].values.astype(float)
            ax.bar(cat_status_plot.index, vals,
                   bottom=bottoms, label=col_name,
                   color=bar_colors_stack.get(col_name, C["accent"]),
                   edgecolor=C["bg_card"], linewidth=1)
            bottoms += vals
        ax.set_ylabel("Records")
        ax.set_xlabel("")
        plt.xticks(rotation=25, ha="right", fontsize=8.5)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.legend(title="Predicted", fontsize=8, title_fontsize=8)
        fig.patch.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — TRAFFIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_traffic:

    # ── Timestamp note ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ids-info">
    <strong>📌 Dataset Note — No Timestamp Column:</strong>
    The standard UNSW-NB15 CSV files do not include a per-record event timestamp, so chronological
    trend charts are not possible without fabricating data. The charts below use protocol, connection
    state, and byte-volume dimensions — meaningful proxies for traffic pattern analysis.
    </div>
    """, unsafe_allow_html=True)

    # ── Traffic volume charts ──────────────────────────────────────────────────
    if has_bytes and "proto" in preds.columns:
        st.markdown('<div class="sec-title"><span class="dot"></span>Traffic Volume by Protocol</div>', unsafe_allow_html=True)
        tr1, tr2 = st.columns(2)

        with tr1:
            proto_bytes = (
                preds.groupby("proto")[["sbytes","dbytes"]]
                .sum()
                .assign(total_bytes=lambda d: d["sbytes"] + d["dbytes"])
                .sort_values("total_bytes", ascending=False)
                .head(15)
            )
            fig, ax = plt.subplots(figsize=(6, 4.5))
            grad_c = plt.cm.Blues(np.linspace(0.35, 0.85, len(proto_bytes)))
            bars = ax.barh(proto_bytes.index[::-1],
                           proto_bytes["total_bytes"].values[::-1] / 1e6,
                           color=grad_c, height=0.65,
                           edgecolor=C["bg_card"], linewidth=1)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + proto_bytes["total_bytes"].max()/1e6*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f"{w:.0f} MB", va="center",
                        color=C["text_muted"], fontsize=7.5)
            ax.set_xlabel("Total Traffic (MB)")
            ax.set_title("Traffic Volume by Protocol (Top 15)", pad=10)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(left=False)
            ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tr2:
            # Scatter: src vs dst bytes (log scale)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sample = preds.dropna(subset=["sbytes","dbytes"]).sample(
                min(6000, len(preds)), random_state=42
            )
            sc_colors = sample["predicted_status"].map({
                "Suspicious": C["red"], "Benign": C["green"]
            }).fillna(C["teal"])
            ax.scatter(np.log1p(sample["sbytes"]), np.log1p(sample["dbytes"]),
                       c=sc_colors, s=6, alpha=0.45)
            ax.set_xlabel("log(1 + Source Bytes)")
            ax.set_ylabel("log(1 + Destination Bytes)")
            ax.set_title("Source vs Destination Bytes — Benign vs Suspicious", pad=10)
            legend_patches = [
                mpatches.Patch(color=C["green"], label="Benign"),
                mpatches.Patch(color=C["red"],   label="Suspicious"),
            ]
            ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Protocol & State analysis ──────────────────────────────────────────────
    st.markdown('<div class="sec-title"><span class="dot"></span>Protocol & Connection State Breakdown</div>', unsafe_allow_html=True)
    tr3, tr4 = st.columns(2)

    with tr3:
        if "proto" in preds.columns:
            top_protos  = preds["proto"].value_counts().head(12).index
            proto_status = (
                preds[preds["proto"].isin(top_protos)]
                .groupby(["proto","predicted_status"])
                .size().unstack(fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(6, 4.5))
            bottoms2 = np.zeros(len(proto_status))
            clr_map = {"Benign": C["green"], "Suspicious": C["red"]}
            for col_n in proto_status.columns:
                vals = proto_status[col_n].values.astype(float)
                ax.bar(proto_status.index, vals, bottom=bottoms2,
                       label=col_n, color=clr_map.get(col_n, C["accent"]),
                       edgecolor=C["bg_card"], linewidth=1)
                bottoms2 += vals
            ax.set_title("Traffic Status by Protocol (Top 12)", pad=10)
            ax.set_ylabel("Records")
            plt.xticks(rotation=35, ha="right", fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            ax.legend(title="Status", fontsize=8, title_fontsize=8)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tr4:
        if "state" in preds.columns:
            top_states  = preds["state"].value_counts().head(10).index
            state_status = (
                preds[preds["state"].isin(top_states)]
                .groupby(["state","predicted_status"])
                .size().unstack(fill_value=0)
            )
            fig, ax = plt.subplots(figsize=(6, 4.5))
            bottoms3 = np.zeros(len(state_status))
            for col_n in state_status.columns:
                vals = state_status[col_n].values.astype(float)
                ax.bar(state_status.index, vals, bottom=bottoms3,
                       label=col_n, color=clr_map.get(col_n, C["purple"]),
                       edgecolor=C["bg_card"], linewidth=1)
                bottoms3 += vals
            ax.set_title("Traffic Status by Connection State (Top 10)", pad=10)
            ax.set_ylabel("Records")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            ax.legend(title="Status", fontsize=8, title_fontsize=8)
            fig.patch.set_facecolor(C["bg_card"])
            plt.tight_layout(pad=0.8)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── Traffic volume by predicted status ─────────────────────────────────────
    if has_bytes:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title"><span class="dot"></span>Traffic Volume by Predicted Status</div>', unsafe_allow_html=True)
        vol_by_status = (
            preds.groupby("predicted_status")[["sbytes","dbytes"]]
            .sum()
            .assign(src_MB=lambda d: d["sbytes"]/1e6,
                    dst_MB=lambda d: d["dbytes"]/1e6)
        )
        fig, ax = plt.subplots(figsize=(8, 3.5))
        x2   = np.arange(len(vol_by_status))
        w2   = 0.35
        b1   = ax.bar(x2 - w2/2, vol_by_status["src_MB"], w2,
                      label="Source Bytes (MB)",      color=C["accent"], alpha=0.85,
                      edgecolor=C["bg_card"], linewidth=1)
        b2   = ax.bar(x2 + w2/2, vol_by_status["dst_MB"], w2,
                      label="Destination Bytes (MB)", color=C["purple"], alpha=0.85,
                      edgecolor=C["bg_card"], linewidth=1)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max(vol_by_status[["src_MB","dst_MB"]].max())*0.01,
                    f"{h:.0f}", ha="center", fontsize=8, color=C["text_muted"])
        ax.set_xticks(x2); ax.set_xticklabels(vol_by_status.index, fontsize=9)
        ax.set_ylabel("Volume (MB)")
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        fig.patch.set_facecolor(C["bg_card"])
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:3rem;padding:1.2rem 0 0.5rem 0;
            border-top:1px solid {C['border']};
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:0.5rem;">
    <div style="font-size:0.75rem;color:{C['text_muted']};">
        🛡️ <strong style="color:{C['accent']};">CyberShield IDS</strong> &nbsp;·&nbsp;
        UNSW-NB15 Dataset &nbsp;·&nbsp; PySpark MLlib 4.x &nbsp;·&nbsp; Random Forest
    </div>
    <div style="font-size:0.7rem;color:{C['text_muted']};">
        ⚠️ <em>is_ftp_login = FTP-specific indicator only, not a general failed-login count.
        No sport/dsport or per-record timestamp in the UNSW-NB15 training/testing CSVs.</em>
    </div>
</div>
""", unsafe_allow_html=True)
