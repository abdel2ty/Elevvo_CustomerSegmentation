"""
SegmentIQ — Customer Segmentation Intelligence
Clean & Clinical · Medical Data Lab Aesthetic
5 Tabs: Overview · Profiler · Segments · Simulator · Data
Run: streamlit run customer_seg_v5.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SegmentIQ",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PAGE STATE ─────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = st.query_params.get("page", "overview")

def go_to(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,200;0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=Instrument+Serif:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:           #F7F8FA;
    --bg2:          #EEEEF2;
    --surface:      #FFFFFF;
    --surface2:     #F2F3F7;
    --border:       #DDE0E8;
    --border2:      #C8CBD6;
    --ink:          #0D0F14;
    --ink2:         #2E3140;
    --ink3:         #6B6F82;
    --ink4:         #A0A4B4;
    --teal:         #00897B;
    --teal-pale:    #E0F2F1;
    --teal-mid:     #80CBC4;
    --red:          #E53935;
    --red-pale:     #FFEBEE;
    --amber:        #F59E0B;
    --amber-pale:   #FEF3C7;
    --blue:         #1565C0;
    --blue-pale:    #E3F2FD;
    --indigo:       #3949AB;
    --indigo-pale:  #E8EAF6;
}

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HEADER ── */
.header-band {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 2.5rem;
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    min-height: 56px;
    position: sticky; top: 0; z-index: 999;
}
.header-left {
    display: flex; align-items: center; gap: 20px;
    border-right: 1px solid var(--border);
    padding-right: 24px; margin-right: 4px;
}
.header-logo {
    width: 28px; height: 28px; background: var(--ink);
    border-radius: 6px; display: flex; align-items: center; justify-content: center;
}
.header-wordmark { font-size: 0.95rem; font-weight: 600; color: var(--ink); letter-spacing: -0.02em; }
.header-version {
    font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--ink4);
    background: var(--surface2); border: 1px solid var(--border);
    padding: 2px 7px; border-radius: 3px; letter-spacing: 0.04em;
}
.header-nav {
    display: flex; align-items: stretch; gap: 0; flex: 1; padding: 0 1rem;
}
.nav-item {
    display: flex; align-items: center; gap: 6px; padding: 0 16px;
    font-size: 0.78rem; font-weight: 500; color: var(--ink3);
    border-bottom: 2px solid transparent; cursor: pointer;
    text-decoration: none; transition: color .12s, border-color .12s; user-select: none;
}
.nav-item:hover { color: var(--ink); border-bottom-color: var(--border2); }
.nav-item.active { color: var(--ink); font-weight: 600; border-bottom-color: var(--teal); }
.nav-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); }
.header-right {
    display: flex; align-items: center; gap: 10px;
    padding-left: 20px; border-left: 1px solid var(--border); margin-left: 4px;
}
.status-pill {
    display: flex; align-items: center; gap: 5px;
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--teal);
    background: var(--teal-pale); border: 1px solid var(--teal-mid);
    padding: 4px 10px; border-radius: 20px; letter-spacing: 0.04em;
}
.status-dot {
    width: 5px; height: 5px; background: var(--teal); border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── SUBHEADER ── */
.subheader {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 0.55rem 2.5rem; display: flex; align-items: center; justify-content: space-between;
}
.breadcrumb {
    display: flex; align-items: center; gap: 6px;
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--ink4);
    letter-spacing: 0.04em; text-transform: uppercase;
}
.breadcrumb b { color: var(--ink2); }
.breadcrumb-sep { color: var(--border2); }
.ts { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--ink4); letter-spacing: 0.06em; }

/* ── SHELL ── */
.shell { max-width: 1280px; margin: 0 auto; padding: 2rem 2rem 4rem; }

/* ── PAGE HERO ── */
.hero {
    display: flex; align-items: flex-end; justify-content: space-between;
    margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border);
}
.hero-title {
    font-family: 'Instrument Serif', serif; font-size: 2.6rem; font-style: italic;
    font-weight: 400; color: var(--ink); letter-spacing: -0.03em; line-height: 1.05;
}
.hero-title b { font-style: normal; font-weight: 400; color: var(--teal); }
.hero-desc { font-size: 0.8rem; color: var(--ink3); max-width: 360px; text-align: right; line-height: 1.75; font-weight: 300; }
.hero-model-tag { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--ink4); margin-top: 4px; text-align: right; letter-spacing: 0.06em; }

/* ── SECTION LABEL ── */
.s-label {
    font-family: 'DM Mono', monospace; font-size: 0.6rem; font-weight: 500;
    color: var(--ink4); text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 0.9rem;
}

/* ── CARDS ── */
.kpi-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.25rem 1.4rem; position: relative; overflow: hidden;
}
.kpi-accent-bar { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 8px 8px 0 0; }
.kpi-num { font-family: 'DM Mono', monospace; font-size: 2.2rem; font-weight: 300; color: var(--ink); letter-spacing: -0.05em; line-height: 1; margin-top: 4px; }
.kpi-name { font-size: 0.7rem; font-weight: 600; color: var(--ink2); letter-spacing: 0.01em; margin-top: 6px; }
.kpi-meta { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink4); margin-top: 3px; line-height: 1.6; }

.chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem 1.25rem 0.5rem; }
.chart-title { font-family: 'DM Mono', monospace; font-size: 0.62rem; font-weight: 500; color: var(--ink3); text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.3rem; }
.chart-subtitle { font-size: 0.72rem; font-family: 'Instrument Serif', serif; font-style: italic; color: var(--ink2); margin-bottom: 0.8rem; }

/* ── SEGMENT TABLE ── */
.seg-table { width: 100%; border-collapse: collapse; }
.seg-table th {
    font-family: 'DM Mono', monospace; font-size: 0.58rem; font-weight: 500; color: var(--ink4);
    text-transform: uppercase; letter-spacing: 0.14em; padding: 0 0 8px; text-align: left; border-bottom: 1px solid var(--border);
}
.seg-table td { padding: 10px 0; border-bottom: 1px solid var(--bg2); vertical-align: middle; }
.seg-table tr:last-child td { border-bottom: none; }
.seg-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 7px; }
.seg-name { font-size: 0.77rem; font-weight: 500; color: var(--ink); display: flex; align-items: center; }
.seg-tag { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink4); margin-top: 1px; padding-left: 15px; }
.seg-count { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--ink2); }
.seg-bar-wrap { height: 4px; background: var(--bg2); border-radius: 2px; overflow: hidden; width: 80px; }
.seg-bar-fill { height: 100%; border-radius: 2px; }
.seg-avg { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--ink3); }
.strategy-cell { font-size: 0.7rem; color: var(--ink3); font-weight: 300; max-width: 180px; line-height: 1.4; }

/* ── ALERTS ── */
.alert-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px; border-radius: 5px; margin-bottom: 6px; border: 1px solid transparent;
}
.alert-row.ok   { background: var(--teal-pale);  border-color: #B2DFDB; }
.alert-row.warn { background: var(--amber-pale);  border-color: #FDE68A; }
.alert-row.info { background: var(--blue-pale);   border-color: #BBDEFB; }
.alert-row.bad  { background: var(--red-pale);    border-color: #FFCDD2; }
.alert-ico { font-size: 12px; flex-shrink: 0; margin-top: 1px; }
.alert-t { font-size: 0.72rem; font-weight: 600; color: var(--ink); }
.alert-b { font-size: 0.65rem; color: var(--ink3); margin-top: 1px; line-height: 1.5; }

/* ── INLINE METRICS ── */
.inline-metrics { display: flex; gap: 0; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--surface); }
.inline-metric { flex: 1; padding: 0.9rem 1rem; border-right: 1px solid var(--border); }
.inline-metric:last-child { border-right: none; }
.im-val { font-family: 'DM Mono', monospace; font-size: 1.2rem; font-weight: 300; color: var(--ink); letter-spacing: -0.03em; }
.im-lbl { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink4); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }

/* ── PROFILER RESULT ── */
.result-panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.75rem; position: relative; overflow: hidden;
}
.result-ghost {
    font-family: 'Instrument Serif', serif; font-size: 7rem; font-style: italic;
    color: var(--ink); opacity: 0.04; position: absolute; bottom: -20px; right: 10px;
    line-height: 1; pointer-events: none; user-select: none;
}
.result-eyebrow { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink4); text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 4px; }
.result-name { font-family: 'Instrument Serif', serif; font-size: 2rem; font-style: italic; color: var(--ink); letter-spacing: -0.03em; line-height: 1.1; margin: 6px 0 4px; }
.result-tag-line { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--ink4); }
.result-divider { height: 1px; background: var(--border); margin: 1rem 0; }
.cluster-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 8px; border-radius: 5px; margin-bottom: 3px; }
.cluster-row.active { background: var(--surface2); border: 1px solid var(--border); }
.cluster-row-name { font-size: 0.74rem; font-weight: 500; color: var(--ink3); display: flex; align-items: center; gap: 7px; }
.cluster-row-ct { font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--ink4); }

/* ── SIM TILES ── */
.sim-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; margin-bottom: 1.5rem; }
.sim-tile {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.9rem 0.6rem; text-align: center;
    transition: border-color .15s;
}
.sim-tile.shifted { border-color: var(--teal); background: var(--teal-pale); }
.sim-lbl { font-family: 'DM Mono', monospace; font-size: 0.57rem; color: var(--ink4); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.sim-seg-name { font-family: 'Instrument Serif', serif; font-size: 0.9rem; font-style: italic; color: var(--ink); line-height: 1.2; }
.sim-badge { font-family: 'DM Mono', monospace; font-size: 0.58rem; margin-top: 5px; }
.sim-badge.changed { color: var(--teal); }
.sim-badge.same    { color: var(--ink4); }

/* ── DATA TABLE ── */
.dt-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.dt-table { width: 100%; border-collapse: collapse; }
.dt-table th {
    font-family: 'DM Mono', monospace; font-size: 0.6rem; font-weight: 500;
    color: var(--ink4); text-transform: uppercase; letter-spacing: 0.14em;
    padding: 10px 16px; text-align: left; border-bottom: 1px solid var(--border); background: var(--surface2);
}
.dt-table td { padding: 9px 16px; border-bottom: 1px solid var(--bg2); font-size: 0.78rem; color: var(--ink2); }
.dt-table tr:last-child td { border-bottom: none; }
.dt-table tr:hover td { background: var(--surface2); }
.seg-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 8px; border-radius: 3px; font-family: 'DM Mono', monospace;
    font-size: 0.62rem; font-weight: 500; border: 1px solid transparent;
}

/* ── DEEP DIVE STAT ── */
.stat-row {
    display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; margin-bottom: 1.5rem;
}
.stat-tile {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1rem 1.1rem;
}
.stat-val { font-family: 'DM Mono', monospace; font-size: 1.6rem; font-weight: 300; color: var(--ink); letter-spacing: -0.04em; }
.stat-lbl { font-size: 0.65rem; font-weight: 600; color: var(--ink3); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }
.stat-sub { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink4); margin-top: 2px; }

/* ── SEGMENTS PAGE GRID ── */
.seg-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 1.5rem; position: relative; overflow: hidden;
    transition: border-color .2s;
}
.seg-card:hover { border-color: var(--border2); }
.seg-card-accent { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; border-radius: 8px 0 0 8px; }
.seg-card-id {
    font-family: 'Instrument Serif', serif; font-size: 3.5rem; font-style: italic;
    opacity: 0.06; position: absolute; top: -8px; right: 10px; line-height: 1; user-select: none;
}

/* ── SLIDERS ── */
.stSlider label {
    font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important;
    font-weight: 500 !important; color: var(--ink3) !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
div[data-testid="stSlider"] > div > div > div { background: var(--border) !important; height: 2px !important; }
div[data-testid="stSlider"] > div > div > div > div { background: var(--teal) !important; box-shadow: 0 0 0 3px rgba(0,137,123,0.15) !important; width: 14px !important; height: 14px !important; }
.stSelectbox label, .stNumberInput label, .stMultiSelect label {
    font-family: 'DM Mono', monospace !important; font-size: 0.62rem !important;
    font-weight: 500 !important; color: var(--ink3) !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 5px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; color: var(--ink) !important;
}

/* ── FOOTER ── */
.lab-footer {
    font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink4);
    padding: 1.5rem 0 0.5rem; margin-top: 3rem; border-top: 1px solid var(--border);
    letter-spacing: 0.08em; display: flex; justify-content: space-between; align-items: center;
}
.rule { height: 1px; background: var(--border); margin: 1.8rem 0; }

/* Hide streamlit nav buttons */
div[data-testid="stHorizontalBlock"] button { display: none !important; }
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    np.random.seed(42)
    n = 200
    income_raw = np.concatenate([
        np.random.normal(25, 8, 40), np.random.normal(25, 8, 40),
        np.random.normal(55, 10, 40), np.random.normal(85, 10, 40),
        np.random.normal(85, 10, 40)
    ])
    spending_raw = np.concatenate([
        np.random.normal(20, 10, 40), np.random.normal(75, 10, 40),
        np.random.normal(50, 10, 40), np.random.normal(80, 10, 40),
        np.random.normal(20, 10, 40)
    ])
    age_raw    = np.random.randint(18, 70, n)
    gender_raw = np.random.choice(["Male", "Female"], n)
    income_c   = np.clip(income_raw, 15, 137)
    spending_c = np.clip(spending_raw, 1, 100)
    X = np.column_stack([income_c, spending_c])
    sc = StandardScaler()
    km = KMeans(n_clusters=5, random_state=42, n_init=15)
    km.fit(sc.fit_transform(X))
    labels  = km.predict(sc.transform(X))
    df = pd.DataFrame({
        'Income': income_c.round(1), 'Spending': spending_c.round(1),
        'Age': age_raw, 'Gender': gender_raw, 'Cluster': labels
    })
    centers = sc.inverse_transform(km.cluster_centers_)
    meta = [
        {'name': 'Budget Enthusiasts', 'short': 'Budget',   'tag': 'Low income · Low spend',
         'color': '#E53935', 'pale': '#FFEBEE', 'tint': '#EF9A9A',
         'strategy': 'Flash sales & price alerts'},
        {'name': 'Impulsive Spenders',  'short': 'Impulsive','tag': 'Low income · High spend',
         'color': '#F59E0B', 'pale': '#FEF3C7', 'tint': '#FCD34D',
         'strategy': 'BNPL & loyalty rewards'},
        {'name': 'Standard Customers',  'short': 'Standard', 'tag': 'Mid income · Mid spend',
         'color': '#00897B', 'pale': '#E0F2F1', 'tint': '#80CBC4',
         'strategy': 'Seasonal promos & newsletters'},
        {'name': 'Target Customers',    'short': 'Target',   'tag': 'High income · High spend',
         'color': '#1565C0', 'pale': '#E3F2FD', 'tint': '#90CAF9',
         'strategy': 'Premium bundles & VIP access'},
        {'name': 'Cautious Savers',     'short': 'Cautious', 'tag': 'High income · Low spend',
         'color': '#3949AB', 'pale': '#E8EAF6', 'tint': '#9FA8DA',
         'strategy': 'Value messaging & ROI offers'},
    ]
    return km, sc, df, centers, meta

km, sc, df, centers, meta = load_model_and_data()

def classify(income, spending):
    return int(km.predict(sc.transform(np.array([[income, spending]])))[0])

def CC():
    return dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Sans', color='#6B6F82', size=10.5))

GRID = '#EEEEF2'
TICK = dict(size=9, family='DM Mono', color='#A0A4B4')
AX   = dict(size=9, color='#A0A4B4', family='DM Mono')

# ── SHARED HEADER ─────────────────────────────────────────────
tabs_def = [
    ("overview",  "Overview"),
    ("profiler",  "Profiler"),
    ("segments",  "Segments"),
    ("simulator", "Simulator"),
    ("data",      "Data"),
]
page_labels = {k: v for k, v in tabs_def}

nav_html = "".join(
    f'<a class="nav-item {"active" if page==k else ""}" href="?page={k}" target="_self">'
    f'{"<span class=\'nav-dot\'></span>" if page==k else ""}{v}</a>'
    for k, v in tabs_def
)
st.markdown(f"""
<div class="header-band">
  <div class="header-left">
    <div class="header-logo">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <rect x="1" y="1" width="5" height="5" fill="#00897B"/>
        <rect x="8" y="1" width="5" height="5" fill="rgba(255,255,255,0.35)"/>
        <rect x="1" y="8" width="5" height="5" fill="rgba(255,255,255,0.18)"/>
        <rect x="8" y="8" width="5" height="5" fill="rgba(255,255,255,0.55)"/>
      </svg>
    </div>
    <span class="header-wordmark">SegmentIQ</span>
    <span class="header-version">v5.0</span>
  </div>
  <div class="header-nav">{nav_html}</div>
  <div class="header-right">
    <div class="status-pill"><span class="status-dot"></span>LIVE · 200 RECORDS</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Invisible streamlit buttons for navigation
_nc = st.columns(len(tabs_def))
for _c, (_k, _l) in zip(_nc, tabs_def):
    with _c:
        if st.button(_l, key=f"nav_{_k}"):
            go_to(_k)

# ── SUBHEADER ─────────────────────────────────────────────────
breadcrumb_map = {
    "overview":  ("Analysis", "Overview Dashboard"),
    "profiler":  ("Analysis", "Customer Profiler"),
    "segments":  ("Analysis", "Segment Deep Dive"),
    "simulator": ("Analysis", "What-If Simulator"),
    "data":      ("Analysis", "Data Explorer"),
}
bc_parent, bc_current = breadcrumb_map.get(page, ("Analysis", page.title()))
st.markdown(f"""
<div class="subheader">
  <div class="breadcrumb">
    <span>{bc_parent}</span>
    <span class="breadcrumb-sep">/</span>
    <b>{bc_current}</b>
  </div>
  <div class="ts">K-MEANS · 5 CLUSTERS · N_INIT=15 · SEED=42</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPER: shared scatter map
# ══════════════════════════════════════════════════════════════
def make_scatter(highlight=None, you=None, height=360):
    fig = go.Figure()
    for i, mi in enumerate(meta):
        mask = df['Cluster'] == i
        op = 0.75 if highlight is None or i == highlight else 0.15
        sz = 7.5 if highlight is None or i == highlight else 5
        fig.add_trace(go.Scatter(
            x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
            mode='markers',
            marker=dict(color=mi['color'], size=sz, opacity=op, line=dict(color='white', width=1)),
            name=mi['short'],
            hovertemplate=f'<b>{mi["name"]}</b><br>Income: %{{x:.0f}}k<br>Score: %{{y:.0f}}<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1], mode='markers',
        marker=dict(symbol='cross-thin', color='#0D0F14', size=16, line=dict(color='#0D0F14', width=2)),
        name='Centroids', hoverinfo='skip'
    ))
    if you is not None:
        inc, sp, color = you
        fig.add_trace(go.Scatter(
            x=[inc], y=[sp], mode='markers',
            marker=dict(symbol='star', color=color, size=22, line=dict(color='#0D0F14', width=1.5)),
            name='You', hovertemplate=f'You · {inc}k · Score {sp}<extra></extra>'
        ))
    fig.update_layout(
        **CC(),
        margin=dict(l=0, r=0, t=0, b=0), height=height,
        xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title_font=AX, showline=True, linecolor='#DDE0E8', linewidth=1),
        yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title_font=AX, showline=True, linecolor='#DDE0E8', linewidth=1),
        legend=dict(font=dict(size=9.5, family='DM Mono'), bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#DDE0E8', borderwidth=1, x=0.01, y=0.99, xanchor='left'),
    )
    return fig


def footer():
    st.markdown("""
    <div class="lab-footer">
      <div style="display:flex;gap:16px;">
        <span>SegmentIQ v5.0</span>
        <span style="color:#DDE0E8">·</span>
        <span>K-Means Clustering</span>
        <span style="color:#DDE0E8">·</span>
        <span>200 synthetic records · 5 segments · 2 features</span>
        <span style="color:#DDE0E8">·</span>
        <span>ML Internship · Task 2</span>
      </div>
      <div>sklearn 1.x · seed=42 · n_init=15</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div><div class="hero-title">Customer Segmentation<br><b>Overview</b></div></div>
      <div>
        <div class="hero-desc">Unsupervised K-Means clustering applied to income and spending score data. Five behavioural archetypes identified from 200 synthetic customer records.</div>
        <div class="hero-model-tag">sklearn.cluster.KMeans · StandardScaler · 2 features</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="s-label">Segment Breakdown</div>', unsafe_allow_html=True)
    kc = st.columns(5, gap="small")
    for i, (col, mi) in enumerate(zip(kc, meta)):
        cnt = int((df['Cluster']==i).sum())
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-accent-bar" style="background:{mi['color']};"></div>
              <div class="kpi-num">{cnt}</div>
              <div class="kpi-name">{mi['name']}</div>
              <div class="kpi-meta">{df[df['Cluster']==i]['Income'].mean():.0f}k avg income<br>score {df[df['Cluster']==i]['Spending'].mean():.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)

    col_map, col_right = st.columns([1.7, 1], gap="medium")
    with col_map:
        st.markdown('<div class="chart-card"><div class="chart-title">01 — Cluster Map</div><div class="chart-subtitle">Income versus Spending Score · all 200 records</div>', unsafe_allow_html=True)
        st.plotly_chart(make_scatter(height=360), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        avg_inc_all = df['Income'].mean()
        avg_sp_all  = df['Spending'].mean()
        avg_age_all = df['Age'].mean()
        st.markdown(f"""
        <div class="inline-metrics" style="margin-bottom:1rem;">
          <div class="inline-metric"><div class="im-val">{avg_inc_all:.1f}k</div><div class="im-lbl">Avg Income</div></div>
          <div class="inline-metric"><div class="im-val">{avg_sp_all:.1f}</div><div class="im-lbl">Avg Score</div></div>
          <div class="inline-metric"><div class="im-val">{avg_age_all:.0f}</div><div class="im-lbl">Avg Age</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="chart-card" style="padding-bottom:1.25rem;"><div class="chart-title">02 — Segment Reference</div><div class="chart-subtitle">Profile & recommended action</div>', unsafe_allow_html=True)
        st.markdown('<table class="seg-table"><thead><tr><th>Segment</th><th>N</th><th>Share</th><th>Action</th></tr></thead><tbody>', unsafe_allow_html=True)
        for i, mi in enumerate(meta):
            cnt = int((df['Cluster']==i).sum())
            pct = cnt / 200
            bar_w = int(pct * 80)
            st.markdown(f"""
            <tr>
              <td><div class="seg-name"><span class="seg-dot" style="background:{mi['color']};"></span>{mi['short']}</div><div class="seg-tag">{mi['tag']}</div></td>
              <td><div class="seg-count">{cnt}</div></td>
              <td><div class="seg-bar-wrap"><div class="seg-bar-fill" style="width:{bar_w}px;background:{mi['color']};opacity:.7;"></div></div><div class="seg-avg">{pct*100:.0f}%</div></td>
              <td><div class="strategy-cell">{mi['strategy']}</div></td>
            </tr>""", unsafe_allow_html=True)
        st.markdown('</tbody></table></div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    col_dist, col_age, col_live = st.columns(3, gap="medium")
    with col_dist:
        st.markdown('<div class="chart-card"><div class="chart-title">03 — Income Distribution</div><div class="chart-subtitle">Frequency by cluster</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        for i, mi in enumerate(meta):
            fig_hist.add_trace(go.Histogram(x=df[df['Cluster']==i]['Income'], nbinsx=14, name=mi['short'],
                marker=dict(color=mi['color'], opacity=0.72, line=dict(color='white', width=0.5)),
                hovertemplate=f'{mi["name"]}<br>%{{x:.0f}}k · %{{y}}<extra></extra>'))
        fig_hist.update_layout(**CC(), barmode='overlay', margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            legend=dict(font=dict(size=8.5, family='DM Mono'), bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.32, x=0), bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_age:
        st.markdown('<div class="chart-card"><div class="chart-title">04 — Age Profile</div><div class="chart-subtitle">Distribution per cluster</div>', unsafe_allow_html=True)
        fig_box = go.Figure()
        for i, mi in enumerate(meta):
            fig_box.add_trace(go.Box(y=df[df['Cluster']==i]['Age'], name=mi['short'],
                marker=dict(color=mi['color'], size=3), line=dict(color=mi['color'], width=1.5),
                fillcolor=mi['pale'], boxmean=True,
                hovertemplate=f'{mi["name"]}<br>Age: %{{y}}<extra></extra>'))
        fig_box.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9, family='DM Mono', color='#A0A4B4')),
            yaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_live:
        st.markdown('<div class="chart-card" style="padding-bottom:1.25rem;"><div class="chart-title">05 — Live Classifier</div><div class="chart-subtitle">Quick classify a profile</div>', unsafe_allow_html=True)
        inc_ov  = st.slider("Income (k$)", 15, 137, 65, key="ov_inc")
        sp_ov   = st.slider("Spending Score", 1, 100, 50, key="ov_sp")
        cl_ov   = classify(inc_ov, sp_ov)
        mo      = meta[cl_ov]
        st.markdown(f"""
        <div style="background:{mo['pale']};border:1px solid {mo['tint']};border-radius:6px;padding:.9rem 1rem;margin:.7rem 0 .5rem;">
          <div style="font-family:'DM Mono',monospace;font-size:.56rem;color:{mo['color']};text-transform:uppercase;letter-spacing:.14em;margin-bottom:3px;">Predicted Segment</div>
          <div style="font-family:'Instrument Serif',serif;font-size:1.3rem;font-style:italic;color:{mo['color']};">{mo['name']}</div>
          <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:{mo['color']};opacity:.7;margin-top:3px;">{mo['tag']}</div>
        </div>
        <div style="padding:8px 10px;background:var(--surface2);border:1px solid var(--border);border-radius:5px;margin-top:6px;">
          <div style="font-family:'DM Mono',monospace;font-size:.56rem;color:var(--ink4);text-transform:uppercase;letter-spacing:.12em;margin-bottom:2px;">Action</div>
          <div style="font-size:.73rem;font-weight:500;color:var(--ink2);">{mo['strategy']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    col_hm, col_sp = st.columns([1.4, 1], gap="medium")
    with col_hm:
        st.markdown('<div class="chart-card"><div class="chart-title">06 — Decision Boundary Map</div><div class="chart-subtitle">Full income × spending space coloured by cluster assignment</div>', unsafe_allow_html=True)
        h_grid = np.arange(15, 138, 4); s_grid = np.arange(1, 101, 4)
        Z = np.array([[classify(h, s) for h in h_grid] for s in s_grid])
        cs = [[0.00, meta[0]['pale']], [0.25, meta[1]['pale']], [0.50, meta[2]['pale']], [0.75, meta[3]['pale']], [1.00, meta[4]['pale']]]
        fig_hm = go.Figure(go.Heatmap(x=h_grid, y=s_grid, z=Z, colorscale=cs,
            hovertemplate='Income: %{x}k · Score: %{y} → Cluster %{z}<extra></extra>', showscale=False))
        fig_hm.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1], mode='markers+text',
            marker=dict(symbol='cross-thin', color='#0D0F14', size=14, line=dict(color='#0D0F14', width=2)),
            text=[mi['short'] for mi in meta], textposition='top center',
            textfont=dict(size=8, family='DM Mono', color='#0D0F14'), hoverinfo='skip', showlegend=False))
        fig_hm.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=260,
            xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'))
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sp:
        st.markdown('<div class="chart-card"><div class="chart-title">07 — Spending Score Profile</div><div class="chart-subtitle">Mean ± 1 std deviation per cluster</div>', unsafe_allow_html=True)
        means = [df[df['Cluster']==i]['Spending'].mean() for i in range(5)]
        stds  = [df[df['Cluster']==i]['Spending'].std()  for i in range(5)]
        fig_sp2 = go.Figure(go.Bar(
            x=[mi['short'] for mi in meta], y=means,
            error_y=dict(type='data', array=stds, visible=True, color='#6B6F82', thickness=1.2, width=4),
            marker=dict(color=[mi['color'] for mi in meta], opacity=0.8, cornerradius=4),
            text=[f'{m:.0f}' for m in means], textposition='outside',
            textfont=dict(size=9.5, family='DM Mono', color='#6B6F82'),
            hovertemplate='%{x}: %{y:.1f} ± %{error_y.array:.1f}<extra></extra>',
            showlegend=False, width=0.55))
        fig_sp2.update_layout(**CC(), margin=dict(l=0,r=0,t=10,b=0), height=260,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9.5, family='DM Mono', color='#6B6F82')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK, title="Spending Score",
                       title_font=AX, showline=True, linecolor='#DDE0E8', range=[0,115]), bargap=0.35)
        st.plotly_chart(fig_sp2, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — PROFILER
# ══════════════════════════════════════════════════════════════
elif page == "profiler":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div><div class="hero-title">Customer<br><b>Profiler</b></div></div>
      <div>
        <div class="hero-desc">Enter a customer profile to receive instant segment classification, behavioural insights, and a recommended marketing action.</div>
        <div class="hero-model-tag">Real-time K-Means prediction · 2-feature input</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="s-label">Customer Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card" style="padding-bottom:1.25rem;">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2,2,1,1])
    with c1: income   = st.slider("Annual Income (k$)", 15, 137, 65, key="pf_inc")
    with c2: spending = st.slider("Spending Score (1–100)", 1, 100, 50, key="pf_sp")
    with c3: age      = st.number_input("Age", min_value=18, max_value=80, value=35, key="pf_age")
    with c4: gender   = st.selectbox("Gender", ["Male", "Female"], key="pf_gen")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

    cluster = classify(income, spending)
    m = meta[cluster]
    cl_avg_inc = df[df['Cluster']==cluster]['Income'].mean()
    cl_avg_sp  = df[df['Cluster']==cluster]['Spending'].mean()
    cl_avg_age = df[df['Cluster']==cluster]['Age'].mean()

    st.markdown('<div class="s-label">Classification Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1, 2.2, 1.3], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="result-panel" style="border-left:3px solid {m['color']};">
          <div class="result-ghost">{cluster}</div>
          <div class="result-eyebrow">Cluster {cluster} of 5</div>
          <div class="result-name">{m['name']}</div>
          <div class="result-tag-line">{m['tag']}</div>
          <div style="margin-top:10px;display:inline-flex;align-items:center;gap:6px;
                      background:{m['pale']};border:1px solid {m['tint']};
                      border-radius:3px;padding:4px 10px;">
            <span style="width:7px;height:7px;border-radius:50%;background:{m['color']};display:inline-block;"></span>
            <span style="font-family:'DM Mono',monospace;font-size:.62rem;color:{m['color']};letter-spacing:.06em;">{m['short'].upper()}</span>
          </div>
          <div class="result-divider"></div>
        """, unsafe_allow_html=True)

        for i, mi in enumerate(meta):
            cnt = int((df['Cluster']==i).sum())
            is_a = i == cluster
            st.markdown(f"""
            <div class="cluster-row {"active" if is_a else ""}">
              <div class="cluster-row-name">
                <span style="width:7px;height:7px;border-radius:50%;background:{mi['color']};display:inline-block;"></span>
                <span style="{"color:var(--ink);font-weight:600;" if is_a else ""}">{mi['short']}</span>
              </div>
              <span class="cluster-row-ct">{cnt}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="chart-card"><div class="chart-title">Cluster Map · Your Position</div><div class="chart-subtitle">Star marker shows your input coordinates</div>', unsafe_allow_html=True)
        st.plotly_chart(make_scatter(highlight=cluster, you=(income, spending, m['color']), height=280),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

        # Radar
        st.markdown('<div class="chart-card"><div class="chart-title">Profile Radar</div><div class="chart-subtitle">Normalised dimensions vs cluster averages</div>', unsafe_allow_html=True)
        inc_norm  = (income - 15) / (137 - 15) * 100
        age_fit   = max(0, 100 - abs(age - cl_avg_age) * 3)
        sp_fit    = max(0, 100 - abs(spending - cl_avg_sp) * 2)
        cats = ['Income', 'Spending', 'Age Fit', 'Cluster Fit', 'Engagement']
        vals = [inc_norm, spending, age_fit, sp_fit, (inc_norm + spending) / 2]
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill='toself',
            fillcolor=f"rgba({int(m['color'][1:3],16)},{int(m['color'][3:5],16)},{int(m['color'][5:7],16)},0.12)",
            line=dict(color=m['color'], width=2),
            name='Profile',
        ))
        fig_rad.update_layout(**CC(), margin=dict(l=30,r=30,t=20,b=10), height=200,
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=8, family='DM Mono'), gridcolor=GRID, linecolor='#DDE0E8'),
                angularaxis=dict(tickfont=dict(size=9, family='DM Mono', color='#6B6F82')),
                bgcolor='rgba(0,0,0,0)'
            ), showlegend=False)
        st.plotly_chart(fig_rad, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        # Insight alerts
        tips = []
        if income < 35:
            tips.append(("bad","⚑","Low Income Bracket","Price sensitivity high — prioritise value messaging."))
        elif income < 70:
            tips.append(("warn","→","Mid Income Range","Balance quality with clear affordability signals."))
        else:
            tips.append(("ok","✓","High Income Bracket","Receptive to premium and aspirational products."))

        if spending < 30:
            tips.append(("bad","⚑","Low Spending Score","Disengaged — re-engagement campaigns required."))
        elif spending < 65:
            tips.append(("warn","→","Moderate Spender","Growth potential with targeted nudges."))
        else:
            tips.append(("ok","✓","High Spending Score","Active buyer — focus on upsell and retention."))

        if age < 30:
            tips.append(("info","◉","Young Demographic","Social proof and trend-driven offers perform well."))
        elif age > 55:
            tips.append(("info","◉","Mature Demographic","Trust, quality, and loyalty programmes resonate."))

        inc_d = income - cl_avg_inc
        tips.append(("ok" if abs(inc_d) < 10 else "warn", "◈", "Cluster Fit",
            f"Income {abs(inc_d):.0f}k {'above' if inc_d>0 else 'below'} cluster avg ({cl_avg_inc:.0f}k)"))

        sp_d = spending - cl_avg_sp
        tips.append(("ok" if abs(sp_d) < 10 else "warn", "◈", "Spend Fit",
            f"Score {abs(sp_d):.0f} pts {'above' if sp_d>0 else 'below'} cluster avg ({cl_avg_sp:.0f})"))

        for sev, ico, title, body in tips[:5]:
            st.markdown(f"""
            <div class="alert-row {sev}">
              <span class="alert-ico">{ico}</span>
              <div><div class="alert-t">{title}</div><div class="alert-b">{body}</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:10px;padding:10px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;">
          <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;">Recommended Action</div>
          <div style="font-size:.78rem;font-weight:500;color:var(--ink2);line-height:1.5;">{m['strategy']}</div>
        </div>""", unsafe_allow_html=True)

        # Gender/age context
        gender_ct = int((df[(df['Cluster']==cluster) & (df['Gender']==gender)].shape[0]))
        st.markdown(f"""
        <div style="margin-top:8px;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:6px;">
          <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;">Cluster Demographics</div>
          <div style="display:flex;gap:12px;">
            <div><div style="font-family:'DM Mono',monospace;font-size:1rem;font-weight:300;color:var(--ink);">{int(cl_avg_age)}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);text-transform:uppercase;">Avg Age</div></div>
            <div style="width:1px;background:var(--border);"></div>
            <div><div style="font-family:'DM Mono',monospace;font-size:1rem;font-weight:300;color:var(--ink);">{gender_ct}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);text-transform:uppercase;">{gender} in cluster</div></div>
            <div style="width:1px;background:var(--border);"></div>
            <div><div style="font-family:'DM Mono',monospace;font-size:1rem;font-weight:300;color:var(--ink);">{int((df['Cluster']==cluster).sum())}</div><div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);text-transform:uppercase;">Total</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — SEGMENTS
# ══════════════════════════════════════════════════════════════
elif page == "segments":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div><div class="hero-title">Segment<br><b>Deep Dive</b></div></div>
      <div>
        <div class="hero-desc">Detailed statistical analysis and behavioural profile for each of the five identified customer archetypes.</div>
        <div class="hero-model-tag">Per-cluster statistics · Income · Spending · Age · Gender</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    seg_sel = st.selectbox(
        "Select Segment to Analyse",
        options=list(range(5)),
        format_func=lambda i: f"Cluster {i} — {meta[i]['name']} · {meta[i]['tag']}",
        key="seg_sel"
    )
    m   = meta[seg_sel]
    sdf = df[df['Cluster']==seg_sel]

    st.markdown(f"""
    <div style="background:{m['pale']};border:1px solid {m['tint']};border-radius:8px;
                padding:1.25rem 1.6rem;margin:1rem 0 1.6rem;
                display:flex;align-items:center;justify-content:space-between;gap:2rem;">
      <div>
        <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:{m['color']};text-transform:uppercase;letter-spacing:.16em;margin-bottom:3px;">Cluster {seg_sel}</div>
        <div style="font-family:'Instrument Serif',serif;font-size:2rem;font-style:italic;color:{m['color']};letter-spacing:-0.03em;">{m['name']}</div>
        <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:{m['color']};opacity:.7;margin-top:2px;">{m['tag']}</div>
      </div>
      <div style="font-family:'DM Sans',sans-serif;font-size:.8rem;font-weight:300;color:var(--ink2);max-width:280px;line-height:1.6;text-align:right;">
        Recommended: <b style="font-weight:600;">{m['strategy']}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat tiles
    st.markdown('<div class="stat-row">', unsafe_allow_html=True)
    stats_data = [
        (f"{sdf['Income'].mean():.1f}k",    "Avg Income",    f"σ = {sdf['Income'].std():.1f}k · range {sdf['Income'].min():.0f}–{sdf['Income'].max():.0f}k"),
        (f"{sdf['Spending'].mean():.1f}",   "Avg Spending",  f"σ = {sdf['Spending'].std():.1f} · range {sdf['Spending'].min():.0f}–{sdf['Spending'].max():.0f}"),
        (f"{sdf['Age'].mean():.1f}",         "Avg Age",       f"range {sdf['Age'].min()}–{sdf['Age'].max()} yrs"),
        (f"{len(sdf)}",                      "Cluster Size",  f"{len(sdf)/200*100:.0f}% of total · {int((sdf['Gender']=='Female').sum())}F / {int((sdf['Gender']=='Male').sum())}M"),
    ]
    st.markdown("</div>", unsafe_allow_html=True)

    sc4 = st.columns(4, gap="small")
    for col, (val, lbl, sub) in zip(sc4, stats_data):
        with col:
            st.markdown(f"""
            <div class="stat-tile" style="border-top:3px solid {m['color']};">
              <div class="stat-val" style="color:{m['color']};">{val}</div>
              <div class="stat-lbl">{lbl}</div>
              <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="medium")
    with ch1:
        st.markdown('<div class="chart-card"><div class="chart-title">Income Distribution</div><div class="chart-subtitle">Histogram with cluster context overlay</div>', unsafe_allow_html=True)
        fig_hi = go.Figure()
        for i, mi in enumerate(meta):
            fig_hi.add_trace(go.Histogram(x=df[df['Cluster']==i]['Income'], nbinsx=18, name=mi['short'],
                marker=dict(color=mi['color'], opacity=0.18 if i!=seg_sel else 0.8, line=dict(color='white', width=0.5)),
                hovertemplate=f'{mi["name"]}<br>%{{x:.0f}}k · %{{y}}<extra></extra>'))
        fig_hi.update_layout(**CC(), barmode='overlay', margin=dict(l=0,r=0,t=0,b=0), height=210,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            showlegend=False, bargap=0.05)
        st.plotly_chart(fig_hi, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-card"><div class="chart-title">Spending Distribution</div><div class="chart-subtitle">Score frequency for selected cluster</div>', unsafe_allow_html=True)
        fig_si = go.Figure()
        for i, mi in enumerate(meta):
            fig_si.add_trace(go.Histogram(x=df[df['Cluster']==i]['Spending'], nbinsx=18, name=mi['short'],
                marker=dict(color=mi['color'], opacity=0.18 if i!=seg_sel else 0.8, line=dict(color='white', width=0.5)),
                hovertemplate=f'{mi["name"]}<br>Score %{{x:.0f}} · %{{y}}<extra></extra>'))
        fig_si.update_layout(**CC(), barmode='overlay', margin=dict(l=0,r=0,t=0,b=0), height=210,
            xaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            showlegend=False, bargap=0.05)
        st.plotly_chart(fig_si, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    ch3, ch4 = st.columns([1.5, 1], gap="medium")
    with ch3:
        st.markdown('<div class="chart-card"><div class="chart-title">Cluster Map · Segment Highlighted</div><div class="chart-subtitle">Selected segment is full-opacity; others dimmed</div>', unsafe_allow_html=True)
        st.plotly_chart(make_scatter(highlight=seg_sel, height=270), use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch4:
        st.markdown('<div class="chart-card"><div class="chart-title">Age × Spending Scatter</div><div class="chart-subtitle">Segment highlighted in selected cluster</div>', unsafe_allow_html=True)
        fig_as = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster']==i
            fig_as.add_trace(go.Scatter(
                x=df.loc[mask,'Age'], y=df.loc[mask,'Spending'], mode='markers',
                marker=dict(color=mi['color'], size=6 if i==seg_sel else 4,
                            opacity=0.75 if i==seg_sel else 0.15,
                            line=dict(color='white' if i==seg_sel else 'rgba(0,0,0,0)', width=0.8)),
                name=mi['short'],
                hovertemplate=f'{mi["name"]}<br>Age: %{{x}}<br>Spending: %{{y:.0f}}<extra></extra>'))
        fig_as.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=270,
            xaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            showlegend=False)
        st.plotly_chart(fig_as, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # All 5 segment cards overview
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="s-label">All Segments At A Glance</div>', unsafe_allow_html=True)
    g5 = st.columns(5, gap="small")
    for col, mi, i in zip(g5, meta, range(5)):
        cnt = int((df['Cluster']==i).sum())
        with col:
            st.markdown(f"""
            <div class="seg-card" style="{"border-color:"+m['tint']+";" if i==seg_sel else ""}">
              <div class="seg-card-accent" style="background:{mi['color']};"></div>
              <div class="seg-card-id" style="color:{mi['color']};">{i}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:{mi['color']};text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;">Cluster {i}</div>
              <div style="font-family:'Instrument Serif',serif;font-size:1.1rem;font-style:italic;color:var(--ink);margin-bottom:4px;line-height:1.2;">{mi['name']}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);margin-bottom:10px;">{mi['tag']}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.65rem;color:var(--ink3);">{cnt} customers</div>
              <div style="font-size:.67rem;color:var(--ink3);margin-top:4px;line-height:1.4;">{mi['strategy']}</div>
            </div>""", unsafe_allow_html=True)

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 4 — SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "simulator":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div><div class="hero-title">What-If<br><b>Simulator</b></div></div>
      <div>
        <div class="hero-desc">Set a baseline customer profile and observe how changes in income or spending drive segment reassignment across seven scenarios.</div>
        <div class="hero-model-tag">Scenario analysis · Income sweep · Boundary detection</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="s-label">Baseline Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card" style="padding-bottom:1.25rem;">', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1: base_inc = st.slider("Annual Income (k$)", 15, 137, 55, key="sim_inc")
    with b2: base_sp  = st.slider("Spending Score (1–100)", 1, 100, 50, key="sim_sp")
    st.markdown('</div>', unsafe_allow_html=True)

    base_cl = classify(base_inc, base_sp)
    bm = meta[base_cl]

    st.markdown(f"""
    <div style="background:{bm['pale']};border:1px solid {bm['tint']};border-radius:6px;
                padding:1rem 1.5rem;margin:1rem 0 1.5rem;
                display:flex;align-items:center;gap:2rem;">
      <div>
        <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:{bm['color']};text-transform:uppercase;letter-spacing:.14em;margin-bottom:2px;">Baseline Classification</div>
        <div style="font-family:'Instrument Serif',serif;font-size:1.8rem;font-style:italic;color:{bm['color']};">{bm['name']}</div>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:{bm['color']};opacity:.8;line-height:1.9;">Cluster {base_cl}<br>{bm['tag']}</div>
      <div style="margin-left:auto;font-size:.75rem;font-weight:400;color:var(--ink3);">{bm['strategy']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="s-label">7 Scenario Adjustments</div>', unsafe_allow_html=True)

    scenarios = [
        ("+10k Income",  min(base_inc+10,137), base_sp),
        ("+20k Income",  min(base_inc+20,137), base_sp),
        ("−10k Income",  max(base_inc-10, 15), base_sp),
        ("+20 Score",    base_inc, min(base_sp+20,100)),
        ("+40 Score",    base_inc, min(base_sp+40,100)),
        ("−20 Score",    base_inc, max(base_sp-20, 1)),
        ("Premium",      min(base_inc+25,137), min(base_sp+25,100)),
    ]

    st.markdown('<div class="sim-grid">', unsafe_allow_html=True)
    sim_cols = st.columns(7, gap="small")
    for col, (lbl, inc_s, sp_s) in zip(sim_cols, scenarios):
        c   = classify(inc_s, sp_s)
        mi  = meta[c]
        chg = c != base_cl
        with col:
            st.markdown(f"""
            <div class="sim-tile {"shifted" if chg else ""}">
              <div class="sim-lbl">{lbl}</div>
              <div class="sim-seg-name" style="color:{mi['color']};">{mi['name']}</div>
              <div class="sim-badge {"changed" if chg else "same"}">
                {"↳ SHIFTED" if chg else "· SAME"}
              </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    sw1, sw2 = st.columns(2, gap="medium")

    with sw1:
        st.markdown('<div class="chart-card"><div class="chart-title">Income Sweep</div><div class="chart-subtitle">Cluster assignment as income varies · spending fixed</div>', unsafe_allow_html=True)
        inc_range  = np.arange(15, 138, 2)
        seg_sweep  = [classify(i, base_sp) for i in inc_range]
        fig_sw = go.Figure()
        for i in range(5):
            mask = np.array(seg_sweep) == i
            if mask.any():
                fig_sw.add_trace(go.Scatter(
                    x=inc_range[mask], y=np.ones(mask.sum()) * i, mode='markers',
                    marker=dict(color=meta[i]['color'], size=10, opacity=0.8, symbol='square',
                                line=dict(color='white', width=1)),
                    name=meta[i]['short'],
                    hovertemplate=f'Income: %{{x}}k → <b>{meta[i]["name"]}</b><extra></extra>'))
        fig_sw.add_vline(x=base_inc, line=dict(color='#0D0F14', width=1.5, dash='dot'),
            annotation_text=f"  baseline", annotation_font=dict(size=9, family='DM Mono', color='#0D0F14'))
        fig_sw.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=220,
            xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Cluster ID", gridcolor=GRID, zeroline=False, dtick=1, tickfont=TICK, title_font=AX, range=[-0.5,4.5]),
            legend=dict(font=dict(size=9, family='DM Mono'), bgcolor='rgba(255,255,255,0.95)', bordercolor='#DDE0E8', borderwidth=1, orientation='h', y=-0.32))
        st.plotly_chart(fig_sw, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with sw2:
        st.markdown('<div class="chart-card"><div class="chart-title">Spending Sweep</div><div class="chart-subtitle">Cluster assignment as spending varies · income fixed</div>', unsafe_allow_html=True)
        sp_range  = np.arange(1, 101, 2)
        sp_sweep  = [classify(base_inc, s) for s in sp_range]
        fig_sw2 = go.Figure()
        for i in range(5):
            mask = np.array(sp_sweep) == i
            if mask.any():
                fig_sw2.add_trace(go.Scatter(
                    x=sp_range[mask], y=np.ones(mask.sum()) * i, mode='markers',
                    marker=dict(color=meta[i]['color'], size=10, opacity=0.8, symbol='square',
                                line=dict(color='white', width=1)),
                    name=meta[i]['short'],
                    hovertemplate=f'Spending: %{{x}} → <b>{meta[i]["name"]}</b><extra></extra>'))
        fig_sw2.add_vline(x=base_sp, line=dict(color='#0D0F14', width=1.5, dash='dot'),
            annotation_text=f"  baseline", annotation_font=dict(size=9, family='DM Mono', color='#0D0F14'))
        fig_sw2.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=220,
            xaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Cluster ID", gridcolor=GRID, zeroline=False, dtick=1, tickfont=TICK, title_font=AX, range=[-0.5,4.5]),
            legend=dict(font=dict(size=9, family='DM Mono'), bgcolor='rgba(255,255,255,0.95)', bordercolor='#DDE0E8', borderwidth=1, orientation='h', y=-0.32))
        st.plotly_chart(fig_sw2, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # 2D probability heatmap with baseline marked
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card"><div class="chart-title">Proximity Map · Baseline Position vs Decision Boundaries</div><div class="chart-subtitle">Cross-hair marks your baseline; shaded regions show cluster zones</div>', unsafe_allow_html=True)
    h_g = np.arange(15, 138, 4); s_g = np.arange(1, 101, 4)
    Z2 = np.array([[classify(h, s) for h in h_g] for s in s_g])
    cs2 = [[0.0, meta[0]['pale']], [0.25, meta[1]['pale']], [0.5, meta[2]['pale']], [0.75, meta[3]['pale']], [1.0, meta[4]['pale']]]
    fig_prox = go.Figure(go.Heatmap(x=h_g, y=s_g, z=Z2, colorscale=cs2,
        hovertemplate='Income: %{x}k · Score: %{y} → Cluster %{z}<extra></extra>', showscale=False))
    fig_prox.add_trace(go.Scatter(
        x=[base_inc], y=[base_sp], mode='markers',
        marker=dict(symbol='cross-thin', color='#0D0F14', size=20, line=dict(color='#0D0F14', width=3)),
        name='Baseline', hovertemplate=f'Baseline<br>{base_inc}k · Score {base_sp}<extra></extra>'))
    for lbl, inc_s, sp_s in scenarios:
        c2 = classify(inc_s, sp_s)
        fig_prox.add_trace(go.Scatter(
            x=[inc_s], y=[sp_s], mode='markers',
            marker=dict(symbol='circle', color=meta[c2]['color'], size=9, opacity=0.9, line=dict(color='white', width=1.5)),
            name=lbl, hovertemplate=f'{lbl}<br>{inc_s}k · {sp_s}<br>→ {meta[c2]["name"]}<extra></extra>'))
    fig_prox.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=280,
        xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
        yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
        legend=dict(font=dict(size=9, family='DM Mono'), bgcolor='rgba(255,255,255,0.95)', bordercolor='#DDE0E8', borderwidth=1, orientation='h', y=-0.16))
    st.plotly_chart(fig_prox, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 5 — DATA
# ══════════════════════════════════════════════════════════════
elif page == "data":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div><div class="hero-title">Data<br><b>Explorer</b></div></div>
      <div>
        <div class="hero-desc">Browse and filter the full synthetic dataset of 200 customer records with cluster labels. Download filtered results as CSV.</div>
        <div class="hero-model-tag">200 synthetic records · seed=42 · numpy · pandas</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary stat row
    f1, f2, f3, f4 = st.columns(4, gap="small")
    for col, (val, lbl, sub) in zip([f1,f2,f3,f4], [
        ("200", "Total Records", "Synthetic · numpy seed=42"),
        ("5",   "Segments",      "K-Means · n_clusters=5"),
        ("2",   "Features",      "Income · Spending Score"),
        (f"{int((df['Gender']=='Female').sum())}/{int((df['Gender']=='Male').sum())}", "F / M Split", "Randomly assigned"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-num" style="font-size:1.8rem;">{val}</div>
              <div class="kpi-name">{lbl}</div>
              <div class="kpi-meta">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="s-label">Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card" style="padding-bottom:1.25rem;">', unsafe_allow_html=True)

    fa, fb, fc, fd = st.columns([2, 1.5, 1.5, 1])
    with fa:
        seg_filter = st.multiselect("Segment", options=list(range(5)),
            format_func=lambda i: f"Cluster {i} — {meta[i]['name']}",
            default=list(range(5)), key="dt_seg")
    with fb:
        gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male","Female"], key="dt_gen")
    with fc:
        age_range = st.slider("Age Range", 18, 80, (18, 80), key="dt_age")
    with fd:
        sort_by = st.selectbox("Sort By", ["Income ↓", "Income ↑", "Spending ↓", "Spending ↑", "Age ↓", "Age ↑"], key="dt_sort")

    st.markdown('</div>', unsafe_allow_html=True)

    fdf = df[
        df['Cluster'].isin(seg_filter) &
        df['Gender'].isin(gender_filter) &
        df['Age'].between(age_range[0], age_range[1])
    ].copy()
    fdf['Segment'] = fdf['Cluster'].apply(lambda x: meta[x]['name'])

    sort_map = {"Income ↓":("Income",False),"Income ↑":("Income",True),"Spending ↓":("Spending",False),"Spending ↑":("Spending",True),"Age ↓":("Age",False),"Age ↑":("Age",True)}
    sk, sa = sort_map[sort_by]
    fdf = fdf.sort_values(sk, ascending=sa).reset_index(drop=True)

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin:.6rem 0 .5rem;">
      <div style="font-family:'DM Mono',monospace;font-size:.62rem;color:var(--ink3);">
        Showing <b style="color:var(--ink);">{min(100,len(fdf))}</b> of <b style="color:var(--ink);">{len(fdf)}</b> filtered records (200 total)
      </div>
    </div>
    """, unsafe_allow_html=True)

    rows = ""
    for _, row in fdf.head(100).iterrows():
        mi = meta[int(row['Cluster'])]
        rows += f"""
        <tr>
          <td style="font-family:'DM Mono',monospace;color:var(--ink3);">{int(row.name)+1}</td>
          <td style="font-family:'DM Mono',monospace;">{row['Income']:.1f}k</td>
          <td style="font-family:'DM Mono',monospace;">{row['Spending']:.1f}</td>
          <td style="font-family:'DM Mono',monospace;">{row['Age']}</td>
          <td>{row['Gender']}</td>
          <td>
            <span class="seg-chip" style="background:{mi['pale']};color:{mi['color']};border-color:{mi['tint']};">
              <span style="width:5px;height:5px;border-radius:50%;background:{mi['color']};display:inline-block;"></span>
              {mi['short']}
            </span>
          </td>
          <td style="font-size:.7rem;color:var(--ink3);">{mi['strategy']}</td>
        </tr>"""

    st.markdown(f"""
    <div class="dt-wrap">
      <table class="dt-table">
        <thead><tr><th>#</th><th>Income</th><th>Spending</th><th>Age</th><th>Gender</th><th>Segment</th><th>Recommended Action</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    if len(fdf) > 100:
        st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:.6rem;color:var(--ink4);margin-top:.4rem;text-align:center;">Showing first 100 of {len(fdf)} — download CSV for full dataset</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

    csv = fdf[['Income','Spending','Age','Gender','Segment','Cluster']].to_csv(index=False)
    st.download_button("⬇  Download Filtered CSV", data=csv, file_name="segmentiq_data.csv", mime="text/csv")

    # Distribution overview charts
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="s-label">Filtered Dataset Overview</div>', unsafe_allow_html=True)

    dv1, dv2, dv3 = st.columns(3, gap="medium")
    with dv1:
        st.markdown('<div class="chart-card"><div class="chart-title">Segment Counts (Filtered)</div><div class="chart-subtitle">Records per cluster in current filter</div>', unsafe_allow_html=True)
        counts_f = [int((fdf['Cluster']==i).sum()) for i in range(5)]
        fig_cf = go.Figure(go.Bar(
            x=[mi['short'] for mi in meta], y=counts_f,
            marker=dict(color=[mi['color'] for mi in meta], opacity=0.82, cornerradius=4),
            text=counts_f, textposition='outside',
            textfont=dict(size=10, family='DM Mono', color='#6B6F82'),
            hovertemplate='%{x}: %{y}<extra></extra>', showlegend=False, width=0.6))
        fig_cf.update_layout(**CC(), margin=dict(l=0,r=0,t=10,b=0), height=200,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9.5, family='DM Mono', color='#A0A4B4')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK, showline=True, linecolor='#DDE0E8'), bargap=0.3)
        st.plotly_chart(fig_cf, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with dv2:
        st.markdown('<div class="chart-card"><div class="chart-title">Income vs Spending (Filtered)</div><div class="chart-subtitle">Scatter of current filtered records</div>', unsafe_allow_html=True)
        fig_fsc = go.Figure()
        for i, mi in enumerate(meta):
            mask = fdf['Cluster']==i
            if mask.any():
                fig_fsc.add_trace(go.Scatter(
                    x=fdf.loc[mask,'Income'], y=fdf.loc[mask,'Spending'], mode='markers',
                    marker=dict(color=mi['color'], size=6, opacity=0.7, line=dict(color='white', width=0.8)),
                    name=mi['short'],
                    hovertemplate=f'{mi["name"]}<br>%{{x:.0f}}k · %{{y:.0f}}<extra></extra>'))
        fig_fsc.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            legend=dict(font=dict(size=9, family='DM Mono'), bgcolor='rgba(0,0,0,0)', borderwidth=0, orientation='h', y=-0.32))
        st.plotly_chart(fig_fsc, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with dv3:
        st.markdown('<div class="chart-card"><div class="chart-title">Age Distribution (Filtered)</div><div class="chart-subtitle">Histogram of selected records</div>', unsafe_allow_html=True)
        fig_fa = go.Figure(go.Histogram(x=fdf['Age'], nbinsx=20,
            marker=dict(color='#0D0F14', opacity=0.55, line=dict(color='white', width=0.5)),
            hovertemplate='Age %{x}: %{y} customers<extra></extra>', name='Age'))
        fig_fa.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor='#DDE0E8'),
            showlegend=False, bargap=0.08)
        st.plotly_chart(fig_fa, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    footer()
    st.markdown('</div>', unsafe_allow_html=True)
