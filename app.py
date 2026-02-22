"""
SegmentIQ — Customer Segmentation Intelligence
Neo-Brutalist Design · Redesigned from scratch
Run: streamlit run customer_seg_v3.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib, os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SegmentIQ",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PAGE STATE ─────────────────────────────────────────────────
params = st.query_params
if "page" not in st.session_state:
    st.session_state.page = params.get("page", "dashboard")

def go_to(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

# ══════════════════════════════════════════════════════════════
#  STYLES — Neo-Brutalist / Soft Gradient Hybrid
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:           #F2EFE9;
    --bg2:          #E8E4DC;
    --ink:          #1A1814;
    --ink2:         #3D3A35;
    --ink3:         #7A766E;
    --ink4:         #A8A39A;
    --border:       #1A1814;
    --border-light: #C8C4BC;
    --yellow:       #F5D547;
    --yellow-pale:  #FBF1A8;
    --coral:        #F26B5B;
    --coral-pale:   #FDE8E5;
    --blue:         #3B82F6;
    --blue-pale:    #DBEAFE;
    --green:        #16A34A;
    --green-pale:   #DCFCE7;
    --purple:       #7C3AED;
    --purple-pale:  #EDE9FE;
    --teal:         #0D9488;
    --teal-pale:    #CCFBF1;
    --white:        #FFFFFF;
    --shadow:       4px 4px 0px #1A1814;
    --shadow-lg:    6px 6px 0px #1A1814;
    --shadow-sm:    2px 2px 0px #1A1814;
    --radius:       4px;
}

html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; visibility: hidden !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── TOP BAR ── */
.topbar {
    background: var(--ink);
    padding: 0.9rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid var(--ink);
    position: sticky; top: 0; z-index: 999;
}
.brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--white);
    letter-spacing: -0.04em;
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-dot {
    width: 10px; height: 10px;
    background: var(--yellow);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 3px rgba(245,213,71,0.25);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 3px rgba(245,213,71,0.25); }
    50%       { box-shadow: 0 0 0 6px rgba(245,213,71,0.1); }
}
.brand-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 400;
    color: var(--ink);
    background: var(--yellow);
    padding: 3px 8px;
    border-radius: 2px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.nav-tabs {
    display: flex;
    gap: 2px;
    align-items: center;
    background: rgba(255,255,255,0.06);
    padding: 4px;
    border-radius: 6px;
}
.nav-tab {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    padding: 6px 16px;
    border-radius: 4px;
    cursor: pointer;
    text-decoration: none;
    letter-spacing: 0.02em;
    transition: all .15s;
    user-select: none;
    border: 1px solid transparent;
}
.nav-tab:hover {
    color: var(--white);
    background: rgba(255,255,255,0.1);
}
.nav-tab.active {
    color: var(--ink);
    background: var(--yellow);
    font-weight: 700;
    border-color: var(--yellow);
}

/* ── SHELL ── */
.shell {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 2rem 4rem;
}

/* ── PAGE HEADER ── */
.pg-header {
    border-bottom: 3px solid var(--ink);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
}
.pg-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.05em;
    line-height: 0.95;
}
.pg-title span {
    color: var(--coral);
}
.pg-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink3);
    text-align: right;
    line-height: 1.8;
}

/* ── CARDS ── */
.bcard {
    background: var(--white);
    border: 2px solid var(--ink);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    transition: box-shadow .15s, transform .15s;
}
.bcard:hover {
    box-shadow: var(--shadow-lg);
    transform: translate(-1px,-1px);
}
.bcard-yellow { background: var(--yellow-pale); }
.bcard-coral  { background: var(--coral-pale); }
.bcard-blue   { background: var(--blue-pale); }
.bcard-green  { background: var(--green-pale); }

/* ── KPI CARDS ── */
.kpi-val {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.06em;
    line-height: 1;
}
.kpi-lbl {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--ink3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 8px;
}
.kpi-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink4);
    margin-top: 4px;
}

/* ── SECTION LABEL ── */
.sec-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--ink3);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--ink);
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-lbl::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--coral);
    border-radius: 50%;
    display: inline-block;
}

/* ── SEGMENT BADGE ── */
.seg-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border: 2px solid var(--ink);
    border-radius: 2px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    box-shadow: var(--shadow-sm);
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ── RESULT PANEL ── */
.result-panel {
    background: var(--white);
    border: 2px solid var(--ink);
    border-radius: var(--radius);
    box-shadow: var(--shadow-lg);
    padding: 2rem;
    position: relative;
    overflow: hidden;
}
.result-number {
    font-family: 'Syne', sans-serif;
    font-size: 8rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.08em;
    color: var(--ink);
    opacity: 0.08;
    position: absolute;
    bottom: -20px;
    right: -10px;
    pointer-events: none;
}
.result-cluster-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--ink3);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 4px;
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin: 8px 0 4px;
}
.result-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--ink3);
    margin-bottom: 16px;
}

/* ── INSIGHT CARDS ── */
.insight-strip {
    display: flex;
    gap: 8px;
    padding: 12px 14px;
    border: 2px solid var(--ink);
    border-radius: var(--radius);
    margin-bottom: 8px;
    box-shadow: var(--shadow-sm);
}
.insight-strip.ok     { background: var(--green-pale); }
.insight-strip.warn   { background: var(--yellow-pale); }
.insight-strip.bad    { background: var(--coral-pale); }
.insight-strip.info   { background: var(--blue-pale); }
.insight-ico          { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
.insight-t { font-size: 0.75rem; font-weight: 700; color: var(--ink); }
.insight-b { font-size: 0.68rem; color: var(--ink2); margin-top: 2px; line-height: 1.5; }

/* ── STRATEGY CARD ── */
.strat-card {
    background: var(--yellow);
    border: 2px solid var(--ink);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.2rem 1.4rem;
    margin-top: 12px;
}
.strat-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    font-weight: 600;
    color: var(--ink2);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 4px;
}
.strat-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.4;
}

/* ── SEGMENTS LIST ── */
.seg-row-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 12px;
    border: 2px solid var(--border-light);
    border-radius: var(--radius);
    margin-bottom: 5px;
    background: var(--bg);
    transition: border-color .15s;
}
.seg-row-item.active {
    border-color: var(--ink);
    background: var(--white);
    box-shadow: var(--shadow-sm);
}
.seg-row-name {
    font-size: 0.77rem;
    font-weight: 600;
    color: var(--ink2);
    display: flex;
    align-items: center;
    gap: 8px;
}
.seg-row-ct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--ink3);
    background: var(--bg2);
    padding: 2px 8px;
    border-radius: 2px;
}

/* ── SLIDERS ── */
.stSlider label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--ink2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: var(--border-light) !important;
    height: 3px !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--ink) !important;
    box-shadow: 2px 2px 0px var(--ink) !important;
}
.stSelectbox label, .stNumberInput label, .stRadio label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--ink2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: var(--white) !important;
    border: 2px solid var(--ink) !important;
    border-radius: var(--radius) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── DATA TABLE ── */
.dt-wrap { overflow-x: auto; }
.dt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}
.dt-table th {
    background: var(--ink);
    color: var(--white);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 10px 14px;
    text-align: left;
}
.dt-table td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border-light);
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ink2);
    font-size: 0.76rem;
}
.dt-table tr:nth-child(even) td { background: var(--bg); }
.dt-table tr:hover td { background: var(--yellow-pale); }

/* ── EXPLORER SIM CARDS ── */
.sim-tile {
    background: var(--white);
    border: 2px solid var(--ink);
    border-radius: var(--radius);
    padding: 1rem 0.75rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .15s, transform .15s;
}
.sim-tile:hover {
    box-shadow: var(--shadow);
    transform: translate(-1px,-1px);
}
.sim-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: var(--ink3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.sim-seg {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 800;
    color: var(--ink);
}
.sim-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    margin-top: 5px;
    font-weight: 600;
}
.changed  { color: var(--coral); }
.same     { color: var(--ink4); }

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: var(--ink4);
    padding: 2rem 0 0.5rem;
    margin-top: 4rem;
    border-top: 2px solid var(--ink);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.footer-sep { color: var(--border-light); margin: 0 8px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }

/* Hide default streamlit buttons in nav */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    display: none !important;
}
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
    X_sc = sc.fit_transform(X)
    km = KMeans(n_clusters=5, random_state=42, n_init=15)
    km.fit(X_sc)

    labels  = km.predict(sc.transform(X))
    df = pd.DataFrame({
        'Income': income_c.round(1), 'Spending': spending_c.round(1),
        'Age': age_raw, 'Gender': gender_raw, 'Cluster': labels
    })
    centers = sc.inverse_transform(km.cluster_centers_)

    meta = [
        {'name': 'Budget Enthusiasts', 'short': 'Budget',   'tag': 'Low income · Low spend',
         'color': '#F26B5B', 'pale': '#FDE8E5', 'accent': '#C0392B',
         'strategy': 'Flash sales, discount codes & price alerts', 'emoji': '🏷️'},
        {'name': 'Impulsive Spenders',  'short': 'Impulsive','tag': 'Low income · High spend',
         'color': '#F5A623', 'pale': '#FEF3DC', 'accent': '#D4860C',
         'strategy': 'Loyalty rewards, BNPL & curated picks', 'emoji': '⚡'},
        {'name': 'Standard Customers',  'short': 'Standard', 'tag': 'Mid income · Mid spend',
         'color': '#16A34A', 'pale': '#DCFCE7', 'accent': '#15803D',
         'strategy': 'Seasonal promos & newsletter campaigns', 'emoji': '📊'},
        {'name': 'Target Customers',    'short': 'Target',   'tag': 'High income · High spend',
         'color': '#3B82F6', 'pale': '#DBEAFE', 'accent': '#1D4ED8',
         'strategy': 'Premium bundles & VIP early access', 'emoji': '🎯'},
        {'name': 'Cautious Savers',     'short': 'Cautious', 'tag': 'High income · Low spend',
         'color': '#7C3AED', 'pale': '#EDE9FE', 'accent': '#5B21B6',
         'strategy': 'Value messaging & exclusive high-ROI offers', 'emoji': '💎'},
    ]
    return km, sc, df, centers, meta

km, sc, df, centers, meta = load_model_and_data()

def classify(income, spending):
    return int(km.predict(sc.transform(np.array([[income, spending]])))[0])

def chart_cfg():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Grotesk', color='#7A766E', size=11),
    )

GRID = '#E8E4DC'
TICK = dict(size=9.5, family='IBM Plex Mono', color='#7A766E')

# ── NAV ────────────────────────────────────────────────────────
page  = st.session_state.page
tabs_def = [
    ("dashboard", "◼ Overview"),
    ("profiler",  "◈ Profiler"),
    ("deepdive",  "◉ Segments"),
    ("simulator", "⟳ Simulator"),
    ("datatable", "≡ Data"),
]

nav_links = "".join(
    f'<a class="nav-tab {"active" if page==k else ""}" href="?page={k}" target="_self">{v}</a>'
    for k, v in tabs_def
)

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <span class="brand-dot"></span>
    SegmentIQ
    <span class="brand-badge">v3 · K-Means</span>
  </div>
  <div class="nav-tabs">{nav_links}</div>
</div>
""", unsafe_allow_html=True)

# Invisible streamlit buttons for JS-free navigation
_nc = st.columns(len(tabs_def))
for _c, (_k, _l) in zip(_nc, tabs_def):
    with _c:
        if st.button(_l.split(" ",1)[-1], key=f"nav_{_k}", use_container_width=True):
            go_to(_k)


# ══════════════════════════════════════════════════════════════
#  PAGE — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "dashboard":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pg-header">
      <div class="pg-title">Customer<br><span>Overview</span></div>
      <div class="pg-meta">
        200 records · 5 segments<br>
        K-Means · sklearn 1.x<br>
        seed = 42 · n_init = 15
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row
    kc = st.columns(5, gap="small")
    kpi_data = []
    for i, mi in enumerate(meta):
        cnt = int((df['Cluster']==i).sum())
        avg_inc = df[df['Cluster']==i]['Income'].mean()
        avg_sp  = df[df['Cluster']==i]['Spending'].mean()
        kpi_data.append((cnt, avg_inc, avg_sp, mi))

    for col, (cnt, avg_inc, avg_sp, mi) in zip(kc, kpi_data):
        with col:
            st.markdown(f"""
            <div class="bcard" style="background:{mi['pale']};padding:1.2rem;">
              <div class="kpi-val">{cnt}</div>
              <div class="kpi-lbl" style="color:{mi['accent']};">{mi['short']}</div>
              <div class="kpi-sub">
                avg {avg_inc:.0f}k income<br>
                score {avg_sp:.0f}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

    # ── Main charts
    ca, cb = st.columns([1.5, 1], gap="medium")

    with ca:
        st.markdown('<div class="sec-lbl">Segmentation Map — Income × Spending</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster'] == i
            fig1.add_trace(go.Scatter(
                x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
                mode='markers',
                marker=dict(color=mi['color'], size=8, opacity=0.75,
                            line=dict(color='white', width=1)),
                name=mi['short'],
                hovertemplate=f'<b>{mi["name"]}</b><br>Income: %{{x:.0f}}k<br>Score: %{{y:.0f}}<extra></extra>',
            ))
        fig1.add_trace(go.Scatter(
            x=centers[:,0], y=centers[:,1], mode='markers',
            marker=dict(symbol='diamond', color='#1A1814', size=14,
                        line=dict(color='#F5D547', width=2)),
            name='Centroids', hoverinfo='skip'
        ))
        fig1.update_layout(
            **chart_cfg(),
            margin=dict(l=0,r=0,t=0,b=0), height=380,
            xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#7A766E'),
                       showline=True, linecolor='#1A1814', linewidth=2),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#7A766E'),
                       showline=True, linecolor='#1A1814', linewidth=2),
            legend=dict(font=dict(size=10, family='Space Grotesk'), bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='#1A1814', borderwidth=2, x=0.01, y=0.99, xanchor='left'),
        )
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

    with cb:
        st.markdown('<div class="sec-lbl">Size Distribution</div>', unsafe_allow_html=True)
        counts = [int((df['Cluster']==i).sum()) for i in range(5)]
        names  = [mi['name'] for mi in meta]
        colors = [mi['color'] for mi in meta]

        fig_pie = go.Figure(go.Pie(
            labels=names, values=counts,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color='#1A1814', width=2)),
            textinfo='percent',
            textfont=dict(family='IBM Plex Mono', size=11),
            hovertemplate='<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>',
        ))
        fig_pie.add_annotation(
            text=f"<b>200</b><br><span style='font-size:10px'>customers</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, family='Syne', color='#1A1814'),
            align='center'
        )
        fig_pie.update_layout(
            **chart_cfg(),
            margin=dict(l=0,r=0,t=0,b=0), height=240,
            legend=dict(font=dict(size=10, family='Space Grotesk'),
                        bgcolor='rgba(0,0,0,0)', borderwidth=0, orientation='v'),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="sec-lbl" style="margin-top:1.2rem">Avg Metrics</div>', unsafe_allow_html=True)
        avg_inc_all = [df[df['Cluster']==i]['Income'].mean() for i in range(5)]
        avg_sp_all  = [df[df['Cluster']==i]['Spending'].mean() for i in range(5)]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[mi['short'] for mi in meta], y=avg_inc_all, name='Avg Income',
            marker=dict(color='#1A1814', cornerradius=3),
            width=0.36, offsetgroup=0,
        ))
        fig_bar.add_trace(go.Bar(
            x=[mi['short'] for mi in meta], y=avg_sp_all, name='Avg Spend',
            marker=dict(color='#F5D547', cornerradius=3, line=dict(color='#1A1814', width=1.5)),
            width=0.36, offsetgroup=1,
        ))
        fig_bar.update_layout(
            **chart_cfg(), barmode='group',
            margin=dict(l=0,r=0,t=0,b=0), height=160,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9, family='IBM Plex Mono', color='#7A766E')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK),
            legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)', orientation='h', y=1.1, x=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — CUSTOMER PROFILER
# ══════════════════════════════════════════════════════════════
elif page == "profiler":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pg-header">
      <div class="pg-title">Customer<br><span>Profiler</span></div>
      <div class="pg-meta">
        Real-time classification<br>
        Adjust sliders → instant result<br>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Input row
    st.markdown('<div class="sec-lbl">Enter Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="bcard">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([2,2,1,1])
    with c1: income   = st.slider("Annual Income (k$)", 15, 137, 65, key="p_inc")
    with c2: spending = st.slider("Spending Score (1–100)", 1, 100, 50, key="p_sp")
    with c3: age      = st.number_input("Age", min_value=18, max_value=80, value=35, key="p_age")
    with c4: gender   = st.selectbox("Gender", ["Male", "Female"], key="p_gen")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    cluster = classify(income, spending)
    m = meta[cluster]
    cluster_avg_inc = df[df['Cluster']==cluster]['Income'].mean()
    cluster_avg_sp  = df[df['Cluster']==cluster]['Spending'].mean()

    st.markdown('<div class="sec-lbl">Classification Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1, 2, 1.2], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="result-panel" style="border-left: 6px solid {m['color']};">
          <div class="result-number">{cluster}</div>
          <div class="result-cluster-id">Cluster {cluster} of 5</div>
          <div class="result-name">{m['name']}</div>
          <div class="result-tag">{m['tag']}</div>
          <div class="seg-badge" style="background:{m['pale']};color:{m['accent']};">
            {m['emoji']} {m['short']}
          </div>
          <div style="margin-top:1.2rem;">
        """, unsafe_allow_html=True)

        for i, mi in enumerate(meta):
            cnt = int((df['Cluster']==i).sum())
            is_active = i == cluster
            color_dot = mi['color']
            name_style = f"color:{mi['color']};font-weight:700;" if is_active else ""
            st.markdown(f"""
            <div class="seg-row-item {"active" if is_active else ""}">
              <div class="seg-row-name">
                <span style="width:8px;height:8px;border-radius:50%;background:{color_dot};display:inline-block;"></span>
                <span style="{name_style}">{mi['short']}</span>
              </div>
              <span class="seg-row-ct">{cnt}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with r2:
        fig_sc = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster'] == i
            opacity = 0.8 if i == cluster else 0.3
            size    = 8 if i == cluster else 6
            fig_sc.add_trace(go.Scatter(
                x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
                mode='markers',
                marker=dict(color=mi['color'], size=size, opacity=opacity,
                            line=dict(color='white', width=0.8)),
                name=mi['short'],
                hovertemplate=f'{mi["name"]}<br>%{{x:.0f}}k · %{{y:.0f}}<extra></extra>',
            ))
        fig_sc.add_trace(go.Scatter(
            x=[income], y=[spending], mode='markers',
            marker=dict(symbol='star', color=m['color'], size=22,
                        line=dict(color='#1A1814', width=2)),
            name='You',
            hovertemplate=f'You · {income}k · Spend {spending}<extra></extra>'
        ))
        fig_sc.update_layout(
            **chart_cfg(),
            margin=dict(l=0,r=0,t=0,b=0), height=280,
            xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#7A766E'),
                       showline=True, linecolor='#1A1814', linewidth=2),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#7A766E'),
                       showline=True, linecolor='#1A1814', linewidth=2),
            legend=dict(font=dict(size=9), bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='#C8C4BC', borderwidth=1, orientation='h', y=-0.18),
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})

        # Radar chart
        cats = ['Income', 'Spending', 'Age-Fit', 'Cluster Size', 'Engagement']
        inc_norm    = (income - 15) / (137 - 15) * 100
        sp_norm     = spending
        age_fit     = max(0, 100 - abs(age - df[df['Cluster']==cluster]['Age'].mean())*3)
        size_norm   = int((df['Cluster']==cluster).sum()) / 2
        engagement  = (inc_norm + sp_norm) / 2

        fig_rad = go.Figure(go.Scatterpolar(
            r=[inc_norm, sp_norm, age_fit, size_norm, engagement, inc_norm],
            theta=cats + [cats[0]],
            fill='toself',
            fillcolor=f"rgba{tuple(int(m['color'].lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.15,)}",
            line=dict(color=m['color'], width=2.5),
            name='Customer Profile',
        ))
        fig_rad.update_layout(
            **chart_cfg(),
            margin=dict(l=30,r=30,t=30,b=10), height=200,
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], tickfont=TICK,
                                gridcolor=GRID, linecolor='#C8C4BC'),
                angularaxis=dict(tickfont=dict(size=9, family='IBM Plex Mono', color='#7A766E')),
                bgcolor='rgba(0,0,0,0)',
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_rad, use_container_width=True, config={'displayModeBar': False})

    with r3:
        tips = []
        if income < 30:
            tips.append(("bad","⚠","Low Income","Price sensitivity high — lead with value prop."))
        elif income < 60:
            tips.append(("warn","→","Mid Income","Balance quality with clear affordability."))
        else:
            tips.append(("ok","✓","High Income","Receptive to premium products & services."))

        if spending < 30:
            tips.append(("bad","⚠","Low Spender","Disengaged — re-engagement campaigns needed."))
        elif spending < 65:
            tips.append(("warn","→","Moderate Spender","Growth potential with targeted nudges."))
        else:
            tips.append(("ok","✓","High Spender","Active buyer — upsell & retention focus."))

        if age < 30:
            tips.append(("info","★","Young Segment","Social proof & trend-driven messaging."))
        elif age > 55:
            tips.append(("info","★","Mature Segment","Trust, quality & loyalty programs."))

        inc_diff = income - cluster_avg_inc
        fit = abs(inc_diff)
        tips.append(("ok" if fit < 10 else "warn",
                     "◈", "Cluster Fit",
                     f"Income {abs(inc_diff):.0f}k {'above' if inc_diff>0 else 'below'} cluster avg ({cluster_avg_inc:.0f}k)"))

        for sev, ico, title, body in tips[:5]:
            st.markdown(f"""
            <div class="insight-strip {sev}">
              <span class="insight-ico">{ico}</span>
              <div>
                <div class="insight-t">{title}</div>
                <div class="insight-b">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="strat-card">
          <div class="strat-eyebrow">Recommended Action</div>
          <div class="strat-text">{m['strategy']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — SEGMENT DEEP DIVE
# ══════════════════════════════════════════════════════════════
elif page == "deepdive":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pg-header">
      <div class="pg-title">Segment<br><span>Deep Dive</span></div>
      <div class="pg-meta">
        Per-cluster statistics<br>
        Age, gender, income & spend<br>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Segment selector
    seg_sel = st.selectbox(
        "Select Segment",
        options=list(range(5)),
        format_func=lambda i: f"{meta[i]['emoji']} {meta[i]['name']} — {meta[i]['tag']}",
        key="dd_seg"
    )
    m = meta[seg_sel]
    seg_df = df[df['Cluster']==seg_sel]

    st.markdown(f"""
    <div style="background:{m['pale']};border:2px solid {m['color']};border-radius:4px;
                box-shadow:4px 4px 0px {m['color']};padding:1.2rem 1.5rem;margin:1.2rem 0 1.8rem;
                display:flex;align-items:center;gap:2rem;">
      <div style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;
                  color:{m['color']};letter-spacing:-0.05em;">{m['emoji']} {m['name']}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:{m['accent']};
                  opacity:.8;line-height:1.8;">{m['tag']}<br>{len(seg_df)} customers · Cluster {seg_sel}</div>
      <div style="margin-left:auto;font-family:'Space Grotesk',sans-serif;font-size:.8rem;
                  font-weight:600;color:{m['accent']};max-width:220px;">
        Strategy: {m['strategy']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat cards
    sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
    stats = [
        (f"{seg_df['Income'].mean():.1f}k", "Avg Income", f"σ = {seg_df['Income'].std():.1f}k"),
        (f"{seg_df['Spending'].mean():.1f}", "Avg Spending", f"σ = {seg_df['Spending'].std():.1f}"),
        (f"{seg_df['Age'].mean():.1f}", "Avg Age", f"range {seg_df['Age'].min()}–{seg_df['Age'].max()} yrs"),
        (f"{(seg_df['Gender']=='Female').sum()}", "Female", f"{(seg_df['Gender']=='Male').sum()} Male"),
    ]
    for col, (val, lbl, sub) in zip([sc1,sc2,sc3,sc4], stats):
        with col:
            st.markdown(f"""
            <div class="bcard" style="background:{m['pale']};padding:1.2rem;">
              <div class="kpi-val" style="color:{m['accent']};font-size:2.2rem;">{val}</div>
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    ca, cb = st.columns(2, gap="medium")

    with ca:
        st.markdown('<div class="sec-lbl">Income Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=seg_df['Income'], nbinsx=18,
            marker=dict(color=m['color'], opacity=0.85, line=dict(color='#1A1814', width=1)),
            name='Income',
            hovertemplate='Income: %{x:.0f}k<br>Count: %{y}<extra></extra>',
        ))
        fig_hist.update_layout(
            **chart_cfg(),
            margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK,
                       showline=True, linecolor='#1A1814', linewidth=2),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK,
                       showline=True, linecolor='#1A1814', linewidth=2),
            showlegend=False, bargap=0.08,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    with cb:
        st.markdown('<div class="sec-lbl">Spending Distribution</div>', unsafe_allow_html=True)
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Histogram(
            x=seg_df['Spending'], nbinsx=18,
            marker=dict(color='#1A1814', opacity=0.85, line=dict(color=m['color'], width=1.5)),
            name='Spending',
            hovertemplate='Spending: %{x:.0f}<br>Count: %{y}<extra></extra>',
        ))
        fig_sp.update_layout(
            **chart_cfg(),
            margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK,
                       showline=True, linecolor='#1A1814', linewidth=2),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK,
                       showline=True, linecolor='#1A1814', linewidth=2),
            showlegend=False, bargap=0.08,
        )
        st.plotly_chart(fig_sp, use_container_width=True, config={'displayModeBar': False})

    # Age vs spending scatter for this segment
    st.markdown('<div class="sec-lbl">Age × Spending — Cluster Context</div>', unsafe_allow_html=True)
    fig_age = go.Figure()
    for i, mi in enumerate(meta):
        mask = df['Cluster'] == i
        op = 0.85 if i == seg_sel else 0.15
        sz = 8   if i == seg_sel else 5
        fig_age.add_trace(go.Scatter(
            x=df.loc[mask,'Age'], y=df.loc[mask,'Spending'],
            mode='markers',
            marker=dict(color=mi['color'], size=sz, opacity=op,
                        line=dict(color='white' if i==seg_sel else 'rgba(0,0,0,0)', width=0.8)),
            name=mi['short'],
            hovertemplate=f'{mi["name"]}<br>Age: %{{x}}<br>Spending: %{{y:.0f}}<extra></extra>',
        ))
    fig_age.update_layout(
        **chart_cfg(),
        margin=dict(l=0,r=0,t=0,b=0), height=220,
        xaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   showline=True, linecolor='#1A1814', linewidth=2),
        yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   showline=True, linecolor='#1A1814', linewidth=2),
        legend=dict(font=dict(size=9), bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#C8C4BC', borderwidth=1, orientation='h', y=-0.2),
    )
    st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "simulator":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pg-header">
      <div class="pg-title">What-If<br><span>Simulator</span></div>
      <div class="pg-meta">
        Observe segment shifts<br>
        from income/spend changes<br>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Baseline Customer</div>', unsafe_allow_html=True)
    st.markdown('<div class="bcard">', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1: base_inc = st.slider("Annual Income (k$)", 15, 137, 55, key="sim_inc")
    with e2: base_sp  = st.slider("Spending Score (1–100)", 1, 100, 50, key="sim_sp")
    st.markdown('</div>', unsafe_allow_html=True)

    base_cluster = classify(base_inc, base_sp)
    bm = meta[base_cluster]

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1.5rem;margin:1.5rem 0 2rem;
                padding:1rem 1.5rem;background:{bm['pale']};
                border:2px solid {bm['color']};border-radius:4px;
                box-shadow:4px 4px 0px {bm['color']};">
      <div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:{bm['accent']};
                    text-transform:uppercase;letter-spacing:.14em;">Baseline Classification</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                    color:{bm['color']};letter-spacing:-0.04em;">{bm['emoji']} {bm['name']}</div>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:{bm['accent']};
                  line-height:1.8;">{bm['tag']}<br>Cluster {base_cluster}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Scenario Analysis — 7 Adjustments</div>', unsafe_allow_html=True)

    scenarios = [
        ("+10k Income",   min(base_inc+10,137), base_sp),
        ("+20k Income",   min(base_inc+20,137), base_sp),
        ("−10k Income",   max(base_inc-10, 15), base_sp),
        ("+20 Spending",  base_inc, min(base_sp+20,100)),
        ("+40 Spending",  base_inc, min(base_sp+40,100)),
        ("−20 Spending",  base_inc, max(base_sp-20,1)),
        ("Premium",       min(base_inc+25,137), min(base_sp+25,100)),
    ]

    sim_cols = st.columns(7, gap="small")
    for col, (lbl, inc_s, sp_s) in zip(sim_cols, scenarios):
        c   = classify(inc_s, sp_s)
        mi  = meta[c]
        changed = c != base_cluster
        with col:
            bg_c = mi['pale'] if changed else "#FFFFFF"
            brd_c = mi['color'] if changed else "#C8C4BC"
            st.markdown(f"""
            <div class="sim-tile" style="background:{bg_c};border-color:{brd_c};">
              <div class="sim-lbl">{lbl}</div>
              <div class="sim-seg" style="color:{mi['color']};">{mi['emoji']}</div>
              <div class="sim-seg" style="font-size:.8rem;">{mi['short']}</div>
              <div class="sim-delta {'changed' if changed else 'same'}">
                {'↳ SHIFTED' if changed else '· same'}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    # Income sweep line
    st.markdown('<div class="sec-lbl">Income Sweep (spending fixed at baseline)</div>', unsafe_allow_html=True)
    inc_range = np.arange(15, 138, 2)
    seg_sweep = [classify(i, base_sp) for i in inc_range]

    fig_sw = go.Figure()
    # Color background bands by segment
    for i in range(5):
        mask = np.array(seg_sweep) == i
        if mask.any():
            fig_sw.add_trace(go.Scatter(
                x=inc_range[mask], y=np.ones(mask.sum()) * i,
                mode='markers',
                marker=dict(color=meta[i]['color'], size=12, opacity=0.8, symbol='square',
                            line=dict(color='#1A1814', width=1)),
                name=meta[i]['short'],
                hovertemplate=f'Income: %{{x}}k → <b>{meta[i]["name"]}</b><extra></extra>',
            ))
    fig_sw.add_vline(x=base_inc, line=dict(color='#1A1814', width=2, dash='dot'),
                    annotation_text=f"  baseline ({base_inc}k)",
                    annotation_font=dict(size=10, family='IBM Plex Mono', color='#1A1814'))
    fig_sw.update_layout(
        **chart_cfg(),
        margin=dict(l=0,r=0,t=0,b=0), height=200,
        xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   showline=True, linecolor='#1A1814', linewidth=2),
        yaxis=dict(title="Cluster ID", gridcolor=GRID, zeroline=False, dtick=1, tickfont=TICK,
                   range=[-0.5, 4.5]),
        legend=dict(font=dict(size=9), bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#C8C4BC', borderwidth=1, orientation='h', y=-0.28),
    )
    st.plotly_chart(fig_sw, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — DATA TABLE
# ══════════════════════════════════════════════════════════════
elif page == "datatable":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pg-header">
      <div class="pg-title">Customer<br><span>Data</span></div>
      <div class="pg-meta">
        200 synthetic records<br>
        Full attribute view<br>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    st.markdown('<div class="sec-lbl">Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="bcard">', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        seg_filter = st.multiselect(
            "Segment",
            options=list(range(5)),
            format_func=lambda i: f"{meta[i]['emoji']} {meta[i]['name']}",
            default=list(range(5)),
            key="dt_seg"
        )
    with f2:
        gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male","Female"], key="dt_gen")
    with f3:
        age_range = st.slider("Age Range", 18, 80, (18, 80), key="dt_age")
    st.markdown('</div>', unsafe_allow_html=True)

    fdf = df[
        df['Cluster'].isin(seg_filter) &
        df['Gender'].isin(gender_filter) &
        df['Age'].between(age_range[0], age_range[1])
    ].copy()
    fdf['Segment'] = fdf['Cluster'].apply(lambda x: meta[x]['name'])

    st.markdown(f'<div style="margin:.8rem 0 .4rem;font-family:\'IBM Plex Mono\',monospace;font-size:.65rem;color:#7A766E;">Showing <b style="color:#1A1814;">{len(fdf)}</b> of 200 customers</div>', unsafe_allow_html=True)

    # Build HTML table
    rows = ""
    for _, row in fdf.head(80).iterrows():
        mi = meta[int(row['Cluster'])]
        rows += f"""
        <tr>
          <td><span style="font-family:'IBM Plex Mono',monospace;font-size:.75rem;">{row['Income']:.1f}k</span></td>
          <td><span style="font-family:'IBM Plex Mono',monospace;font-size:.75rem;">{row['Spending']:.1f}</span></td>
          <td>{row['Age']}</td>
          <td>{row['Gender']}</td>
          <td>
            <span style="background:{mi['pale']};color:{mi['accent']};border:1px solid {mi['color']};
                         padding:2px 10px;border-radius:2px;font-size:.68rem;font-weight:700;
                         font-family:'IBM Plex Mono',monospace;">
              {mi['emoji']} {mi['short']}
            </span>
          </td>
        </tr>"""

    st.markdown(f"""
    <div class="dt-wrap" style="background:#fff;border:2px solid #1A1814;border-radius:4px;
                                box-shadow:var(--shadow);overflow:hidden;">
      <table class="dt-table">
        <thead>
          <tr>
            <th>Income</th><th>Spending</th><th>Age</th><th>Gender</th><th>Segment</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    if len(fdf) > 80:
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.62rem;color:#A8A39A;margin-top:.5rem;text-align:center;">Showing top 80 of {len(fdf)} results</div>', unsafe_allow_html=True)

    # Download
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    csv = fdf[['Income','Spending','Age','Gender','Segment','Cluster']].to_csv(index=False)
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="segmentiq_data.csv",
        mime="text/csv",
        key="dl_csv"
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="shell" style="padding-top:0;padding-bottom:1rem;">
  <div class="app-footer">
    SegmentIQ v3.0
    <span class="footer-sep">·</span> K-Means Clustering
    <span class="footer-sep">·</span> 200 synthetic customers
    <span class="footer-sep">·</span> 5 segments
    <span class="footer-sep">·</span> ML Internship · Task 2
  </div>
</div>
""", unsafe_allow_html=True)
