"""
SegmentIQ — Customer Segmentation Intelligence
Clean & Clinical · Medical Data Lab Aesthetic
Single Page: Overview Dashboard
Run: streamlit run customer_seg_v4.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SegmentIQ · Overview",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
#  STYLES — Clean & Clinical / Medical Data Lab
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
    --rule:         #DDE0E8;
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

/* ── HEADER BAND ── */
.header-band {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 2.5rem;
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    min-height: 56px;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 20px;
    border-right: 1px solid var(--border);
    padding-right: 24px;
    margin-right: 4px;
}
.header-logo {
    width: 28px; height: 28px;
    background: var(--ink);
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.header-wordmark {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.02em;
}
.header-version {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--ink4);
    letter-spacing: 0.04em;
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 2px 7px;
    border-radius: 3px;
}
.header-nav {
    display: flex;
    align-items: stretch;
    gap: 0;
    flex: 1;
    padding: 0 1rem;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--ink3);
    border-bottom: 2px solid transparent;
    cursor: pointer;
    letter-spacing: 0.01em;
    text-decoration: none;
    transition: color .12s, border-color .12s;
    user-select: none;
}
.nav-item:hover { color: var(--ink); border-bottom-color: var(--border2); }
.nav-item.active {
    color: var(--ink);
    font-weight: 600;
    border-bottom-color: var(--teal);
}
.nav-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--teal);
    flex-shrink: 0;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-left: 20px;
    border-left: 1px solid var(--border);
    margin-left: 4px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--teal);
    background: var(--teal-pale);
    border: 1px solid var(--teal-mid);
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.status-dot {
    width: 5px; height: 5px;
    background: var(--teal);
    border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── SUBHEADER ── */
.subheader {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0.6rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.breadcrumb {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink4);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.breadcrumb b { color: var(--ink2); }
.breadcrumb-sep { color: var(--border2); }
.ts {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--ink4);
    letter-spacing: 0.06em;
}

/* ── SHELL ── */
.shell {
    max-width: 1280px;
    margin: 0 auto;
    padding: 2rem 2rem 4rem;
}

/* ── PAGE HERO ── */
.hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.6rem;
    font-style: italic;
    font-weight: 400;
    color: var(--ink);
    letter-spacing: -0.03em;
    line-height: 1.05;
}
.hero-title b {
    font-style: normal;
    font-weight: 400;
    color: var(--teal);
}
.hero-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: var(--ink3);
    max-width: 340px;
    text-align: right;
    line-height: 1.7;
    font-weight: 300;
}
.hero-model-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--ink4);
    margin-top: 4px;
    text-align: right;
    letter-spacing: 0.06em;
}

/* ── SECTION LABEL ── */
.s-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    font-weight: 500;
    color: var(--ink4);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.9rem;
}

/* ── KPI GRID ── */
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 8px 8px 0 0;
}
.kpi-num {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: 300;
    color: var(--ink);
    letter-spacing: -0.05em;
    line-height: 1;
    margin-top: 4px;
}
.kpi-name {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--ink2);
    letter-spacing: 0.01em;
    margin-top: 6px;
}
.kpi-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--ink4);
    margin-top: 3px;
    line-height: 1.6;
}

/* ── CHART CARD ── */
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.25rem 0.5rem;
}
.chart-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: var(--ink3);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.3rem;
}
.chart-subtitle {
    font-size: 0.72rem;
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    color: var(--ink2);
    margin-bottom: 0.8rem;
}

/* ── SEGMENT TABLE ── */
.seg-table { width: 100%; border-collapse: collapse; }
.seg-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    color: var(--ink4);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    padding: 0 0 8px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.seg-table td {
    padding: 10px 0;
    border-bottom: 1px solid var(--bg2);
    vertical-align: middle;
}
.seg-table tr:last-child td { border-bottom: none; }
.seg-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 7px;
    flex-shrink: 0;
}
.seg-name {
    font-size: 0.77rem;
    font-weight: 500;
    color: var(--ink);
    display: flex;
    align-items: center;
}
.seg-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--ink4);
    margin-top: 1px;
    padding-left: 15px;
}
.seg-count {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--ink2);
    font-weight: 400;
}
.seg-bar-wrap {
    height: 4px;
    background: var(--bg2);
    border-radius: 2px;
    overflow: hidden;
    width: 80px;
}
.seg-bar-fill { height: 100%; border-radius: 2px; }
.seg-avg {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--ink3);
}
.strategy-cell {
    font-size: 0.7rem;
    color: var(--ink3);
    font-weight: 300;
    max-width: 180px;
    line-height: 1.4;
}

/* ── ALERT ROW ── */
.alert-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 5px;
    margin-bottom: 6px;
    border: 1px solid transparent;
}
.alert-row.ok   { background: var(--teal-pale);  border-color: #B2DFDB; }
.alert-row.warn { background: var(--amber-pale);  border-color: #FDE68A; }
.alert-row.info { background: var(--blue-pale);   border-color: #BBDEFB; }
.alert-ico  { font-size: 12px; flex-shrink: 0; margin-top: 1px; }
.alert-t    { font-size: 0.72rem; font-weight: 600; color: var(--ink); }
.alert-b    { font-size: 0.65rem; color: var(--ink3); margin-top: 1px; line-height: 1.5; }

/* ── INLINE METRIC ── */
.inline-metrics {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--surface);
}
.inline-metric {
    flex: 1;
    padding: 0.9rem 1rem;
    border-right: 1px solid var(--border);
}
.inline-metric:last-child { border-right: none; }
.im-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    font-weight: 300;
    color: var(--ink);
    letter-spacing: -0.03em;
}
.im-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--ink4);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}

/* ── SLIDERS ── */
.stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    color: var(--ink3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: var(--border) !important;
    height: 2px !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--teal) !important;
    box-shadow: 0 0 0 3px rgba(0,137,123,0.15) !important;
    width: 14px !important; height: 14px !important;
}
.stSelectbox label, .stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    color: var(--ink3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    color: var(--ink) !important;
}
.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--ink) !important;
}

/* ── FOOTER ── */
.lab-footer {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--ink4);
    padding: 1.5rem 0 0.5rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.08em;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-left { display: flex; gap: 20px; }
.footer-sep  { color: var(--border); }

/* ── DIVIDER ── */
.rule { height: 1px; background: var(--border); margin: 1.8rem 0; }

/* Hide streamlit nav buttons */
div[data-testid="stHorizontalBlock"] button { display: none !important; }

/* Scrollbar */
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
    labels = km.predict(sc.transform(X))
    df = pd.DataFrame({
        'Income': income_c.round(1), 'Spending': spending_c.round(1),
        'Age': age_raw, 'Gender': gender_raw, 'Cluster': labels
    })
    centers = sc.inverse_transform(km.cluster_centers_)
    meta = [
        {'name': 'Budget Enthusiasts', 'short': 'Budget',    'tag': 'Low income · Low spend',
         'color': '#E53935', 'pale': '#FFEBEE', 'tint': '#EF9A9A',
         'strategy': 'Flash sales & price alerts'},
        {'name': 'Impulsive Spenders',  'short': 'Impulsive', 'tag': 'Low income · High spend',
         'color': '#F59E0B', 'pale': '#FEF3C7', 'tint': '#FCD34D',
         'strategy': 'BNPL & loyalty rewards'},
        {'name': 'Standard Customers',  'short': 'Standard',  'tag': 'Mid income · Mid spend',
         'color': '#00897B', 'pale': '#E0F2F1', 'tint': '#80CBC4',
         'strategy': 'Seasonal promos & newsletters'},
        {'name': 'Target Customers',    'short': 'Target',    'tag': 'High income · High spend',
         'color': '#1565C0', 'pale': '#E3F2FD', 'tint': '#90CAF9',
         'strategy': 'Premium bundles & VIP access'},
        {'name': 'Cautious Savers',     'short': 'Cautious',  'tag': 'High income · Low spend',
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

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="header-band">
  <div class="header-left">
    <div class="header-logo">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <rect x="1" y="1" width="5" height="5" fill="#00897B"/>
        <rect x="8" y="1" width="5" height="5" fill="rgba(255,255,255,0.4)"/>
        <rect x="1" y="8" width="5" height="5" fill="rgba(255,255,255,0.2)"/>
        <rect x="8" y="8" width="5" height="5" fill="rgba(255,255,255,0.6)"/>
      </svg>
    </div>
    <span class="header-wordmark">SegmentIQ</span>
    <span class="header-version">v4.0</span>
  </div>
  <div class="header-nav">
    <a class="nav-item active" href="#"><span class="nav-dot"></span>Overview</a>
    <a class="nav-item" href="#">Profiler</a>
    <a class="nav-item" href="#">Segments</a>
    <a class="nav-item" href="#">Simulator</a>
    <a class="nav-item" href="#">Data</a>
  </div>
  <div class="header-right">
    <div class="status-pill"><span class="status-dot"></span>LIVE · 200 RECORDS</div>
  </div>
</div>
<div class="subheader">
  <div class="breadcrumb">
    <span>Analysis</span>
    <span class="breadcrumb-sep">/</span>
    <b>Overview Dashboard</b>
  </div>
  <div class="ts">K-MEANS · 5 CLUSTERS · N_INIT=15 · SEED=42</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="shell">', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-title">Customer Segmentation<br><b>Overview</b></div>
  </div>
  <div>
    <div class="hero-desc">
      Unsupervised K-Means clustering applied to income<br>
      and spending score data. Five behavioural archetypes<br>
      identified from 200 synthetic customer records.
    </div>
    <div class="hero-model-tag">sklearn.cluster.KMeans · StandardScaler · 2 features</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ─────────────────────────────────────────────────
st.markdown('<div class="s-label">Segment Breakdown</div>', unsafe_allow_html=True)

kc = st.columns(5, gap="small")
for i, (col, mi) in enumerate(zip(kc, meta)):
    cnt     = int((df['Cluster']==i).sum())
    avg_inc = df[df['Cluster']==i]['Income'].mean()
    avg_sp  = df[df['Cluster']==i]['Spending'].mean()
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-accent-bar" style="background:{mi['color']};"></div>
          <div class="kpi-num">{cnt}</div>
          <div class="kpi-name">{mi['name']}</div>
          <div class="kpi-meta">
            {avg_inc:.0f}k avg income<br>
            score {avg_sp:.0f}
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)

# ── MAIN ROW ──────────────────────────────────────────────────
col_map, col_right = st.columns([1.7, 1], gap="medium")

with col_map:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">01 — Cluster Map</div>
      <div class="chart-subtitle">Income versus Spending Score · all 200 records</div>
    """, unsafe_allow_html=True)

    fig_map = go.Figure()
    for i, mi in enumerate(meta):
        mask = df['Cluster'] == i
        fig_map.add_trace(go.Scatter(
            x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
            mode='markers',
            marker=dict(color=mi['color'], size=7.5, opacity=0.7,
                        line=dict(color='white', width=1)),
            name=mi['short'],
            hovertemplate=(
                f'<span style="font-family:DM Mono">'
                f'<b>{mi["name"]}</b><br>'
                f'Income: %{{x:.0f}}k<br>'
                f'Score: %{{y:.0f}}'
                f'</span><extra></extra>'
            ),
        ))
    # Centroid markers
    fig_map.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1], mode='markers',
        marker=dict(symbol='cross-thin', color='#0D0F14', size=16,
                    line=dict(color='#0D0F14', width=2)),
        name='Centroids', hoverinfo='skip'
    ))
    fig_map.update_layout(
        **CC(),
        margin=dict(l=0, r=0, t=0, b=0), height=360,
        xaxis=dict(
            title="Annual Income (k$)", gridcolor=GRID, zeroline=False,
            tickfont=TICK, title_font=dict(size=9.5, color='#A0A4B4', family='DM Mono'),
            showline=True, linecolor='#DDE0E8', linewidth=1,
        ),
        yaxis=dict(
            title="Spending Score", gridcolor=GRID, zeroline=False,
            tickfont=TICK, title_font=dict(size=9.5, color='#A0A4B4', family='DM Mono'),
            showline=True, linecolor='#DDE0E8', linewidth=1,
        ),
        legend=dict(
            font=dict(size=9.5, family='DM Mono'), bgcolor='rgba(255,255,255,0.96)',
            bordercolor='#DDE0E8', borderwidth=1, x=0.01, y=0.99, xanchor='left',
            itemsizing='constant', tracegroupgap=2,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Inline summary metrics
    avg_inc_all = df['Income'].mean()
    avg_sp_all  = df['Spending'].mean()
    avg_age_all = df['Age'].mean()
    st.markdown(f"""
    <div class="inline-metrics" style="margin-bottom:1rem;">
      <div class="inline-metric">
        <div class="im-val">{avg_inc_all:.1f}k</div>
        <div class="im-lbl">Avg Income</div>
      </div>
      <div class="inline-metric">
        <div class="im-val">{avg_sp_all:.1f}</div>
        <div class="im-lbl">Avg Score</div>
      </div>
      <div class="inline-metric">
        <div class="im-val">{avg_age_all:.0f}</div>
        <div class="im-lbl">Avg Age</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Segment reference table
    st.markdown("""
    <div class="chart-card" style="padding-bottom:1.25rem;">
      <div class="chart-title">02 — Segment Reference</div>
      <div class="chart-subtitle">Profile & recommended action</div>
      <table class="seg-table">
        <thead>
          <tr>
            <th>Segment</th>
            <th>N</th>
            <th>Spread</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
    """, unsafe_allow_html=True)

    for i, mi in enumerate(meta):
        cnt     = int((df['Cluster']==i).sum())
        pct     = cnt / 200
        avg_inc = df[df['Cluster']==i]['Income'].mean()
        avg_sp  = df[df['Cluster']==i]['Spending'].mean()
        bar_w   = int(pct * 80)
        st.markdown(f"""
          <tr>
            <td>
              <div class="seg-name">
                <span class="seg-dot" style="background:{mi['color']};"></span>
                {mi['short']}
              </div>
              <div class="seg-tag">{mi['tag']}</div>
            </td>
            <td>
              <div class="seg-count">{cnt}</div>
            </td>
            <td>
              <div class="seg-bar-wrap">
                <div class="seg-bar-fill" style="width:{bar_w}px;background:{mi['color']};opacity:.7;"></div>
              </div>
              <div class="seg-avg" style="margin-top:3px;">{pct*100:.0f}%</div>
            </td>
            <td>
              <div class="strategy-cell">{mi['strategy']}</div>
            </td>
          </tr>
        """, unsafe_allow_html=True)

    st.markdown("</tbody></table></div>", unsafe_allow_html=True)

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

# ── SECOND ROW — Distributions + Live Classifier ──────────────
col_dist, col_age, col_classifier = st.columns([1, 1, 1], gap="medium")

with col_dist:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">03 — Income Distribution</div>
      <div class="chart-subtitle">Frequency by cluster</div>
    """, unsafe_allow_html=True)

    fig_hist = go.Figure()
    for i, mi in enumerate(meta):
        fig_hist.add_trace(go.Histogram(
            x=df[df['Cluster']==i]['Income'], nbinsx=14,
            name=mi['short'],
            marker=dict(color=mi['color'], opacity=0.75, line=dict(color='white', width=0.5)),
            hovertemplate=f'{mi["name"]}<br>%{{x:.0f}}k · count %{{y}}<extra></extra>',
        ))
    fig_hist.update_layout(
        **CC(), barmode='overlay',
        margin=dict(l=0,r=0,t=0,b=0), height=200,
        xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title_font=dict(size=9, color='#A0A4B4', family='DM Mono'),
                   showline=True, linecolor='#DDE0E8'),
        yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title_font=dict(size=9, color='#A0A4B4', family='DM Mono'),
                   showline=True, linecolor='#DDE0E8'),
        legend=dict(font=dict(size=8.5, family='DM Mono'), bgcolor='rgba(0,0,0,0)',
                    orientation='h', y=-0.28, x=0, borderwidth=0),
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_age:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">04 — Age Profile</div>
      <div class="chart-subtitle">Distribution per cluster</div>
    """, unsafe_allow_html=True)

    fig_box = go.Figure()
    for i, mi in enumerate(meta):
        fig_box.add_trace(go.Box(
            y=df[df['Cluster']==i]['Age'],
            name=mi['short'],
            marker=dict(color=mi['color'], size=3),
            line=dict(color=mi['color'], width=1.5),
            fillcolor=mi['pale'],
            boxmean=True,
            hovertemplate=f'{mi["name"]}<br>Age: %{{y}}<extra></extra>',
        ))
    fig_box.update_layout(
        **CC(),
        margin=dict(l=0,r=0,t=0,b=0), height=200,
        xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9, family='DM Mono', color='#A0A4B4')),
        yaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title_font=dict(size=9, color='#A0A4B4', family='DM Mono'),
                   showline=True, linecolor='#DDE0E8'),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_classifier:
    st.markdown("""
    <div class="chart-card" style="padding-bottom:1.25rem;">
      <div class="chart-title">05 — Live Classifier</div>
      <div class="chart-subtitle">Adjust inputs to classify</div>
    """, unsafe_allow_html=True)

    income_live   = st.slider("Income (k$)", 15, 137, 65, key="ov_inc")
    spending_live = st.slider("Spending Score", 1, 100, 50, key="ov_sp")

    cluster_live  = classify(income_live, spending_live)
    ml = meta[cluster_live]
    cluster_avg_inc = df[df['Cluster']==cluster_live]['Income'].mean()
    cluster_avg_sp  = df[df['Cluster']==cluster_live]['Spending'].mean()

    st.markdown(f"""
    <div style="background:{ml['pale']};border:1px solid {ml['tint']};border-radius:6px;
                padding:1rem 1.1rem;margin: 0.8rem 0 0.6rem;">
      <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:{ml['color']};
                  text-transform:uppercase;letter-spacing:.14em;margin-bottom:4px;">
        Predicted Segment
      </div>
      <div style="font-family:'Instrument Serif',serif;font-size:1.4rem;font-style:italic;
                  color:{ml['color']};letter-spacing:-0.02em;line-height:1.1;">
        {ml['name']}
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:{ml['color']};
                  opacity:.7;margin-top:4px;">{ml['tag']}</div>
    </div>
    """, unsafe_allow_html=True)

    inc_diff = income_live - cluster_avg_inc
    sp_diff  = spending_live - cluster_avg_sp

    tips = []
    if income_live > 70:
        tips.append(("ok","✓","High Income","Premium segment affinity."))
    elif income_live < 35:
        tips.append(("warn","⚑","Low Income","Price-sensitive profile."))
    else:
        tips.append(("info","→","Mid Income","Standard market segment."))

    tips.append(("info","◈","Cluster Δ",
                 f"Income {abs(inc_diff):.0f}k {'↑' if inc_diff>0 else '↓'} from cluster avg "
                 f"({cluster_avg_inc:.0f}k)"))

    for sev, ico, title, body in tips:
        st.markdown(f"""
        <div class="alert-row {sev}">
          <span class="alert-ico">{ico}</span>
          <div>
            <div class="alert-t">{title}</div>
            <div class="alert-b">{body}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:8px;padding:9px 11px;background:var(--surface2);
                border:1px solid var(--border);border-radius:5px;">
      <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:var(--ink4);
                  text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;">
        Recommended Action
      </div>
      <div style="font-size:.75rem;font-weight:500;color:var(--ink2);">
        {ml['strategy']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── BOTTOM ROW — Heatmap + Spending Bars ──────────────────────
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

col_hm, col_sb = st.columns([1.4, 1], gap="medium")

with col_hm:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">06 — Decision Boundary Map</div>
      <div class="chart-subtitle">Full income × spending input space coloured by cluster assignment</div>
    """, unsafe_allow_html=True)

    h_grid = np.arange(15, 138, 4)
    s_grid = np.arange(1, 101, 4)
    Z = np.array([[classify(h, s) for h in h_grid] for s in s_grid])
    colorscale = [
        [0.00, meta[0]['pale']], [0.25, meta[1]['pale']],
        [0.50, meta[2]['pale']], [0.75, meta[3]['pale']], [1.00, meta[4]['pale']]
    ]
    fig_hm = go.Figure(go.Heatmap(
        x=h_grid, y=s_grid, z=Z,
        colorscale=colorscale,
        hovertemplate='Income: %{x}k · Score: %{y} → Cluster %{z}<extra></extra>',
        showscale=False,
    ))
    # Overlay centroids
    fig_hm.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1], mode='markers+text',
        marker=dict(symbol='cross-thin', color='#0D0F14', size=14,
                    line=dict(color='#0D0F14', width=2)),
        text=[mi['short'] for mi in meta],
        textposition='top center',
        textfont=dict(size=8, family='DM Mono', color='#0D0F14'),
        hoverinfo='skip', showlegend=False,
    ))
    fig_hm.update_layout(
        **CC(),
        margin=dict(l=0,r=0,t=0,b=0), height=260,
        xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)',
                   tickfont=TICK, title_font=dict(size=9, color='#A0A4B4', family='DM Mono'),
                   showline=True, linecolor='#DDE0E8'),
        yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)',
                   tickfont=TICK, title_font=dict(size=9, color='#A0A4B4', family='DM Mono'),
                   showline=True, linecolor='#DDE0E8'),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_sb:
    st.markdown("""
    <div class="chart-card">
      <div class="chart-title">07 — Spending Score Profile</div>
      <div class="chart-subtitle">Mean score with ±1 std deviation</div>
    """, unsafe_allow_html=True)

    means   = [df[df['Cluster']==i]['Spending'].mean() for i in range(5)]
    stds    = [df[df['Cluster']==i]['Spending'].std()  for i in range(5)]
    names_s = [mi['short'] for mi in meta]
    colors_s= [mi['color'] for mi in meta]

    fig_sp = go.Figure()
    fig_sp.add_trace(go.Bar(
        x=names_s, y=means,
        error_y=dict(type='data', array=stds, visible=True,
                     color='#6B6F82', thickness=1.2, width=4),
        marker=dict(color=colors_s, opacity=0.8, cornerradius=4,
                    line=dict(color='white', width=0)),
        text=[f'{m:.0f}' for m in means], textposition='outside',
        textfont=dict(size=9.5, family='DM Mono', color='#6B6F82'),
        hovertemplate='%{x}: mean %{y:.1f} ± %{error_y.array:.1f}<extra></extra>',
        showlegend=False, width=0.55,
    ))
    fig_sp.update_layout(
        **CC(),
        margin=dict(l=0,r=0,t=10,b=0), height=260,
        xaxis=dict(gridcolor='rgba(0,0,0,0)',
                   tickfont=dict(size=9.5, family='DM Mono', color='#6B6F82')),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK,
                   title="Spending Score",
                   title_font=dict(size=9, color='#A0A4B4', family='DM Mono'),
                   showline=True, linecolor='#DDE0E8', range=[0, 110]),
        bargap=0.35,
    )
    st.plotly_chart(fig_sp, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="lab-footer">
  <div class="footer-left">
    <span>SegmentIQ v4.0</span>
    <span class="footer-sep">·</span>
    <span>K-Means Clustering</span>
    <span class="footer-sep">·</span>
    <span>200 synthetic records · 5 segments · 2 features</span>
    <span class="footer-sep">·</span>
    <span>ML Internship · Task 2</span>
  </div>
  <div>sklearn 1.x · seed=42 · n_init=15</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
