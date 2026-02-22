"""
SegmentIQ — Customer Segmentation Intelligence
Dark Luxury Terminal · New Design Identity
Run: streamlit run customer_seg_v2.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib, os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SegmentIQ",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── PAGE STATE ─────────────────────────────────────────────────
params = st.query_params
if "page" not in st.session_state:
    st.session_state.page = params.get("page", "segment")

def go_to(p):
    st.session_state.page = p
    st.query_params["page"] = p
    st.rerun()

# ══════════════════════════════════════════════════════════════
#  STYLES — Dark Luxury Terminal
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300;1,9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:          #0E0F14;
    --surface:     #151720;
    --surface2:    #1C1F2E;
    --border:      #272B3D;
    --border2:     #1F2234;
    --text:        #E8E9F0;
    --text2:       #9DA3BE;
    --text3:       #5A6180;
    --accent:      #E8A430;
    --accent-dim:  rgba(232,164,48,0.12);
    --accent-brd:  rgba(232,164,48,0.3);
    --green:       #34D399;
    --green-dim:   rgba(52,211,153,0.1);
    --red:         #F87171;
    --red-dim:     rgba(248,113,113,0.1);
    --blue:        #60A5FA;
    --blue-dim:    rgba(96,165,250,0.1);
    --purple:      #A78BFA;
    --purple-dim:  rgba(167,139,250,0.1);
    --teal:        #2DD4BF;
    --teal-dim:    rgba(45,212,191,0.1);
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; visibility: hidden !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SCANLINE TEXTURE ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,255,255,0.008) 2px,
        rgba(255,255,255,0.008) 4px
    );
    pointer-events: none;
}

/* ── NAV ── */
.nav {
    background: rgba(14,15,20,0.92);
    backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 1.2rem 2.5rem;
    position: sticky; top: 0; z-index: 999;
}
.nav-brand { display: flex; align-items: center; gap: 14px; }
.nav-wordmark {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1;
}
.nav-wordmark em {
    font-style: italic;
    font-weight: 300;
    color: var(--accent);
}
.nav-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 400;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent-brd);
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.08em;
}
.nav-links { display: flex; gap: 4px; align-items: center; }
.nav-link {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text3);
    padding: 6px 14px;
    border-radius: 6px;
    transition: all .15s;
    cursor: pointer;
    text-decoration: none;
    user-select: none;
    letter-spacing: 0.01em;
}
.nav-link:hover { color: var(--text); background: var(--surface2); }
.nav-link.active {
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-brd);
    font-weight: 600;
}

/* ── SHELL ── */
.shell {
    max-width: 1160px;
    margin: 0 auto;
    padding: 0.5rem 1.5rem 3rem;
    position: relative; z-index: 1;
}

/* ── PAGE HEADER ── */
.page-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.page-eyebrow::before {
    content: '▸';
    font-size: 0.65rem;
}
.page-title {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-size: 3rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.04em;
    line-height: 1.05;
    margin-bottom: 0.8rem;
}
.page-title em {
    font-style: italic;
    font-weight: 300;
    color: var(--accent);
}
.page-desc {
    font-size: 0.88rem;
    color: var(--text2);
    max-width: 480px;
    line-height: 1.75;
    font-weight: 400;
}

/* ── SECTION LABELS ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text3);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── CARD ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
}

/* ── RESULT BLOCK ── */
.result-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 1.75rem;
    box-shadow: 0 2px 24px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.result-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent) 0%, rgba(232,164,48,0.2) 60%, transparent 100%);
}
.result-glow {
    position: absolute;
    bottom: -50px; right: -50px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,164,48,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.cluster-numeral {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-size: 7rem;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.06em;
    color: var(--text);
}
.cluster-numeral .denom {
    font-size: 2rem;
    font-weight: 300;
    color: var(--text3);
    letter-spacing: -0.02em;
}
.cluster-sublabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 400;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 8px;
}
.seg-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    border-radius: 6px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.76rem;
    font-weight: 600;
    margin-top: 14px;
    border: 1px solid transparent;
}
.seg-divider {
    height: 1px;
    background: var(--border);
    margin: 1.2rem 0 0.9rem;
}
.seg-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 10px;
    border-radius: 7px;
    margin-bottom: 3px;
}
.seg-row.active { background: var(--surface2); }
.seg-row-name {
    font-size: 0.77rem;
    font-weight: 500;
    color: var(--text2);
}
.seg-row-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    color: var(--text3);
}

/* ── INSIGHT ── */
.insight {
    display: flex;
    gap: 11px;
    padding: 11px 13px;
    border-radius: 9px;
    margin-bottom: 6px;
    border: 1px solid transparent;
}
.insight.ok   { background: var(--green-dim);  border-color: rgba(52,211,153,0.2); }
.insight.warn { background: var(--accent-dim); border-color: var(--accent-brd); }
.insight.bad  { background: var(--red-dim);    border-color: rgba(248,113,113,0.2); }
.insight.info { background: var(--blue-dim);   border-color: rgba(96,165,250,0.2); }
.insight-ico  { font-size: 12px; margin-top: 2px; flex-shrink: 0; }
.insight-title { font-size: 0.77rem; font-weight: 700; color: var(--text); margin-bottom: 2px; }
.insight-body  { font-size: 0.7rem; color: var(--text2); line-height: 1.55; }

/* ── STRATEGY BOX ── */
.strategy-box {
    background: linear-gradient(135deg, var(--surface2) 0%, rgba(232,164,48,0.05) 100%);
    border: 1px solid var(--accent-brd);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.strategy-box::before {
    content: '';
    position: absolute;
    top: -20px; right: -20px;
    width: 80px; height: 80px;
    border-radius: 50%;
    background: rgba(232,164,48,0.06);
}
.strat-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 6px;
    opacity: 0.7;
}
.strat-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.4;
}
.strat-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text3);
    margin-top: 6px;
}

/* ── SIM CARDS ── */
.sim-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: border-color .2s, box-shadow .2s;
}
.sim-card:hover {
    border-color: var(--accent-brd);
    box-shadow: 0 4px 16px rgba(232,164,48,0.08);
}
.sim-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 400;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.sim-seg {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
}
.sim-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-top: 5px;
}
.pos { color: var(--green); }
.neg { color: var(--red); }
.neu { color: var(--text3); }

/* ── BASELINE BANNER ── */
.baseline-banner {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.75rem;
    margin: 1.5rem 0 2.25rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.baseline-seg {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.03em;
    font-style: italic;
}

/* ── STAT CARDS ── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.75rem;
}
.stat-val {
    font-family: 'Fraunces', serif;
    font-optical-sizing: auto;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.04em;
    margin-bottom: 6px;
    line-height: 1;
}
.stat-lbl {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stat-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    color: var(--text3);
    margin-top: 5px;
}

/* ── SLIDERS ── */
.stSlider label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: var(--border) !important;
    height: 3px !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,164,48,0.18) !important;
}
.stSelectbox label, .stNumberInput label, .stRadio label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
}
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}

/* ── FOOTER ── */
.app-footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--text3);
    padding: 2.5rem 0 1rem;
    margin-top: 4rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.08em;
}
.footer-sep { color: var(--border); margin: 0 10px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
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

    if os.path.exists('kmeans_model.pkl') and os.path.exists('customer_scaler.pkl'):
        km = joblib.load('kmeans_model.pkl')
        sc = joblib.load('customer_scaler.pkl')
    else:
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        km = KMeans(n_clusters=5, random_state=42, n_init=15)
        km.fit(X_sc)

    labels  = km.predict(sc.transform(X))
    df = pd.DataFrame({
        'Income': income_c, 'Spending': spending_c,
        'Age': age_raw, 'Gender': gender_raw, 'Cluster': labels
    })
    centers = sc.inverse_transform(km.cluster_centers_)

    meta = [
        {'name': 'Budget Enthusiasts', 'short': 'Budget',   'tag': 'Low income · Low spend',   'color': '#F87171', 'dim': 'rgba(248,113,113,0.1)',  'brd': 'rgba(248,113,113,0.25)', 'strategy': 'Flash sales, discount codes & price alerts'},
        {'name': 'Impulsive Spenders',  'short': 'Impulsive','tag': 'Low income · High spend',  'color': '#E8A430', 'dim': 'rgba(232,164,48,0.1)',   'brd': 'rgba(232,164,48,0.3)',  'strategy': 'Loyalty rewards, BNPL & curated picks'},
        {'name': 'Standard Customers',  'short': 'Standard', 'tag': 'Mid income · Mid spend',   'color': '#34D399', 'dim': 'rgba(52,211,153,0.1)',   'brd': 'rgba(52,211,153,0.25)', 'strategy': 'Seasonal promos & newsletter campaigns'},
        {'name': 'Target Customers',    'short': 'Target',   'tag': 'High income · High spend', 'color': '#60A5FA', 'dim': 'rgba(96,165,250,0.1)',   'brd': 'rgba(96,165,250,0.25)', 'strategy': 'Premium bundles & VIP early access'},
        {'name': 'Cautious Savers',     'short': 'Cautious', 'tag': 'High income · Low spend',  'color': '#A78BFA', 'dim': 'rgba(167,139,250,0.1)',  'brd': 'rgba(167,139,250,0.25)','strategy': 'Value messaging & exclusive high-ROI offers'},
    ]
    return km, sc, df, centers, meta

km, sc, df, centers, meta = load_model_and_data()

def classify(income, spending):
    return int(km.predict(sc.transform(np.array([[income, spending]])))[0])

CHART = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color='#5A6180', size=11),
)
GRID  = '#1F2234'
TICK  = dict(size=9.5, family='JetBrains Mono', color='#5A6180')

# ── NAV ────────────────────────────────────────────────────────
page  = st.session_state.page
pages = [("segment","Segment"), ("explorer","Explorer"), ("analytics","Analytics"), ("about","About")]

nav_links = "".join(
    f'<a class="nav-link {"active" if page==k else ""}" href="?page={k}" target="_self">{v}</a>'
    for k, v in pages
)
st.markdown(f"""
<div class="nav">
  <div class="nav-brand">
    <span class="nav-wordmark">Segment<em>IQ</em></span>
    <span class="nav-tag">K-Means · 5 Clusters</span>
  </div>
  <div class="nav-links">{nav_links}</div>
</div>
""", unsafe_allow_html=True)

_nc = st.columns(len(pages))
for _c, (_k, _l) in zip(_nc, pages):
    with _c:
        if st.button(_l, key=f"nav_{_k}"):
            go_to(_k)

# ══════════════════════════════════════════════════════════════
#  PAGE — SEGMENT
# ══════════════════════════════════════════════════════════════
if page == "segment":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin: 2.5rem 0 2rem;">
      <div class="page-eyebrow">Customer Classification</div>
      <div class="page-title">Customer<br><em>Segmentation</em></div>
      <div class="page-desc">Adjust the profile below — the segment prediction and insights update live as you move the sliders.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: income   = st.slider("Annual Income (k$)", 15, 137, 65, key="s_inc")
    with c2: spending = st.slider("Spending Score (1–100)", 1, 100, 50, key="s_sp")
    with c3: age      = st.number_input("Age", min_value=18, max_value=80, value=35, key="s_age")
    with c4: gender   = st.selectbox("Gender", ["Male", "Female"], key="s_gen")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Prediction Output</div>', unsafe_allow_html=True)

    cluster = classify(income, spending)
    m = meta[cluster]
    cluster_avg_inc = df[df['Cluster']==cluster]['Income'].mean()
    cluster_avg_sp  = df[df['Cluster']==cluster]['Spending'].mean()

    col_result, col_charts, col_insights = st.columns([1.1, 2.3, 1.4], gap="medium")

    # ── Result Block
    with col_result:
        st.markdown(f"""
        <div class="result-block">
          <div class="result-glow"></div>
          <div class="cluster-numeral">{cluster}<span class="denom"> /4</span></div>
          <div class="cluster-sublabel">Cluster ID · out of 5</div>
          <div class="seg-pill" style="background:{m['dim']};color:{m['color']};border-color:{m['brd']};">
            <span style="font-size:8px;">●</span> {m['name']}
          </div>
          <div class="seg-divider"></div>
        """, unsafe_allow_html=True)

        for i, mi in enumerate(meta):
            count  = int((df['Cluster'] == i).sum())
            active = "active" if i == cluster else ""
            name_color = mi['color'] if i == cluster else 'var(--text3)'
            st.markdown(f"""
            <div class="seg-row {active}">
              <span class="seg-row-name" style="color:{name_color};">
                <span style="color:{mi['color']};margin-right:6px;font-size:8px;">●</span>{mi['short']}
              </span>
              <span class="seg-row-count">{count}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts
    with col_charts:
        # Scatter map
        fig1 = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster'] == i
            fig1.add_trace(go.Scatter(
                x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
                mode='markers',
                marker=dict(color=mi['color'], size=7, opacity=0.65,
                            line=dict(color='rgba(255,255,255,0.1)', width=0.5)),
                name=mi['short'],
                hovertemplate=f'Income: %{{x:.0f}}k · Spending: %{{y:.0f}}<extra>{mi["name"]}</extra>',
            ))
        fig1.add_trace(go.Scatter(
            x=centers[:,0], y=centers[:,1], mode='markers',
            marker=dict(symbol='diamond', color=GRID, size=12,
                        line=dict(color='rgba(232,164,48,0.8)', width=1.5)),
            name='Centroids', hoverinfo='skip'
        ))
        fig1.add_trace(go.Scatter(
            x=[income], y=[spending], mode='markers',
            marker=dict(symbol='star', color=m['color'], size=20,
                        line=dict(color='white', width=1.5)),
            name='You',
            hovertemplate=f'You · {income}k · Spend {spending}<extra></extra>'
        ))
        fig1.update_layout(
            **CHART,
            margin=dict(l=0, r=16, t=12, b=6), height=250,
            xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
            legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)', borderwidth=0, orientation='h',
                        y=-0.18, x=0),
        )
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

        # Income vs Spending bar
        avg_inc = [df[df['Cluster']==i]['Income'].mean() for i in range(5)]
        avg_sp  = [df[df['Cluster']==i]['Spending'].mean() for i in range(5)]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[mi['short'] for mi in meta], y=avg_inc, name='Avg Income (k$)',
            marker=dict(color='#60A5FA', opacity=0.8, cornerradius=4, line=dict(width=0)),
            width=0.36, offsetgroup=0,
            hovertemplate='%{x} · Income: %{y:.0f}k<extra></extra>',
        ))
        fig2.add_trace(go.Bar(
            x=[mi['short'] for mi in meta], y=avg_sp, name='Avg Spending',
            marker=dict(color='#E8A430', opacity=0.8, cornerradius=4, line=dict(width=0)),
            width=0.36, offsetgroup=1,
            hovertemplate='%{x} · Spending: %{y:.0f}<extra></extra>',
        ))
        fig2.update_layout(
            **CHART, barmode='group',
            margin=dict(l=0, r=16, t=36, b=4), height=188,
            title=dict(text="Avg Income vs Spending by Cluster",
                       font=dict(size=11, color='#9DA3BE', family='Plus Jakarta Sans'), x=0),
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10, color='#9DA3BE')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK),
            legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)', borderwidth=0),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # ── Insights
    with col_insights:
        tips = []
        if income < 30:
            tips.append(("bad","⚠","Low Income Bracket","Price sensitivity expected — lead with value."))
        elif income < 60:
            tips.append(("warn","→","Mid Income Range","Balance quality messaging with clear value prop."))
        else:
            tips.append(("ok","✓","High Income Bracket","Receptive to premium & aspirational products."))

        if spending < 30:
            tips.append(("bad","⚠","Low Spending Score","Disengaged — re-engagement campaigns needed."))
        elif spending < 65:
            tips.append(("warn","→","Moderate Spender","Growth potential with targeted nudges."))
        else:
            tips.append(("ok","✓","High Spending Score","Active buyer — focus on upsell & retention."))

        if age < 30:
            tips.append(("info","★","Young Demographic","Social proof & trend-driven offers work best."))
        elif age > 55:
            tips.append(("info","★","Mature Demographic","Trust, quality & loyalty programs resonate."))

        inc_diff = income - cluster_avg_inc
        tips.append(("ok" if abs(inc_diff) < 10 else "warn",
                     "◈", "Cluster Fit",
                     f"Income is {abs(inc_diff):.0f}k {'above' if inc_diff>0 else 'below'} cluster avg ({cluster_avg_inc:.0f}k)."))

        for sev, ico, title, body in tips[:5]:
            st.markdown(f"""
            <div class="insight {sev}">
              <span class="insight-ico">{ico}</span>
              <div>
                <div class="insight-title">{title}</div>
                <div class="insight-body">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="strategy-box">
          <div class="strat-eyebrow">Recommended Strategy</div>
          <div class="strat-val">{m['strategy']}</div>
          <div class="strat-sub">{m['tag']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "explorer":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin: 2.5rem 0 2rem;">
      <div class="page-eyebrow">What-If Analysis</div>
      <div class="page-title">Segment <em>Explorer</em></div>
      <div class="page-desc">Set a baseline profile and observe how income or spending shifts drive cluster reassignment.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Baseline Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1: base_inc = st.slider("Annual Income (k$)", 15, 137, 55, key="e_inc")
    with e2: base_sp  = st.slider("Spending Score (1–100)", 1, 100, 50, key="e_sp")
    st.markdown('</div>', unsafe_allow_html=True)

    base_cluster = classify(base_inc, base_sp)
    bm = meta[base_cluster]

    st.markdown(f"""
    <div class="baseline-banner">
      <div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;margin-bottom:3px;">Baseline Segment</div>
        <div class="baseline-seg">{bm['name']}</div>
      </div>
      <div class="seg-pill" style="background:{bm['dim']};color:{bm['color']};border-color:{bm['brd']};">
        <span style="font-size:8px;">●</span> Cluster {base_cluster}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Adjustment Scenarios</div>', unsafe_allow_html=True)

    scenarios = [
        ("+10k Income",   min(base_inc+10,137), base_sp),
        ("+20k Income",   min(base_inc+20,137), base_sp),
        ("−10k Income",   max(base_inc-10, 15), base_sp),
        ("+20 Spending",  base_inc, min(base_sp+20,100)),
        ("+40 Spending",  base_inc, min(base_sp+40,100)),
        ("−20 Spending",  base_inc, max(base_sp-20,1)),
        ("Premium Move",  min(base_inc+25,137), min(base_sp+25,100)),
    ]

    sim_cols = st.columns(len(scenarios), gap="small")
    sim_segs = []
    for col, (lbl, inc_s, sp_s) in zip(sim_cols, scenarios):
        c   = classify(inc_s, sp_s)
        sim_segs.append(c)
        mi  = meta[c]
        changed = c != base_cluster
        dc  = "pos" if changed else "neu"
        with col:
            st.markdown(f"""
            <div class="sim-card">
              <div class="sim-lbl">{lbl}</div>
              <div class="sim-seg" style="color:{mi['color']};">{mi['short']}</div>
              <div class="sim-delta {dc}">{'↳ shifted' if changed else '· same'}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # Income sweep
    inc_range  = np.arange(15, 138, 2)
    seg_sweep  = [classify(i, base_sp) for i in inc_range]

    fig_sweep = go.Figure()
    for i in range(5):
        mask = np.array(seg_sweep) == i
        if mask.any():
            fig_sweep.add_trace(go.Scatter(
                x=inc_range[mask], y=np.array(seg_sweep)[mask],
                mode='markers',
                marker=dict(color=meta[i]['color'], size=9, opacity=0.8,
                            line=dict(color='rgba(255,255,255,0.1)', width=0.5)),
                name=meta[i]['short'],
                hovertemplate=f'Income: %{{x}}k → {meta[i]["name"]}<extra></extra>',
            ))
    fig_sweep.add_vline(x=base_inc, line=dict(color='#E8A430', width=1.5, dash='dot'))
    fig_sweep.add_annotation(
        x=base_inc, y=4.6, text=f"  baseline", showarrow=False,
        font=dict(color='#E8A430', size=10, family='JetBrains Mono'), xanchor='left'
    )
    fig_sweep.update_layout(
        **CHART,
        margin=dict(l=0,r=16,t=36,b=6), height=220,
        title=dict(text="Cluster Assignment · Income Sweep (Spending Fixed)",
                   font=dict(size=11, color='#9DA3BE', family='Plus Jakarta Sans'), x=0),
        xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False,
                   tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
        yaxis=dict(title="Cluster ID", gridcolor=GRID, zeroline=False, dtick=1,
                   tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
        legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)', borderwidth=0, orientation='h', y=-0.2),
    )
    st.plotly_chart(fig_sweep, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "analytics":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin: 2.5rem 0 2rem;">
      <div class="page-eyebrow">Data Analytics</div>
      <div class="page-title">Cluster <em>Analytics</em></div>
      <div class="page-desc">Distribution maps, age breakdown, and a full 2D density map across the input space.</div>
    </div>
    """, unsafe_allow_html=True)

    r1a, r1b = st.columns(2, gap="medium")

    with r1a:
        st.markdown('<div class="sec-label">Segmentation Map</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster'] == i
            fig_sc.add_trace(go.Scatter(
                x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
                mode='markers',
                marker=dict(color=mi['color'], size=7, opacity=0.65,
                            line=dict(color='rgba(255,255,255,0.08)', width=0.5)),
                name=mi['short'],
                hovertemplate='%{x:.0f}k · %{y:.0f}<extra>' + mi['name'] + '</extra>',
            ))
        fig_sc.add_trace(go.Scatter(
            x=centers[:,0], y=centers[:,1], mode='markers',
            marker=dict(symbol='diamond', color='#E8E9F0', size=11,
                        line=dict(color='rgba(232,164,48,0.9)', width=1.5)),
            name='Centroids', hoverinfo='skip'
        ))
        fig_sc.update_layout(
            **CHART, margin=dict(l=0,r=16,t=10,b=6), height=310,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False,
                       tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
            legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)', borderwidth=0),
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})

    with r1b:
        st.markdown('<div class="sec-label">Cluster Sizes</div>', unsafe_allow_html=True)
        counts = [int((df['Cluster']==i).sum()) for i in range(5)]
        fig_bar = go.Figure(go.Bar(
            x=[mi['short'] for mi in meta], y=counts,
            marker=dict(color=[mi['color'] for mi in meta], opacity=0.82,
                        cornerradius=5, line=dict(width=0)),
            text=[str(c) for c in counts], textposition='outside',
            textfont=dict(size=11, color='#9DA3BE', family='JetBrains Mono'),
            hovertemplate='%{x}: %{y} customers<extra></extra>',
            showlegend=False, width=0.62,
        ))
        fig_bar.update_layout(
            **CHART, margin=dict(l=0,r=16,t=10,b=6), height=310,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10.5, color='#9DA3BE')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    r2a, r2b = st.columns(2, gap="medium")

    with r2a:
        st.markdown('<div class="sec-label">Age Distribution by Cluster</div>', unsafe_allow_html=True)
        fig_age = go.Figure()
        for i, mi in enumerate(meta):
            ages = df[df['Cluster']==i]['Age']
            fig_age.add_trace(go.Box(
                y=ages, name=mi['short'],
                marker_color=mi['color'],
                line_color=mi['color'],
                fillcolor=mi['dim'],
                hovertemplate='%{y} yrs<extra>' + mi['name'] + '</extra>',
            ))
        fig_age.update_layout(
            **CHART, margin=dict(l=0,r=16,t=10,b=6), height=270,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10.5, color='#9DA3BE')),
            yaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK,
                       title_font=dict(size=10, color='#5A6180')),
            showlegend=False,
        )
        st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})

    with r2b:
        st.markdown('<div class="sec-label">Density Map · Income × Spending</div>', unsafe_allow_html=True)
        h_grid = np.arange(15, 138, 5)
        s_grid = np.arange(1, 101, 4)
        Z = np.array([[classify(h, s) for h in h_grid] for s in s_grid])
        fig_hm = go.Figure(go.Heatmap(
            x=h_grid, y=s_grid, z=Z,
            colorscale=[
                [0.00, '#2A1A1A'], [0.25, '#2A2010'],
                [0.50, '#0F2A1E'], [0.75, '#0F1E2A'], [1.00, '#1A0F2A']
            ],
            hovertemplate='Income: %{x}k · Spending: %{y} → Cluster %{z}<extra></extra>',
            colorbar=dict(tickfont=TICK, thickness=10, outlinewidth=0,
                          title=dict(text='Cluster', side='right',
                          font=dict(size=10, family='Plus Jakarta Sans', color='#5A6180'))),
        ))
        fig_hm.update_layout(
            **CHART, margin=dict(l=0,r=10,t=10,b=10), height=270,
            xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)',
                       tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
            yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)',
                       tickfont=TICK, title_font=dict(size=10, color='#5A6180')),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin: 2.5rem 0 2rem;">
      <div class="page-eyebrow">Documentation</div>
      <div class="page-title">About <em>SegmentIQ</em></div>
      <div class="page-desc">Model architecture, cluster reference, and feature documentation.</div>
    </div>
    """, unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4, gap="medium")
    for col, val, lbl, sub in [
        (s1, "200",     "Training Records", "Synthetic · seed 42"),
        (s2, "5",       "Clusters",         "K-Means · n_init=15"),
        (s3, "2",       "Input Features",   "Income + Spending Score"),
        (s4, "K-Means", "Algorithm",        "scikit-learn · sklearn"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-val">{val}</div>
              <div class="stat-lbl">{lbl}</div>
              <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Cluster Reference</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    hc = st.columns([0.4, 2, 1.5, 1.2, 3])
    for h, lbl in zip(hc, ["#", "Segment", "Income", "Spending", "Strategy"]):
        with h:
            st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;font-weight:500;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:.4rem 0;">{lbl}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:var(--border);margin:.3rem 0 .6rem;"></div>', unsafe_allow_html=True)

    inc_ranges = ["15–40k","15–40k","40–70k","70–137k","70–137k"]
    sp_ranges  = ["1–40","60–100","35–70","60–100","1–40"]
    for i, (mi, inc_r, sp_r) in enumerate(zip(meta, inc_ranges, sp_ranges)):
        bg = "background:var(--surface2);" if i%2==0 else ""
        fc = st.columns([0.4, 2, 1.5, 1.2, 3])
        data_items = [
            (str(i),       True,  mi['color']),
            (mi['name'],   False, mi['color']),
            (inc_r,        True,  'var(--text2)'),
            (sp_r,         True,  'var(--text2)'),
            (mi['strategy'],False,'var(--text2)'),
        ]
        for col, (txt, mono, col_c) in zip(fc, data_items):
            with col:
                ff = "font-family:'JetBrains Mono',monospace;font-size:.76rem;" if mono else "font-size:.79rem;"
                st.markdown(f'<div style="{ff}color:{col_c};padding:.44rem 0;{bg}">{txt}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Segment Cards</div>', unsafe_allow_html=True)
    gcols = st.columns(5, gap="medium")
    for col, mi, i in zip(gcols, meta, range(5)):
        count = int((df['Cluster']==i).sum())
        with col:
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid {mi['brd']};border-radius:12px;
                        padding:1.75rem 1rem;text-align:center;
                        box-shadow:0 2px 12px {mi['dim']};">
              <div style="font-family:'Fraunces',serif;font-size:2.8rem;font-weight:600;
                          color:{mi['color']};line-height:1;font-style:italic;">{i}</div>
              <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:.76rem;
                          font-weight:700;color:{mi['color']};margin-top:10px;">{mi['name']}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;
                          color:var(--text3);margin-top:6px;">{count} customers</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                          color:var(--text3);margin-top:4px;opacity:.7;">{mi['tag']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="shell" style="padding-top:0;padding-bottom:1rem;">
  <div class="app-footer">
    SegmentIQ v2.0
    <span class="footer-sep">·</span>K-Means Clustering
    <span class="footer-sep">·</span>200 customers
    <span class="footer-sep">·</span>5 segments
    <span class="footer-sep">·</span>Task 2 · ML Internship
  </div>
</div>
""", unsafe_allow_html=True)
