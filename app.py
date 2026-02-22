"""
SegmentIQ — Customer Segmentation Intelligence
Glassmorphism · Soft Violet · Frosted Panels
5 Tabs: Overview · Profiler · Segments · Simulator · Data
Run: streamlit run customer_seg_v7.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="SegmentIQ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-a:        #0D0B1A;
    --bg-b:        #130F24;
    --bg-c:        #180D2E;
    --violet:      #8B5CF6;
    --violet2:     #A78BFA;
    --violet3:     #C4B5FD;
    --violet-dim:  rgba(139,92,246,0.15);
    --violet-brd:  rgba(139,92,246,0.3);
    --violet-glow: rgba(139,92,246,0.08);
    --pink:        #EC4899;
    --pink-dim:    rgba(236,72,153,0.12);
    --glass-bg:    rgba(255,255,255,0.04);
    --glass-bg2:   rgba(255,255,255,0.07);
    --glass-brd:   rgba(255,255,255,0.1);
    --glass-brd2:  rgba(255,255,255,0.16);
    --text:        #F1EEFF;
    --text2:       #B8AED8;
    --text3:       #7C6FA0;
    --text4:       #3D3560;
    --seg0:        #F87171;
    --seg1:        #FBBF24;
    --seg2:        #34D399;
    --seg3:        #60A5FA;
    --seg4:        #A78BFA;
    --seg0-d:      rgba(248,113,113,0.15);
    --seg1-d:      rgba(251,191,36,0.15);
    --seg2-d:      rgba(52,211,153,0.15);
    --seg3-d:      rgba(96,165,250,0.15);
    --seg4-d:      rgba(167,139,250,0.15);
}

html, body, [class*="css"], .stApp {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: var(--bg-a) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% -10%, rgba(139,92,246,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(236,72,153,0.1) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(139,92,246,0.05) 0%, transparent 70%) !important;
    background-attachment: fixed !important;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAV ── */
.gnav {
    background: rgba(13,11,26,0.7);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--glass-brd);
    padding: 0 2.5rem;
    display: flex; align-items: stretch;
    min-height: 58px;
    position: sticky; top: 0; z-index: 999;
}
.g-brand {
    display: flex; align-items: center; gap: 13px;
    padding-right: 28px; border-right: 1px solid var(--glass-brd);
    margin-right: 8px; min-width: 195px;
}
.g-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--violet), var(--pink));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 16px rgba(139,92,246,0.5);
    font-size: 14px; color: white; font-weight: 700;
    flex-shrink: 0;
}
.g-wordmark {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.02em;
}
.g-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.52rem; color: var(--text3);
    letter-spacing: 0.08em; margin-top: 1px;
}
.g-nav-links { display: flex; align-items: stretch; flex: 1; padding: 0 0.5rem; }
.g-nav-item {
    display: flex; align-items: center; gap: 7px;
    padding: 0 18px; font-size: 0.79rem; font-weight: 500;
    color: var(--text3); border-bottom: 2px solid transparent;
    text-decoration: none; cursor: pointer; user-select: none;
    transition: color .15s, border-color .15s; letter-spacing: 0.01em;
}
.g-nav-item:hover { color: var(--text2); border-bottom-color: rgba(139,92,246,0.4); }
.g-nav-item.active {
    color: var(--violet2);
    border-bottom-color: var(--violet);
    font-weight: 600;
}
.g-nav-pill {
    background: var(--violet-dim); border: 1px solid var(--violet-brd);
    padding: 2px 8px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.52rem; color: var(--violet2); letter-spacing: 0.06em;
}
.g-nav-right {
    display: flex; align-items: center; gap: 10px;
    border-left: 1px solid var(--glass-brd); padding-left: 20px; margin-left: 8px;
}
.live-badge {
    display: flex; align-items: center; gap: 6px;
    background: var(--glass-bg); border: 1px solid var(--glass-brd);
    padding: 5px 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: var(--text2); letter-spacing: 0.06em;
    backdrop-filter: blur(8px);
}
.live-dot {
    width: 6px; height: 6px;
    background: var(--seg2); border-radius: 50%;
    box-shadow: 0 0 6px var(--seg2);
    animation: glow-pulse 2s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%,100% { opacity:1; box-shadow: 0 0 6px var(--seg2); }
    50%      { opacity:.6; box-shadow: 0 0 12px var(--seg2); }
}

/* ── PAGE HEADER ── */
.g-page-header {
    padding: 2.4rem 2.5rem 2rem;
    display: flex; align-items: flex-end; justify-content: space-between;
    border-bottom: 1px solid var(--glass-brd);
    background: linear-gradient(180deg, rgba(139,92,246,0.05) 0%, transparent 100%);
}
.g-page-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.04em; line-height: 1.05;
}
.g-page-title span {
    background: linear-gradient(135deg, var(--violet2), var(--pink));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.g-page-desc {
    font-size: 0.8rem; font-weight: 300; color: var(--text2);
    max-width: 340px; text-align: right; line-height: 1.7;
}
.g-page-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: var(--text3);
    margin-top: 4px; text-align: right; letter-spacing: 0.06em;
}

/* ── SHELL ── */
.shell { max-width: 1280px; margin: 0 auto; padding: 2rem 2.5rem 4rem; }

/* ── SECTION LABEL ── */
.g-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 500;
    color: var(--text3); text-transform: uppercase; letter-spacing: 0.2em;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 10px;
}
.g-label::after { content:''; flex:1; height:1px; background: var(--glass-brd); }

/* ── GLASS CARD ── */
.gcard {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-brd);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative; overflow: hidden;
    transition: border-color .2s, box-shadow .2s;
}
.gcard::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.gcard:hover {
    border-color: var(--glass-brd2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(139,92,246,0.08);
}
.gcard-inner { background: var(--glass-bg); padding: 1.3rem; border-radius: 12px; padding-bottom: 0.4rem; }

/* ── KPI CARD ── */
.kpi-g {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-brd);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    position: relative; overflow: hidden;
}
.kpi-g::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.kpi-accent {
    position: absolute; top: 0; left: 0; bottom: 0; width: 3px; border-radius: 16px 0 0 16px;
}
.kpi-val {
    font-family: 'Sora', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.05em; line-height: 1;
    margin-top: 4px;
}
.kpi-name { font-size: 0.72rem; font-weight: 600; color: var(--text2); margin-top: 7px; letter-spacing: 0.01em; }
.kpi-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: var(--text3); margin-top: 3px; line-height: 1.6; }

/* ── CHART CARD LABELS ── */
.ct-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: var(--text3);
    text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 2px;
}
.ct-title {
    font-family: 'Sora', sans-serif;
    font-size: 1rem; font-weight: 600;
    color: var(--text2); margin-bottom: 0.9rem; letter-spacing: -0.02em;
}

/* ── RESULT CARD ── */
.result-g {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-brd);
    border-radius: 16px;
    padding: 1.75rem 1.5rem;
    position: relative; overflow: hidden;
}
.result-g::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.result-ghost {
    font-family: 'Sora', sans-serif; font-size: 8rem; font-weight: 700;
    color: white; opacity: 0.03;
    position: absolute; bottom: -20px; right: -10px;
    line-height: 1; pointer-events: none; user-select: none; letter-spacing: -0.06em;
}
.result-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    color: var(--violet2); text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 4px;
}
.result-name {
    font-family: 'Sora', sans-serif; font-size: 1.7rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.03em; line-height: 1.1; margin: 6px 0 4px;
}
.result-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--text3);
}
.seg-glass-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 20px; margin-top: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;
    border: 1px solid; backdrop-filter: blur(8px);
}
.r-line { height: 1px; background: var(--glass-brd); margin: 1rem 0 0.8rem; }
.cl-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 8px; border-radius: 8px; margin-bottom: 2px;
}
.cl-row.on { background: var(--violet-dim); border: 1px solid var(--violet-brd); }
.cl-name { font-size: 0.74rem; font-weight: 400; color: var(--text2); display: flex; align-items: center; gap: 7px; }
.cl-name.on { color: var(--violet2); font-weight: 600; }
.cl-ct { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--text3); }

/* ── INSIGHT PILLS ── */
.ins-g {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px; border-radius: 10px; margin-bottom: 6px;
    background: var(--glass-bg);
    border: 1px solid var(--glass-brd);
    backdrop-filter: blur(8px);
}
.ins-g.ok   { border-color: rgba(52,211,153,0.3);  background: rgba(52,211,153,0.06); }
.ins-g.warn { border-color: rgba(251,191,36,0.3);  background: rgba(251,191,36,0.06); }
.ins-g.bad  { border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.06); }
.ins-g.info { border-color: rgba(139,92,246,0.3);  background: rgba(139,92,246,0.07); }
.ins-ico  { font-size: 12px; flex-shrink:0; margin-top:1px; }
.ins-t    { font-size: 0.72rem; font-weight: 600; color: var(--text); }
.ins-b    { font-size: 0.65rem; color: var(--text2); margin-top: 1px; line-height: 1.5; }

/* ── STRATEGY GLASS ── */
.strat-g {
    background: var(--violet-dim);
    border: 1px solid var(--violet-brd);
    border-radius: 12px; padding: 1rem 1.2rem; margin-top: 10px;
    backdrop-filter: blur(8px);
}
.strat-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem;
    color: var(--violet2); text-transform: uppercase; letter-spacing: 0.16em;
    margin-bottom: 4px; opacity: .8;
}
.strat-text { font-size: 0.82rem; font-weight: 500; color: var(--text); line-height: 1.45; }

/* ── SIM TILES ── */
.sim-g {
    background: var(--glass-bg); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-brd); border-radius: 12px;
    padding: 0.9rem 0.6rem; text-align: center;
    transition: border-color .2s, background .2s;
}
.sim-g.shifted {
    background: var(--violet-dim);
    border-color: var(--violet-brd);
    box-shadow: 0 0 16px rgba(139,92,246,0.1);
}
.sim-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.56rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 7px; }
.sim-name { font-family: 'Sora', sans-serif; font-size: 0.85rem; font-weight: 600; color: var(--text); line-height: 1.2; }
.sim-dl { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; margin-top: 5px; }
.sim-dl.changed { color: var(--violet2); }
.sim-dl.same    { color: var(--text4); }

/* ── SEG CARDS ── */
.seg-g {
    background: var(--glass-bg); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-brd); border-radius: 14px; padding: 1.4rem;
    position: relative; overflow: hidden; transition: border-color .2s;
}
.seg-g::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.seg-g.selected { border-color: var(--violet-brd); box-shadow: 0 0 24px rgba(139,92,246,0.12); }
.seg-g-num {
    font-family: 'Sora', sans-serif; font-size: 5rem; font-weight: 800;
    opacity: 0.06; position: absolute; top: -10px; right: 10px;
    line-height: 1; user-select: none; letter-spacing: -0.06em;
}
.seg-g-id {
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; color: var(--text3);
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;
}
.seg-g-name {
    font-family: 'Sora', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: var(--text); line-height: 1.1; margin-bottom: 4px; letter-spacing: -0.02em;
}
.seg-g-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.57rem; color: var(--text3); margin-bottom: 10px; }
.seg-g-ct  { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--text2); }

/* ── DATA TABLE ── */
.dt-glass {
    background: var(--glass-bg); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-brd); border-radius: 14px; overflow: hidden;
}
.dt-glass table { width: 100%; border-collapse: collapse; }
.dt-glass th {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; font-weight: 500;
    color: var(--text3); text-transform: uppercase; letter-spacing: 0.14em;
    padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--glass-brd);
    background: rgba(255,255,255,0.03);
}
.dt-glass td { padding: 9px 16px; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.76rem; color: var(--text2); }
.dt-glass tr:last-child td { border-bottom: none; }
.dt-glass tr:hover td { background: rgba(255,255,255,0.03); }
.g-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    font-weight: 500; letter-spacing: 0.05em; border: 1px solid;
}

/* ── INLINE METRICS STRIP ── */
.g-strip {
    display: flex; gap: 0;
    background: var(--glass-bg); backdrop-filter: blur(12px);
    border: 1px solid var(--glass-brd); border-radius: 12px; overflow: hidden;
}
.g-strip-item { flex: 1; padding: 0.9rem 1rem; border-right: 1px solid var(--glass-brd); }
.g-strip-item:last-child { border-right: none; }
.gsi-v { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em; }
.gsi-l { font-family: 'JetBrains Mono', monospace; font-size: 0.56rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }

/* ── SLIDERS ── */
.stSlider label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important; font-weight: 400 !important;
    color: var(--text3) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
# div[data-testid="stSlider"] > div > div > div {
#     background: rgba(255,255,255,0.1) !important; height: 3px !important;
# }
# div[data-testid="stSlider"] > div > div > div > div {
#     background: linear-gradient(135deg, var(--violet), var(--violet2)) !important;
#     box-shadow: 0 0 8px rgba(139,92,246,0.5) !important;
#     width: 16px !important; height: 16px !important; border-radius: 50% !important;
# }
.stSelectbox label, .stNumberInput label, .stMultiSelect label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important; font-weight: 400 !important;
    color: var(--text3) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
.stSelectbox > div > div {
    background: var(--glass-bg) !important; border: 1px solid var(--glass-brd) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important; font-size: 0.82rem !important;
    backdrop-filter: blur(12px) !important;
}
.stNumberInput > div > div > input {
    background: var(--glass-bg) !important; border: 1px solid var(--glass-brd) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: var(--violet-dim) !important;
    border: 1px solid var(--violet-brd) !important;
    color: var(--violet2) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 0.08em !important;
    padding: 7px 20px !important; border-radius: 8px !important;
    backdrop-filter: blur(8px) !important;
}
.stDownloadButton > button:hover { background: rgba(139,92,246,0.22) !important; }

/* ── FOOTER ── */
.g-footer {
    border-top: 1px solid var(--glass-brd);
    padding: 1.4rem 0 0.5rem; margin-top: 3.5rem;
    display: flex; align-items: center; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    color: var(--text3); letter-spacing: 0.06em;
}

.rule { height: 1px; background: var(--glass-brd); margin: 2rem 0; }
div[data-testid="stHorizontalBlock"] button { display: none !important; }
::-webkit-scrollbar { width: 5px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ──────────────────────────────────────────────────────
@st.cache_resource
def load():
    np.random.seed(42)
    n = 200
    income_raw = np.concatenate([
        np.random.normal(25,8,40), np.random.normal(25,8,40),
        np.random.normal(55,10,40), np.random.normal(85,10,40), np.random.normal(85,10,40)
    ])
    spending_raw = np.concatenate([
        np.random.normal(20,10,40), np.random.normal(75,10,40),
        np.random.normal(50,10,40), np.random.normal(80,10,40), np.random.normal(20,10,40)
    ])
    age_raw    = np.random.randint(18, 70, n)
    gender_raw = np.random.choice(["Male","Female"], n)
    inc = np.clip(income_raw, 15, 137)
    spd = np.clip(spending_raw, 1, 100)
    X   = np.column_stack([inc, spd])
    sc  = StandardScaler()
    km  = KMeans(n_clusters=5, random_state=42, n_init=15)
    km.fit(sc.fit_transform(X))
    labels = km.predict(sc.transform(X))
    df = pd.DataFrame({'Income':inc.round(1),'Spending':spd.round(1),'Age':age_raw,'Gender':gender_raw,'Cluster':labels})
    centers = sc.inverse_transform(km.cluster_centers_)
    meta = [
        {'name':'Budget Enthusiasts','short':'Budget',   'tag':'Low income · Low spend',
         'color':'#F87171','dim':'rgba(248,113,113,0.12)','brd':'rgba(248,113,113,0.3)',
         'strategy':'Flash sales & price-drop alerts'},
        {'name':'Impulsive Spenders', 'short':'Impulsive','tag':'Low income · High spend',
         'color':'#FBBF24','dim':'rgba(251,191,36,0.12)','brd':'rgba(251,191,36,0.3)',
         'strategy':'BNPL options & curated impulse picks'},
        {'name':'Standard Customers', 'short':'Standard', 'tag':'Mid income · Mid spend',
         'color':'#34D399','dim':'rgba(52,211,153,0.12)','brd':'rgba(52,211,153,0.3)',
         'strategy':'Seasonal campaigns & email promos'},
        {'name':'Target Customers',   'short':'Target',  'tag':'High income · High spend',
         'color':'#60A5FA','dim':'rgba(96,165,250,0.12)','brd':'rgba(96,165,250,0.3)',
         'strategy':'Premium bundles & VIP early access'},
        {'name':'Cautious Savers',    'short':'Cautious', 'tag':'High income · Low spend',
         'color':'#A78BFA','dim':'rgba(167,139,250,0.12)','brd':'rgba(167,139,250,0.3)',
         'strategy':'Value messaging & exclusive ROI offers'},
    ]
    return km, sc, df, centers, meta

km, sc, df, centers, meta = load()

def classify(inc, spd):
    return int(km.predict(sc.transform(np.array([[inc, spd]])))[0])

def CC():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Manrope', color='#7C6FA0', size=10.5)
    )

GRID = 'rgba(255,255,255,0.06)'
TICK = dict(size=9, family='JetBrains Mono', color='#7C6FA0')
AX   = dict(size=9, color='#7C6FA0', family='JetBrains Mono')

# ── NAV ───────────────────────────────────────────────────────
tabs = [
    ("overview","Overview","01"),
    ("profiler","Profiler","02"),
    ("segments","Segments","03"),
    ("simulator","Simulator","04"),
    ("data","Data","05"),
]
nav_html = "".join(
    f'<a class="g-nav-item {"active" if page==k else ""}" href="?page={k}" target="_self">'
    f'{lbl} <span class="g-nav-pill">{n}</span></a>'
    for k, lbl, n in tabs
)
st.markdown(f"""
<div class="gnav">
  <div class="g-brand">
    <div class="g-logo">✦</div>
    <div>
      <div class="g-wordmark">SegmentIQ</div>
      <div class="g-tagline">K-MEANS · 5 CLUSTERS</div>
    </div>
  </div>
  <div class="g-nav-links">{nav_html}</div>
  <div class="g-nav-right">
    <div class="live-badge"><span class="live-dot"></span>200 RECORDS LIVE</div>
  </div>
</div>
""", unsafe_allow_html=True)

_nc = st.columns(len(tabs))
for _c, (_k,_l,_n) in zip(_nc, tabs):
    with _c:
        if st.button(_l, key=f"nav_{_k}"):
            go_to(_k)

# ── PAGE HEADERS ──────────────────────────────────────────────
page_headers = {
    "overview":  ("Customer","Overview",      "7 panels · cluster map · live classifier",   "sklearn · K-Means · seed=42"),
    "profiler":  ("Customer","Profiler",      "Real-time segment classification + radar",   "2-feature input · instant prediction"),
    "segments":  ("Segment", "Deep Dive",     "Per-cluster stats · distributions",          "Income · Spending · Age · Gender"),
    "simulator": ("What-If", "Simulator",     "Scenarios · income & spend sweeps",          "7 adjustments · boundary detection"),
    "data":      ("Data",    "Explorer",      "Browse · filter · sort · export",            "200 records · CSV download"),
}
ht1, ht2, hdesc, hmeta = page_headers[page]
st.markdown(f"""
<div class="g-page-header">
  <div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:var(--violet2);text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;opacity:.8;">{ht1}</div>
    <div class="g-page-title">{ht2}</div>
  </div>
  <div>
    <div class="g-page-desc">{hdesc}</div>
    <div class="g-page-meta">{hmeta}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SCATTER HELPER ────────────────────────────────────────────
def scatter(highlight=None, you=None, h=360):
    fig = go.Figure()
    for i, mi in enumerate(meta):
        mask = df['Cluster']==i
        op = 0.78 if highlight is None or i==highlight else 0.12
        sz = 7.5  if highlight is None or i==highlight else 5
        fig.add_trace(go.Scatter(
            x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'], mode='markers',
            marker=dict(color=mi['color'], size=sz, opacity=op,
                        line=dict(color='rgba(0,0,0,0.4)', width=0.8)),
            name=mi['short'],
            hovertemplate=f'<b>{mi["name"]}</b><br>Income: %{{x:.0f}}k<br>Score: %{{y:.0f}}<extra></extra>',
        ))
    fig.add_trace(go.Scatter(
        x=centers[:,0], y=centers[:,1], mode='markers',
        marker=dict(symbol='diamond', color='#A78BFA', size=12,
                    line=dict(color='rgba(0,0,0,0.5)', width=1.5)),
        name='Centroids', hoverinfo='skip'
    ))
    if you:
        ii, ss, col = you
        fig.add_trace(go.Scatter(
            x=[ii], y=[ss], mode='markers',
            marker=dict(symbol='star', color=col, size=24,
                        line=dict(color='rgba(0,0,0,0.5)', width=1.5)),
            name='You', hovertemplate=f'You · {ii}k · {ss}<extra></extra>'
        ))
    fig.update_layout(
        **CC(), margin=dict(l=0,r=0,t=0,b=0), height=h,
        xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False,
                   tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
        yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False,
                   tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
        legend=dict(font=dict(size=9.5, family='JetBrains Mono'),
                    bgcolor='rgba(13,11,26,0.7)', bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1, x=0.01, y=0.99),
    )
    return fig

def footer():
    st.markdown("""
    <div class="g-footer">
      <div style="display:flex;gap:16px;align-items:center;">
        <span style="color:var(--violet2);">✦</span>
        <span>SegmentIQ v7.0</span>
        <span style="color:var(--glass-brd)">·</span>
        <span>K-Means Clustering</span>
        <span style="color:var(--glass-brd)">·</span>
        <span>200 synthetic records · 5 segments</span>
        <span style="color:var(--glass-brd)">·</span>
        <span>ML Internship Task 2</span>
      </div>
      <div>sklearn 1.x · seed=42 · n_init=15</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  01 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "overview":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown('<div class="g-label">Segment Breakdown</div>', unsafe_allow_html=True)
    kc = st.columns(5, gap="small")
    for i, (col, mi) in enumerate(zip(kc, meta)):
        cnt = int((df['Cluster']==i).sum())
        with col:
            st.markdown(f"""
            <div class="kpi-g">
              <div class="kpi-accent" style="background:linear-gradient(180deg,{mi['color']},transparent);"></div>
              <div class="kpi-val" style="padding-left:8px;">{cnt}</div>
              <div class="kpi-name" style="color:{mi['color']};padding-left:8px;">{mi['name']}</div>
              <div class="kpi-meta" style="padding-left:8px;">{df[df['Cluster']==i]['Income'].mean():.0f}k income · score {df[df['Cluster']==i]['Spending'].mean():.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.7,1], gap="medium")
    with c1:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">01 — Cluster Map</div><div class="ct-title">Income vs Spending · All 200 Records</div>', unsafe_allow_html=True)
        st.plotly_chart(scatter(h=350), use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        ai = df['Income'].mean(); as_ = df['Spending'].mean(); aa = df['Age'].mean()
        st.markdown(f"""
        <div class="g-strip" style="margin-bottom:12px;">
          <div class="g-strip-item"><div class="gsi-v">{ai:.0f}k</div><div class="gsi-l">Avg Income</div></div>
          <div class="g-strip-item"><div class="gsi-v">{as_:.0f}</div><div class="gsi-l">Avg Score</div></div>
          <div class="g-strip-item"><div class="gsi-v">{aa:.0f}</div><div class="gsi-l">Avg Age</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="gcard" style="padding-bottom:1.1rem;">
          <div class="ct-eyebrow">02 — Segment Reference</div>
          <div class="ct-title">Profile & Action</div>
          <table style="width:100%;border-collapse:collapse;">
            <thead><tr>
              <th style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:0 0 8px;text-align:left;border-bottom:1px solid var(--glass-brd);">Segment</th>
              <th style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:0 0 8px;text-align:left;border-bottom:1px solid var(--glass-brd);">N</th>
              <th style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;letter-spacing:.14em;padding:0 0 8px;text-align:left;border-bottom:1px solid var(--glass-brd);">Action</th>
            </tr></thead><tbody>
        """, unsafe_allow_html=True)
        for i, mi in enumerate(meta):
            cnt = int((df['Cluster']==i).sum())
            st.markdown(f"""<tr>
              <td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
                <div style="display:flex;align-items:center;gap:7px;">
                  <span style="width:8px;height:8px;border-radius:50%;background:{mi['color']};box-shadow:0 0 6px {mi['color']};flex-shrink:0;display:inline-block;"></span>
                  <span style="font-size:.76rem;font-weight:600;color:var(--text);">{mi['short']}</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:.56rem;color:var(--text3);padding-left:15px;">{mi['tag']}</div>
              </td>
              <td style="padding:8px 8px 8px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--text2);">{cnt}</td>
              <td style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:.68rem;color:var(--text3);font-weight:300;line-height:1.4;">{mi['strategy']}</td>
            </tr>""", unsafe_allow_html=True)
        st.markdown('</tbody></table></div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    r2a, r2b, r2c = st.columns(3, gap="medium")
    with r2a:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">03 — Income Distribution</div><div class="ct-title">Frequency by Cluster</div>', unsafe_allow_html=True)
        fig_h = go.Figure()
        for i, mi in enumerate(meta):
            fig_h.add_trace(go.Histogram(x=df[df['Cluster']==i]['Income'], nbinsx=14, name=mi['short'],
                marker=dict(color=mi['color'], opacity=0.7, line=dict(color='rgba(0,0,0,0.3)', width=0.5)),
                hovertemplate=f'{mi["name"]}: %{{x:.0f}}k · %{{y}}<extra></extra>'))
        fig_h.update_layout(**CC(), barmode='overlay', margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            legend=dict(font=dict(size=8.5, family='JetBrains Mono'), bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.34, x=0), bargap=0.05)
        st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r2b:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">04 — Age Profile</div><div class="ct-title">Distribution per Cluster</div>', unsafe_allow_html=True)
        fig_bx = go.Figure()
        for i, mi in enumerate(meta):
            fig_bx.add_trace(go.Box(y=df[df['Cluster']==i]['Age'], name=mi['short'],
                marker=dict(color=mi['color'], size=3), line=dict(color=mi['color'], width=1.5),
                fillcolor=mi['dim'], boxmean=True,
                hovertemplate=f'{mi["name"]}<br>Age: %{{y}}<extra></extra>'))
        fig_bx.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9,family='JetBrains Mono',color='#7C6FA0')),
            yaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            showlegend=False)
        st.plotly_chart(fig_bx, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c:
        st.markdown('<div class="gcard" style="padding-bottom:1.1rem;"><div class="ct-eyebrow">05 — Live Classifier</div><div class="ct-title">Quick Predict</div>', unsafe_allow_html=True)
        oi = st.slider("Income (k$)", 15, 137, 65, key="ov_i")
        os = st.slider("Spending Score", 1, 100, 50, key="ov_s")
        oc = classify(oi, os); om = meta[oc]
        st.markdown(f"""
        <div style="background:{om['dim']};border:1px solid {om['brd']};border-radius:12px;padding:.9rem 1rem;margin:.6rem 0 .5rem;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:{om['color']};text-transform:uppercase;letter-spacing:.14em;margin-bottom:3px;opacity:.8;">Predicted</div>
          <div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;color:{om['color']};letter-spacing:-.02em;">{om['name']}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:{om['color']};opacity:.6;margin-top:2px;">{om['tag']}</div>
        </div>
        <div style="background:var(--glass-bg);border:1px solid var(--glass-brd);border-radius:10px;padding:8px 10px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:2px;">Action</div>
          <div style="font-size:.75rem;font-weight:500;color:var(--text);">{om['strategy']}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    hm1, hm2 = st.columns([1.4,1], gap="medium")
    with hm1:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">06 — Decision Boundary Map</div><div class="ct-title">Full Input Space · Cluster Zones</div>', unsafe_allow_html=True)
        hg = np.arange(15,138,4); sg = np.arange(1,101,4)
        Z  = np.array([[classify(h,s) for h in hg] for s in sg])
        cs = [[0.,meta[0]['dim']],[.25,meta[1]['dim']],[.5,meta[2]['dim']],[.75,meta[3]['dim']],[1.,meta[4]['dim']]]
        fig_hm = go.Figure(go.Heatmap(x=hg, y=sg, z=Z, colorscale=cs, showscale=False,
            hovertemplate='Income: %{x}k · Score: %{y} → Cluster %{z}<extra></extra>'))
        fig_hm.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1], mode='markers+text',
            marker=dict(symbol='diamond', color='#A78BFA', size=12, line=dict(color='rgba(0,0,0,.5)', width=1.5)),
            text=[mi['short'] for mi in meta], textposition='top center',
            textfont=dict(size=8, family='JetBrains Mono', color='#A78BFA'), hoverinfo='skip', showlegend=False))
        fig_hm.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=260,
            xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor=GRID))
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with hm2:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">07 — Spending Score Profile</div><div class="ct-title">Mean ± 1σ per Cluster</div>', unsafe_allow_html=True)
        means = [df[df['Cluster']==i]['Spending'].mean() for i in range(5)]
        stds  = [df[df['Cluster']==i]['Spending'].std()  for i in range(5)]
        fig_sp = go.Figure(go.Bar(
            x=[mi['short'] for mi in meta], y=means,
            error_y=dict(type='data', array=stds, visible=True, color='#7C6FA0', thickness=1.2, width=4),
            marker=dict(color=[mi['color'] for mi in meta], opacity=0.8),
            text=[f'{m:.0f}' for m in means], textposition='outside',
            textfont=dict(size=9.5, family='JetBrains Mono', color='#7C6FA0'),
            hovertemplate='%{x}: %{y:.1f} ± %{error_y.array:.1f}<extra></extra>',
            showlegend=False, width=0.55))
        fig_sp.update_layout(**CC(), margin=dict(l=0,r=0,t=10,b=0), height=260,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9.5,family='JetBrains Mono',color='#7C6FA0')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK, title="Spending Score",
                       title_font=AX, showline=True, linecolor=GRID, range=[0,115]), bargap=0.35)
        st.plotly_chart(fig_sp, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    footer(); st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  02 — PROFILER
# ══════════════════════════════════════════════════════════════
elif page == "profiler":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown('<div class="g-label">Customer Input</div>', unsafe_allow_html=True)
    # st.markdown('<div class="gcard" style="padding-bottom:1.1rem;">', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns([2,2,1,1])
    with p1: income   = st.slider("Annual Income (k$)", 15, 137, 65, key="pf_i")
    with p2: spending = st.slider("Spending Score (1–100)", 1, 100, 50, key="pf_s")
    with p3: age      = st.number_input("Age", min_value=18, max_value=80, value=35, key="pf_a")
    with p4: gender   = st.selectbox("Gender", ["Male","Female"], key="pf_g")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

    cluster = classify(income, spending)
    m = meta[cluster]
    ca_i = df[df['Cluster']==cluster]['Income'].mean()
    ca_s = df[df['Cluster']==cluster]['Spending'].mean()
    ca_a = df[df['Cluster']==cluster]['Age'].mean()

    st.markdown('<div class="g-label">Classification Result</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 2.1, 1.3], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="result-g" style="border-color:{m['brd']};box-shadow:0 0 32px {m['dim']};">
          <div class="result-ghost">{cluster}</div>
          <div class="result-eyebrow">Cluster {cluster} / 5</div>
          <div class="result-name">{m['name']}</div>
          <div class="result-tag">{m['tag']}</div>
          <div class="seg-glass-pill" style="background:{m['dim']};color:{m['color']};border-color:{m['brd']};">
            <span style="width:6px;height:6px;border-radius:50%;background:{m['color']};box-shadow:0 0 6px {m['color']};display:inline-block;"></span>
            {m['short']}
          </div>
          <div class="r-line"></div>
        """, unsafe_allow_html=True)
        for i, mi in enumerate(meta):
            on = i == cluster
            st.markdown(f"""
            <div class="cl-row {"on" if on else ""}">
              <div class="cl-name {"on" if on else ""}">
                <span style="width:7px;height:7px;border-radius:50%;background:{mi['color']};box-shadow:0 0 5px {mi['color']};display:inline-block;"></span>
                {mi['short']}
              </div>
              <span class="cl-ct">{int((df['Cluster']==i).sum())}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Cluster Map · Your Position</div><div class="ct-title">Star Marker = Your Input</div>', unsafe_allow_html=True)
        st.plotly_chart(scatter(highlight=cluster, you=(income, spending, m['color']), h=270),
                        use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)

        st.markdown('<div class="gcard"><div class="ct-eyebrow">Profile Radar</div><div class="ct-title">Normalised Dimensions</div>', unsafe_allow_html=True)
        in_n = (income-15)/(137-15)*100
        af   = max(0, 100-abs(age-ca_a)*3)
        sf   = max(0, 100-abs(spending-ca_s)*2)
        cats = ['Income','Spending','Age Fit','Spend Fit','Engagement']
        vals = [in_n, spending, af, sf, (in_n+spending)/2]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself',
            fillcolor=m['dim'], line=dict(color=m['color'], width=2),
        ))
        fig_r.update_layout(**CC(), margin=dict(l=30,r=30,t=20,b=10), height=195,
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=8,family='JetBrains Mono'), gridcolor=GRID, linecolor=GRID),
                angularaxis=dict(tickfont=dict(size=9,family='JetBrains Mono',color='#7C6FA0')),
                bgcolor='rgba(0,0,0,0)'
            ), showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        tips = []
        if income < 35:
            tips.append(("bad","⚑","Low Income","Price sensitivity high — value messaging."))
        elif income < 70:
            tips.append(("warn","→","Mid Income","Balance quality with affordability."))
        else:
            tips.append(("ok","✓","High Income","Receptive to premium products."))

        if spending < 30:
            tips.append(("bad","⚑","Low Spending","Disengaged — re-engagement needed."))
        elif spending < 65:
            tips.append(("warn","→","Moderate Spend","Nudge campaigns have potential."))
        else:
            tips.append(("ok","✓","High Spending","Active buyer — upsell focus."))

        if age < 30:
            tips.append(("info","◉","Young Demo","Trend-driven offers work best."))
        elif age > 55:
            tips.append(("info","◉","Mature Demo","Trust & loyalty resonate."))

        id_ = income - ca_i
        tips.append(("ok" if abs(id_)<10 else "warn","◈","Cluster Fit",
            f"Income {abs(id_):.0f}k {'↑' if id_>0 else '↓'} avg ({ca_i:.0f}k)"))
        sd_ = spending - ca_s
        tips.append(("ok" if abs(sd_)<10 else "warn","◈","Spend Fit",
            f"Score {abs(sd_):.0f} pts {'↑' if sd_>0 else '↓'} avg ({ca_s:.0f})"))

        for sev, ico, title, body in tips[:5]:
            st.markdown(f"""
            <div class="ins-g {sev}">
              <span class="ins-ico">{ico}</span>
              <div><div class="ins-t">{title}</div><div class="ins-b">{body}</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="strat-g">
          <div class="strat-eyebrow">Recommended Action</div>
          <div class="strat-text">{m['strategy']}</div>
        </div>""", unsafe_allow_html=True)

        gcnt = int(df[(df['Cluster']==cluster)&(df['Gender']==gender)].shape[0])
        st.markdown(f"""
        <div style="margin-top:8px;background:var(--glass-bg);border:1px solid var(--glass-brd);border-radius:12px;padding:10px 12px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px;">Cluster Stats</div>
          <div style="display:flex;gap:14px;">
            <div><div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text);">{int(ca_a)}</div><div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;">Avg Age</div></div>
            <div style="width:1px;background:var(--glass-brd);"></div>
            <div><div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text);">{gcnt}</div><div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;">{gender}</div></div>
            <div style="width:1px;background:var(--glass-brd);"></div>
            <div><div style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text);">{int((df['Cluster']==cluster).sum())}</div><div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:var(--text3);text-transform:uppercase;">Total</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    footer(); st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  03 — SEGMENTS
# ══════════════════════════════════════════════════════════════
elif page == "segments":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    sel = st.selectbox("Select Segment", list(range(5)),
        format_func=lambda i: f"Cluster {i}  ·  {meta[i]['name']}  —  {meta[i]['tag']}",
        key="seg_sel")
    m   = meta[sel]
    sdf = df[df['Cluster']==sel]

    st.markdown(f"""
    <div style="background:{m['dim']};border:1px solid {m['brd']};border-radius:16px;
                padding:1.2rem 1.6rem;margin:.8rem 0 1.4rem;
                display:flex;align-items:center;justify-content:space-between;gap:2rem;
                box-shadow:0 0 32px {m['dim']};">
      <div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.56rem;color:{m['color']};text-transform:uppercase;letter-spacing:.16em;margin-bottom:3px;opacity:.8;">Cluster {sel}</div>
        <div style="font-family:'Sora',sans-serif;font-size:2rem;font-weight:700;color:{m['color']};letter-spacing:-.03em;">{m['name']}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.6rem;color:{m['color']};opacity:.6;margin-top:2px;">{m['tag']}</div>
      </div>
      <div style="font-size:.8rem;font-weight:300;color:var(--text2);max-width:260px;text-align:right;line-height:1.65;">
        Strategy: <b style="font-weight:600;color:var(--text);">{m['strategy']}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    sc4 = st.columns(4, gap="small")
    for col, (val, lbl, sub) in zip(sc4, [
        (f"{sdf['Income'].mean():.1f}k",   "Avg Income",   f"σ = {sdf['Income'].std():.1f}k"),
        (f"{sdf['Spending'].mean():.1f}",  "Avg Spending", f"σ = {sdf['Spending'].std():.1f}"),
        (f"{sdf['Age'].mean():.1f}",        "Avg Age",      f"{sdf['Age'].min()}–{sdf['Age'].max()} yrs"),
        (f"{len(sdf)}",                     "Cluster Size", f"{len(sdf)/2:.0f}% · {int((sdf['Gender']=='Female').sum())}F {int((sdf['Gender']=='Male').sum())}M"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-g">
              <div class="kpi-accent" style="background:linear-gradient(180deg,{m['color']},transparent);"></div>
              <div class="kpi-val" style="padding-left:8px;color:{m['color']};font-size:2.2rem;">{val}</div>
              <div class="kpi-name" style="padding-left:8px;">{lbl}</div>
              <div class="kpi-meta" style="padding-left:8px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Income Distribution</div><div class="ct-title">Cluster Context Overlay</div>', unsafe_allow_html=True)
        fig_di = go.Figure()
        for i, mi in enumerate(meta):
            fig_di.add_trace(go.Histogram(x=df[df['Cluster']==i]['Income'], nbinsx=18, name=mi['short'],
                marker=dict(color=mi['color'], opacity=0.8 if i==sel else 0.12, line=dict(color='rgba(0,0,0,.3)', width=.5)),
                hovertemplate=f'{mi["name"]}: %{{x:.0f}}k · %{{y}}<extra></extra>'))
        fig_di.update_layout(**CC(), barmode='overlay', margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            showlegend=False, bargap=0.05)
        st.plotly_chart(fig_di, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Spending Distribution</div><div class="ct-title">Score Frequency</div>', unsafe_allow_html=True)
        fig_ds = go.Figure()
        for i, mi in enumerate(meta):
            fig_ds.add_trace(go.Histogram(x=df[df['Cluster']==i]['Spending'], nbinsx=18, name=mi['short'],
                marker=dict(color=mi['color'], opacity=0.8 if i==sel else 0.12, line=dict(color='rgba(0,0,0,.3)', width=.5)),
                hovertemplate=f'{mi["name"]}: Score %{{x:.0f}} · %{{y}}<extra></extra>'))
        fig_ds.update_layout(**CC(), barmode='overlay', margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            showlegend=False, bargap=0.05)
        st.plotly_chart(fig_ds, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    d3, d4 = st.columns([1.5,1], gap="medium")
    with d3:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Cluster Map · Highlighted</div><div class="ct-title">Selected Segment in Focus</div>', unsafe_allow_html=True)
        st.plotly_chart(scatter(highlight=sel, h=260), use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)
    with d4:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Age × Spending</div><div class="ct-title">Cluster Context</div>', unsafe_allow_html=True)
        fig_as = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster']==i
            fig_as.add_trace(go.Scatter(x=df.loc[mask,'Age'], y=df.loc[mask,'Spending'], mode='markers',
                marker=dict(color=mi['color'], size=6 if i==sel else 4,
                            opacity=0.75 if i==sel else 0.1,
                            line=dict(color='rgba(0,0,0,.4)' if i==sel else 'rgba(0,0,0,0)', width=.8)),
                name=mi['short'], hovertemplate=f'{mi["name"]}<br>Age: %{{x}}<br>Spending: %{{y}}<extra></extra>'))
        fig_as.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=260,
            xaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            showlegend=False)
        st.plotly_chart(fig_as, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">All 5 Segments</div>', unsafe_allow_html=True)
    g5 = st.columns(5, gap="small")
    for col, mi, i in zip(g5, meta, range(5)):
        cnt = int((df['Cluster']==i).sum())
        with col:
            st.markdown(f"""
            <div class="seg-g {"selected" if i==sel else ""}">
              <div class="seg-g-num" style="color:{mi['color']};">{i}</div>
              <div class="seg-g-id">Cluster {i}</div>
              <div class="seg-g-name" style="color:{mi['color']};">{mi['name']}</div>
              <div class="seg-g-tag">{mi['tag']}</div>
              <div style="height:3px;background:linear-gradient(90deg,{mi['color']},transparent);border-radius:2px;margin-bottom:10px;opacity:.5;"></div>
              <div class="seg-g-ct">{cnt} customers</div>
              <div style="font-size:.67rem;color:var(--text3);margin-top:5px;line-height:1.4;">{mi['strategy']}</div>
            </div>""", unsafe_allow_html=True)

    footer(); st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  04 — SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "simulator":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    st.markdown('<div class="g-label">Baseline Profile</div>', unsafe_allow_html=True)
    # st.markdown('<div class="gcard" style="padding-bottom:1.1rem;">', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1: base_i = st.slider("Annual Income (k$)", 15, 137, 55, key="sim_i")
    with b2: base_s = st.slider("Spending Score (1–100)", 1, 100, 50, key="sim_s")
    st.markdown('</div>', unsafe_allow_html=True)

    bc = classify(base_i, base_s); bm = meta[bc]
    st.markdown(f"""
    <div style="background:{bm['dim']};border:1px solid {bm['brd']};border-radius:14px;
                padding:1rem 1.5rem;margin:.8rem 0 1.4rem;
                display:flex;align-items:center;gap:2rem;
                box-shadow:0 0 24px {bm['dim']};">
      <div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.55rem;color:{bm['color']};text-transform:uppercase;letter-spacing:.14em;margin-bottom:2px;opacity:.8;">Baseline</div>
        <div style="font-family:'Sora',sans-serif;font-size:1.8rem;font-weight:700;color:{bm['color']};letter-spacing:-.03em;">{bm['name']}</div>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:{bm['color']};opacity:.7;line-height:1.9;">Cluster {bc}<br>{bm['tag']}</div>
      <div style="margin-left:auto;font-size:.78rem;font-weight:300;color:var(--text2);">{bm['strategy']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="g-label">7 Scenarios</div>', unsafe_allow_html=True)
    scenarios = [
        ("+10k Inc",  min(base_i+10,137), base_s),
        ("+20k Inc",  min(base_i+20,137), base_s),
        ("−10k Inc",  max(base_i-10, 15), base_s),
        ("+20 Score", base_i, min(base_s+20,100)),
        ("+40 Score", base_i, min(base_s+40,100)),
        ("−20 Score", base_i, max(base_s-20, 1)),
        ("Premium",   min(base_i+25,137), min(base_s+25,100)),
    ]
    sc7 = st.columns(7, gap="small")
    for col, (lbl, is_, ss_) in zip(sc7, scenarios):
        c2 = classify(is_, ss_); mi2 = meta[c2]; chg = c2 != bc
        with col:
            st.markdown(f"""
            <div class="sim-g {"shifted" if chg else ""}">
              <div class="sim-lbl">{lbl}</div>
              <div class="sim-name" style="color:{mi2['color']};">{mi2['name']}</div>
              <div class="sim-dl {"changed" if chg else "same"}">{"↳ SHIFTED" if chg else "· SAME"}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    sw1, sw2 = st.columns(2, gap="medium")
    with sw1:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Income Sweep</div><div class="ct-title">Cluster vs Income · Spending Fixed</div>', unsafe_allow_html=True)
        ir = np.arange(15,138,2); sw = [classify(i, base_s) for i in ir]
        fig_sw1 = go.Figure()
        for i in range(5):
            mask = np.array(sw)==i
            if mask.any():
                fig_sw1.add_trace(go.Scatter(x=ir[mask], y=np.ones(mask.sum())*i, mode='markers',
                    marker=dict(color=meta[i]['color'], size=10, opacity=0.8, symbol='square',
                                line=dict(color='rgba(0,0,0,.4)', width=1)),
                    name=meta[i]['short'], hovertemplate=f'%{{x}}k → {meta[i]["name"]}<extra></extra>'))
        fig_sw1.add_vline(x=base_i, line=dict(color='#A78BFA', width=1.5, dash='dot'),
            annotation_text="  baseline", annotation_font=dict(size=9, family='JetBrains Mono', color='#A78BFA'))
        fig_sw1.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=210,
            xaxis=dict(title="Annual Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Cluster ID", gridcolor=GRID, zeroline=False, dtick=1, tickfont=TICK, title_font=AX, range=[-0.5,4.5]),
            legend=dict(font=dict(size=9, family='JetBrains Mono'), bgcolor='rgba(13,11,26,.8)', bordercolor=GRID, borderwidth=1, orientation='h', y=-0.34))
        st.plotly_chart(fig_sw1, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with sw2:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Spending Sweep</div><div class="ct-title">Cluster vs Spending · Income Fixed</div>', unsafe_allow_html=True)
        sr = np.arange(1,101,2); ss = [classify(base_i, s) for s in sr]
        fig_sw2 = go.Figure()
        for i in range(5):
            mask = np.array(ss)==i
            if mask.any():
                fig_sw2.add_trace(go.Scatter(x=sr[mask], y=np.ones(mask.sum())*i, mode='markers',
                    marker=dict(color=meta[i]['color'], size=10, opacity=0.8, symbol='square',
                                line=dict(color='rgba(0,0,0,.4)', width=1)),
                    name=meta[i]['short'], hovertemplate=f'Score %{{x}} → {meta[i]["name"]}<extra></extra>'))
        fig_sw2.add_vline(x=base_s, line=dict(color='#A78BFA', width=1.5, dash='dot'),
            annotation_text="  baseline", annotation_font=dict(size=9, family='JetBrains Mono', color='#A78BFA'))
        fig_sw2.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=210,
            xaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Cluster ID", gridcolor=GRID, zeroline=False, dtick=1, tickfont=TICK, title_font=AX, range=[-0.5,4.5]),
            legend=dict(font=dict(size=9, family='JetBrains Mono'), bgcolor='rgba(13,11,26,.8)', bordercolor=GRID, borderwidth=1, orientation='h', y=-0.34))
        st.plotly_chart(fig_sw2, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="gcard"><div class="ct-eyebrow">Proximity Map · Baseline & Scenarios vs Boundaries</div><div class="ct-title">Cross-hair = Baseline · Circles = Scenarios</div>', unsafe_allow_html=True)
    hg2 = np.arange(15,138,4); sg2 = np.arange(1,101,4)
    Z2  = np.array([[classify(h,s) for h in hg2] for s in sg2])
    cs2 = [[0.,meta[0]['dim']],[.25,meta[1]['dim']],[.5,meta[2]['dim']],[.75,meta[3]['dim']],[1.,meta[4]['dim']]]
    fig_px = go.Figure(go.Heatmap(x=hg2, y=sg2, z=Z2, colorscale=cs2, showscale=False,
        hovertemplate='Income: %{x}k · Score: %{y} → Cluster %{z}<extra></extra>'))
    fig_px.add_trace(go.Scatter(x=[base_i], y=[base_s], mode='markers',
        marker=dict(symbol='cross-thin', color='#A78BFA', size=20, line=dict(color='#A78BFA', width=3)),
        name='Baseline', hovertemplate=f'Baseline · {base_i}k · {base_s}<extra></extra>'))
    for lbl2, is2, ss2 in scenarios:
        c3 = classify(is2, ss2)
        fig_px.add_trace(go.Scatter(x=[is2], y=[ss2], mode='markers',
            marker=dict(color=meta[c3]['color'], size=9, opacity=0.9, line=dict(color='rgba(0,0,0,.5)', width=1.5)),
            name=lbl2, hovertemplate=f'{lbl2}<br>{is2}k · {ss2} → {meta[c3]["name"]}<extra></extra>'))
    fig_px.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=270,
        xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
        yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)', tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
        legend=dict(font=dict(size=9, family='JetBrains Mono'), bgcolor='rgba(13,11,26,.8)', bordercolor=GRID, borderwidth=1, orientation='h', y=-0.14))
    st.plotly_chart(fig_px, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    footer(); st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  05 — DATA
# ══════════════════════════════════════════════════════════════
elif page == "data":
    st.markdown('<div class="shell">', unsafe_allow_html=True)

    kc = st.columns(4, gap="small")
    for col, (val, lbl, sub) in zip(kc, [
        ("200","Total Records","Synthetic · seed=42"),
        ("5","Segments","K-Means · 5 clusters"),
        ("2","Features","Income + Spending"),
        (f"{int((df['Gender']=='Female').sum())}/{int((df['Gender']=='Male').sum())}","F / M Split","Random assign"),
    ]):
        with col:
            st.markdown(f"""
            <div class="kpi-g">
              <div class="kpi-accent" style="background:linear-gradient(180deg,var(--violet),transparent);"></div>
              <div class="kpi-val" style="padding-left:8px;font-size:2.2rem;">{val}</div>
              <div class="kpi-name" style="padding-left:8px;">{lbl}</div>
              <div class="kpi-meta" style="padding-left:8px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Filters</div>', unsafe_allow_html=True)
    # st.markdown('<div class="gcard" style="padding-bottom:1.1rem;">', unsafe_allow_html=True)
    fa, fb, fc, fd = st.columns([2,1.5,1.5,1])
    with fa:
        seg_f = st.multiselect("Segment", list(range(5)),
            format_func=lambda i: f"Cluster {i} — {meta[i]['name']}",
            default=list(range(5)), key="dt_s")
    with fb:
        gen_f = st.multiselect("Gender", ["Male","Female"], default=["Male","Female"], key="dt_g")
    with fc:
        age_r = st.slider("Age Range", 18, 80, (18,80), key="dt_a")
    with fd:
        sort_b = st.selectbox("Sort By", ["Income ↓","Income ↑","Spending ↓","Spending ↑","Age ↓","Age ↑"], key="dt_sort")
    st.markdown('</div>', unsafe_allow_html=True)

    fdf = df[df['Cluster'].isin(seg_f) & df['Gender'].isin(gen_f) & df['Age'].between(age_r[0],age_r[1])].copy()
    fdf['Segment'] = fdf['Cluster'].apply(lambda x: meta[x]['name'])
    sk, sa = {"Income ↓":("Income",False),"Income ↑":("Income",True),"Spending ↓":("Spending",False),"Spending ↑":("Spending",True),"Age ↓":("Age",False),"Age ↑":("Age",True)}[sort_b]
    fdf = fdf.sort_values(sk, ascending=sa).reset_index(drop=True)

    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.6rem;color:var(--text3);margin:.5rem 0 .5rem;">Showing {min(100,len(fdf))} of {len(fdf)} filtered records</div>', unsafe_allow_html=True)

    rows = ""
    for _, row in fdf.head(100).iterrows():
        mi = meta[int(row['Cluster'])]
        rows += f"""<tr>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--text4);">{int(row.name)+1}</td>
          <td style="font-family:'JetBrains Mono',monospace;">{row['Income']:.1f}k</td>
          <td style="font-family:'JetBrains Mono',monospace;">{row['Spending']:.1f}</td>
          <td style="font-family:'JetBrains Mono',monospace;">{row['Age']}</td>
          <td>{row['Gender']}</td>
          <td><span class="g-chip" style="background:{mi['dim']};color:{mi['color']};border-color:{mi['brd']};"><span style="width:5px;height:5px;border-radius:50%;background:{mi['color']};box-shadow:0 0 5px {mi['color']};display:inline-block;"></span>{mi['short']}</span></td>
          <td style="font-size:.7rem;color:var(--text3);">{mi['strategy']}</td>
        </tr>"""

    st.markdown(f'<div class="dt-glass"><table><thead><tr><th>#</th><th>Income</th><th>Spending</th><th>Age</th><th>Gender</th><th>Segment</th><th>Action</th></tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)

    if len(fdf)>100:
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.58rem;color:var(--text3);margin-top:.4rem;text-align:center;">Showing first 100 of {len(fdf)} — download CSV for full set</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.75rem"></div>', unsafe_allow_html=True)
    csv = fdf[['Income','Spending','Age','Gender','Segment','Cluster']].to_csv(index=False)
    st.download_button("⬇  Download Filtered CSV", data=csv, file_name="segmentiq_export.csv", mime="text/csv")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="g-label">Filtered Overview</div>', unsafe_allow_html=True)
    dv1, dv2, dv3 = st.columns(3, gap="medium")
    with dv1:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Segment Counts</div><div class="ct-title">Filtered Records</div>', unsafe_allow_html=True)
        cf = [int((fdf['Cluster']==i).sum()) for i in range(5)]
        fig_cf = go.Figure(go.Bar(x=[mi['short'] for mi in meta], y=cf,
            marker=dict(color=[mi['color'] for mi in meta], opacity=0.82),
            text=cf, textposition='outside', textfont=dict(size=10, family='JetBrains Mono', color='#7C6FA0'),
            hovertemplate='%{x}: %{y}<extra></extra>', showlegend=False, width=0.6))
        fig_cf.update_layout(**CC(), margin=dict(l=0,r=0,t=10,b=0), height=200,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=9.5, family='JetBrains Mono', color='#7C6FA0')),
            yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=TICK, showline=True, linecolor=GRID), bargap=0.3)
        st.plotly_chart(fig_cf, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)
    with dv2:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Income vs Spending</div><div class="ct-title">Filtered Scatter</div>', unsafe_allow_html=True)
        fig_fsc = go.Figure()
        for i, mi in enumerate(meta):
            mask = fdf['Cluster']==i
            if mask.any():
                fig_fsc.add_trace(go.Scatter(x=fdf.loc[mask,'Income'], y=fdf.loc[mask,'Spending'], mode='markers',
                    marker=dict(color=mi['color'], size=6, opacity=0.7, line=dict(color='rgba(0,0,0,.4)', width=.8)),
                    name=mi['short'], hovertemplate=f'{mi["name"]}<br>%{{x:.0f}}k · %{{y}}<extra></extra>'))
        fig_fsc.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Income (k$)", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Spending Score", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            legend=dict(font=dict(size=9, family='JetBrains Mono'), bgcolor='rgba(0,0,0,0)', borderwidth=0, orientation='h', y=-0.34))
        st.plotly_chart(fig_fsc, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)
    with dv3:
        st.markdown('<div class="gcard"><div class="ct-eyebrow">Age Distribution</div><div class="ct-title">Filtered Records</div>', unsafe_allow_html=True)
        fig_fa = go.Figure(go.Histogram(x=fdf['Age'], nbinsx=20,
            marker=dict(color='#8B5CF6', opacity=0.6, line=dict(color='rgba(0,0,0,.3)', width=.5)),
            hovertemplate='Age %{x}: %{y}<extra></extra>'))
        fig_fa.update_layout(**CC(), margin=dict(l=0,r=0,t=0,b=0), height=200,
            xaxis=dict(title="Age", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            yaxis=dict(title="Count", gridcolor=GRID, zeroline=False, tickfont=TICK, title_font=AX, showline=True, linecolor=GRID),
            showlegend=False, bargap=0.08)
        st.plotly_chart(fig_fa, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    footer(); st.markdown('</div>', unsafe_allow_html=True)
