"""
SegmentIQ — Customer Segmentation Intelligence
Refined · Hierarchical · Professional
Run: streamlit run customer_seg_pro.py
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
    page_icon="◈",
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

# ── STYLES ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:         #F8F7F4;
    --surface:    #FFFFFF;
    --border:     #E5E2DC;
    --border2:    #EEEBE5;
    --text:       #18160F;
    --text2:      #5C5852;
    --text3:      #9C9890;
    --accent:     #1B4FD8;
    --accent-bg:  #EEF3FD;
    --accent-brd: #BFCFFE;
    --gold:       #C47C0A;
    --red:        #DC2626;
    --green:      #059669;
}

html, body, [class*="css"], .stApp {
    font-family: 'Geist', system-ui, sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.015em;
}

[data-testid="collapsedControl"],
section[data-testid="stSidebar"],
#MainMenu, footer, header { display: none !important; visibility: hidden !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAV ── */
.nav {
    background: rgba(248,247,244,0.94);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 1.5rem 2.5rem;
    position: sticky; top: 0; z-index: 999;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-wordmark {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem; color: var(--text);
    letter-spacing: -0.02em; line-height: 1;
}
.nav-wordmark em { color: var(--accent); font-style: italic; }
.nav-badge {
    font-size: 0.73rem; font-weight: 600;
    color: var(--accent); background: var(--accent-bg);
    border: 1px solid var(--accent-brd);
    padding: 2px 9px; border-radius: 20px;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.nav-links { display: flex; gap: 10px; align-items: center; }
.nav-link {
    font-size: 0.9rem; font-weight: 500;
    color: var(--text3); padding: 7px 16px;
    border-radius: 5px; transition: all .15s;
    cursor: pointer; text-decoration: none; user-select: none;
}
.nav-link:hover { color: var(--text); background: var(--border2); }
.nav-link.active { background: var(--text); color: var(--bg); font-weight: 700; }

/* ── SHELL ── */
.shell {
    max-width: 1160px; margin: 0 auto;
    padding: 0.5rem 1.5rem 3rem;
    position: relative; z-index: 1;
}

/* ── PAGE HEADER ── */
.page-eyebrow {
    font-size: 0.67rem; font-weight: 700;
    color: var(--accent); letter-spacing: 0.13em;
    text-transform: uppercase; margin-bottom: 0.65rem;
    display: flex; align-items: center; gap: 8px;
}
.page-eyebrow::before {
    content: ''; width: 18px; height: 2px;
    background: var(--accent); border-radius: 2px;
}
.page-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.7rem; color: var(--text);
    letter-spacing: -0.03em; line-height: 1.08;
    margin-bottom: 0.7rem;
}
.page-title em { color: var(--accent); }
.page-desc {
    font-size: 0.9rem; color: var(--text2);
    max-width: 460px; line-height: 1.7; font-weight: 400;
}

/* ── SECTION LABELS ── */
.sec-label {
    font-size: 0.67rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.13em;
    color: var(--text3); margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border2); }

/* ── CARD ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

/* ── RESULT BLOCK ── */
.result-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 2.25rem 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    position: relative; overflow: hidden;
}
.result-block::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), transparent 100%);
}
.result-block .rb-glow {
    position: absolute; bottom: -40px; right: -40px;
    width: 160px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle, rgba(27,79,216,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.cluster-num {
    font-family: 'Geist Mono', monospace;
    font-size: 6.5rem; font-weight: 600; line-height: 1;
    letter-spacing: -0.07em; color: var(--text);
}
.cluster-num .unit { font-size: 1.8rem; color: var(--text3); font-weight: 400; vertical-align: top; padding-top: 1rem; display: inline-block; }
.cluster-sublabel {
    font-size: 0.7rem; font-weight: 700; color: var(--text3);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px;
}
.segment-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 16px; border-radius: 24px;
    font-size: 0.78rem; font-weight: 700;
    margin-top: 16px; border: 1.5px solid transparent;
}
.seg-divider { height: 1px; background: var(--border2); margin: 1.25rem 0 0.9rem; }
.seg-mini-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 5px 10px; border-radius: 8px; margin-bottom: 3px;
}
.seg-mini-row.active { background: var(--accent-bg); }
.seg-mini-name { font-size: 0.78rem; font-weight: 600; color: var(--text2); }
.seg-mini-count { font-family: 'Geist Mono', monospace; font-size: 0.69rem; color: var(--text3); font-weight: 500; }

/* ── INSIGHT ── */
.insight {
    display: flex; gap: 11px; padding: 11px 13px;
    border-radius: 11px; margin-bottom: 7px; border: 1px solid transparent;
}
.insight.ok   { background: #EDF8F2; border-color: #B8D9CB; }
.insight.warn { background: #FDF8EC; border-color: #F0D898; }
.insight.bad  { background: #FEF2F2; border-color: #FDC5C5; }
.insight.info { background: #EFF5FF; border-color: #BFCFFE; }
.insight-ico  { font-size: 13px; margin-top: 1px; flex-shrink: 0; }
.insight-title { font-size: 0.78rem; font-weight: 700; color: var(--text); margin-bottom: 2px; }
.insight-body  { font-size: 0.71rem; color: var(--text2); line-height: 1.55; }

/* ── POTENTIAL BOX ── */
.potential-box {
    background: linear-gradient(140deg, #0F2D6E 0%, #1B4FD8 100%);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    margin-top: 1rem; position: relative; overflow: hidden;
}
.potential-box::before {
    content: ''; position: absolute; top: -30px; right: -30px;
    width: 100px; height: 100px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.pot-eyebrow { font-size: 0.62rem; font-weight: 700; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px; }
.pot-val { font-family: 'Geist Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #fff; letter-spacing: -0.03em; line-height: 1.2; }
.pot-sub { font-size: 0.69rem; color: rgba(255,255,255,0.38); margin-top: 6px; }

/* ── SIM CARDS ── */
.sim-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 13px; padding: 1.2rem 0.9rem; text-align: center;
    transition: border-color .2s, box-shadow .2s;
}
.sim-card:hover { border-color: var(--accent-brd); box-shadow: 0 4px 14px rgba(27,79,216,0.09); }
.sim-lbl { font-size: 0.62rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 9px; }
.sim-seg { font-family: 'Instrument Serif', serif; font-size: 1.3rem; font-weight: 400; color: var(--text); }
.sim-delta { font-family: 'Geist Mono', monospace; font-size: 0.74rem; font-weight: 600; margin-top: 5px; }
.pos { color: #059669; } .neg { color: #DC2626; } .neu { color: var(--text3); }

/* ── BASELINE BANNER ── */
.baseline-banner {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 14px 14px 0;
    padding: 1.1rem 1.75rem; margin: 1.5rem 0 2.25rem;
    display: flex; align-items: center; gap: 2rem;
}
.baseline-seg {
    font-family: 'Instrument Serif', serif;
    font-size: 1.8rem; font-weight: 400;
    color: var(--accent); letter-spacing: -0.02em;
}

/* ── SLIDERS & INPUTS ── */
.stSlider label { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--text) !important; }
div[data-testid="stSlider"] > div > div > div { background: var(--border) !important; height: 4px !important; }
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(27,79,216,0.15) !important;
}
.stSelectbox label { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--text) !important; }
.stSelectbox > div > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 9px !important; font-size: 0.84rem !important;
    font-weight: 500 !important; font-family: 'Geist', sans-serif !important;
}
.stNumberInput label { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--text) !important; }
.stRadio label { font-size: 0.76rem !important; font-weight: 600 !important; color: var(--text) !important; }

/* ── ABOUT STAT ── */
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.75rem; }
.stat-val { font-family: 'Instrument Serif', serif; font-size: 2.5rem; color: var(--text); letter-spacing: -0.03em; margin-bottom: 5px; line-height: 1; }
.stat-lbl { font-size: 0.7rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; }
.stat-sub { font-size: 0.69rem; color: var(--text3); margin-top: 4px; }

/* ── FOOTER ── */
.app-footer {
    text-align: center; font-size: 0.66rem; color: var(--text3);
    padding: 2.5rem 0 1rem; margin-top: 4rem;
    border-top: 1px solid var(--border2);
    letter-spacing: 0.06em; font-weight: 500;
}
.footer-sep { color: var(--border); margin: 0 8px; }
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
    age_raw = np.random.randint(18, 70, n)
    gender_raw = np.random.choice(["Male", "Female"], n)
    income_c  = np.clip(income_raw, 15, 137)
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

    labels = km.predict(sc.transform(X))
    df = pd.DataFrame({
        'Income': income_c, 'Spending': spending_c,
        'Age': age_raw, 'Gender': gender_raw, 'Cluster': labels
    })
    centers = sc.inverse_transform(km.cluster_centers_)

    meta = [
        {'name': 'Budget Enthusiasts', 'tag': 'Low Income · Low Spend',   'color': '#DC2626', 'light': '#FEF2F2', 'brd': '#FDC5C5', 'strategy': 'Flash sales, discount codes, price alerts'},
        {'name': 'Impulsive Spenders',  'tag': 'Low Income · High Spend',  'color': '#C47C0A', 'light': '#FDF8EC', 'brd': '#F0D898', 'strategy': 'Loyalty rewards, BNPL options, curated picks'},
        {'name': 'Standard Customers',  'tag': 'Mid Income · Mid Spend',   'color': '#059669', 'light': '#EDF8F2', 'brd': '#B8D9CB', 'strategy': 'Seasonal promotions, newsletter campaigns'},
        {'name': 'Target Customers',    'tag': 'High Income · High Spend', 'color': '#1B4FD8', 'light': '#EEF3FD', 'brd': '#BFCFFE', 'strategy': 'Premium bundles, VIP early access programs'},
        {'name': 'Cautious Savers',     'tag': 'High Income · Low Spend',  'color': '#7C3AED', 'light': '#F5F3FF', 'brd': '#DDD6FE', 'strategy': 'Value messaging, exclusive high-ROI offers'},
    ]
    return km, sc, df, centers, meta

km, sc, df, centers, meta = load_model_and_data()

def classify(income, spending):
    f = np.array([[income, spending]])
    return int(km.predict(sc.transform(f))[0])

CHART_DEFAULTS = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Geist', color='#9C9890', size=11),
)
SEG_COLORS = [m['color'] for m in meta]

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
    <span class="nav-badge">K-Means · 5 Clusters</span>
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
#  PAGE: SEGMENT
# ══════════════════════════════════════════════════════════════
if page == "segment":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Customer Classification</div>
      <div class="page-title">Customer Segmentation<br><em>Intelligence</em></div>
      <div class="page-desc">Enter a customer profile below — the segment prediction updates live as you adjust the inputs.</div>
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
    st.markdown('<div class="sec-label">Segment Result</div>', unsafe_allow_html=True)

    cluster = classify(income, spending)
    m = meta[cluster]

    col_result, col_charts, col_insights = st.columns([1.1, 2.3, 1.4], gap="medium")

    with col_result:
        st.markdown(f"""
        <div class="result-block">
          <div class="rb-glow"></div>
          <div class="cluster-num">{cluster}<span class="unit"> /4</span></div>
          <div class="cluster-sublabel">Cluster ID · out of 5</div>
          <span class="segment-badge" style="background:{m['light']};color:{m['color']};border-color:{m['brd']};">
            ● &nbsp;{m['name']}
          </span>
          <div class="seg-divider"></div>
        """, unsafe_allow_html=True)
        for i, mi in enumerate(meta):
            count = int((df['Cluster'] == i).sum())
            active = "active" if i == cluster else ""
            dot_color = mi['color'] if i == cluster else "var(--border)"
            st.markdown(f"""
            <div class="seg-mini-row {active}">
              <span class="seg-mini-name" style="color:{'var(--text)' if i==cluster else 'var(--text3)'};">
                <span style="color:{mi['color']};margin-right:6px;">●</span>{mi['name'].split()[0]}
              </span>
              <span class="seg-mini-count">{count}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_charts:
        # ── Scatter plot
        fig1 = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster'] == i
            fig1.add_trace(go.Scatter(
                x=df.loc[mask, 'Income'], y=df.loc[mask, 'Spending'],
                mode='markers',
                marker=dict(color=mi['color'], size=7, opacity=0.72,
                            line=dict(color='white', width=0.5)),
                name=mi['name'],
                hovertemplate=f'Income: %{{x:.0f}}k · Spending: %{{y:.0f}}<extra>{mi["name"]}</extra>',
            ))
        # centroids
        fig1.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1], mode='markers',
            marker=dict(symbol='diamond', color='#18160F', size=11,
                        line=dict(color='white', width=1.5)),
            name='Centroids', hoverinfo='skip'
        ))
        # user point
        fig1.add_trace(go.Scatter(
            x=[income], y=[spending], mode='markers',
            marker=dict(symbol='star', color=m['color'], size=18,
                        line=dict(color='white', width=2)),
            name=f'You ({income}k, {spending})',
            hovertemplate=f'You · Income: {income}k · Spending: {spending}<extra></extra>'
        ))
        fig1.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0, r=16, t=10, b=6), height=240,
            xaxis=dict(title="Annual Income (k$)", gridcolor='#EEEBE5', zeroline=False,
                       tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
            yaxis=dict(title="Spending Score", gridcolor='#EEEBE5', zeroline=False,
                       tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
            legend=dict(font=dict(size=9.5), bgcolor='rgba(0,0,0,0)', borderwidth=0),
            showlegend=True,
        )
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

        # ── Cluster averages bar
        avg_inc = [df[df['Cluster']==i]['Income'].mean() for i in range(5)]
        avg_sp  = [df[df['Cluster']==i]['Spending'].mean() for i in range(5)]
        seg_names = [mi['name'].split()[0] for mi in meta]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=seg_names, y=avg_inc, name='Avg Income (k$)',
            marker=dict(color='#1B4FD8', opacity=0.85, cornerradius=4, line=dict(width=0)),
            width=0.35, offsetgroup=0,
            hovertemplate='%{x} · Income: %{y:.0f}k<extra></extra>',
        ))
        fig2.add_trace(go.Bar(
            x=seg_names, y=avg_sp, name='Avg Spending Score',
            marker=dict(color='#C47C0A', opacity=0.85, cornerradius=4, line=dict(width=0)),
            width=0.35, offsetgroup=1,
            hovertemplate='%{x} · Spending: %{y:.0f}<extra></extra>',
        ))
        fig2.update_layout(
            **CHART_DEFAULTS,
            barmode='group', margin=dict(l=0, r=16, t=32, b=4), height=195,
            title=dict(text="Avg Income vs Spending by Segment",
                       font=dict(size=11.5, color='#5C5852', family='Geist'), x=0),
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10, color='#18160F')),
            yaxis=dict(gridcolor='#EEEBE5', zeroline=False, tickfont=dict(size=9.5, family='Geist Mono')),
            legend=dict(font=dict(size=9.5), bgcolor='rgba(0,0,0,0)', borderwidth=0),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    with col_insights:
        cluster_avg_inc = df[df['Cluster']==cluster]['Income'].mean()
        cluster_avg_sp  = df[df['Cluster']==cluster]['Spending'].mean()
        tips = []

        # Income tips
        if income < 30:
            tips.append(("bad","⚠","Low Income Bracket","Price sensitivity expected — value messaging is key."))
        elif income < 60:
            tips.append(("warn","→","Mid Income Range","Moderate price sensitivity — balance quality & value."))
        else:
            tips.append(("ok","✓","High Income Bracket","Receptive to premium and aspirational products."))

        # Spending tips
        if spending < 30:
            tips.append(("bad","⚠","Low Spending Score","Disengaged buyer — re-engagement campaigns needed."))
        elif spending < 65:
            tips.append(("warn","→","Moderate Spender","Growth potential with targeted nudges."))
        else:
            tips.append(("ok","✓","High Spending Score","Active buyer — prioritise upsell & retention."))

        # Age tips
        if age < 30:
            tips.append(("info","★","Young Demographic","Social proof, trend-driven offers work best."))
        elif age > 55:
            tips.append(("info","★","Mature Demographic","Trust, quality, and loyalty programs resonate."))

        # Cluster fit
        inc_diff = income - cluster_avg_inc
        tips.append(("ok" if abs(inc_diff) < 10 else "warn",
                     "◈", "Cluster Fit",
                     f"Income is {abs(inc_diff):.0f}k {'above' if inc_diff>0 else 'below'} cluster average of {cluster_avg_inc:.0f}k."))

        for sev, ico, title, body in tips[:5]:
            st.markdown(f"""
            <div class="insight {sev}">
              <span class="insight-ico">{ico}</span>
              <div><div class="insight-title">{title}</div><div class="insight-body">{body}</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="potential-box">
          <div class="pot-eyebrow">Recommended Strategy</div>
          <div class="pot-val">{m['strategy']}</div>
          <div class="pot-sub">{m['tag']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "explorer":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">What-If Analysis</div>
      <div class="page-title">Segment <em>Explorer</em></div>
      <div class="page-desc">Set a baseline profile and see how adjusting income or spending shifts the cluster assignment.</div>
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
        <div style="font-size:.65rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;margin-bottom:3px;">Baseline Segment</div>
        <div class="baseline-seg">{bm['name']}</div>
      </div>
      <span class="segment-badge" style="background:{bm['light']};color:{bm['color']};border-color:{bm['brd']};">
        ● &nbsp;Cluster {base_cluster}
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Adjustment Scenarios</div>', unsafe_allow_html=True)

    scenarios = [
        ("+10k Income",   base_inc+10,  base_sp),
        ("+20k Income",   base_inc+20,  base_sp),
        ("−10k Income",   base_inc-10,  base_sp),
        ("+20 Spending",  base_inc,     min(base_sp+20, 100)),
        ("+40 Spending",  base_inc,     min(base_sp+40, 100)),
        ("−20 Spending",  base_inc,     max(base_sp-20, 1)),
        ("Premium Move",  min(base_inc+25, 137), min(base_sp+25, 100)),
    ]

    sim_cols = st.columns(len(scenarios), gap="small")
    sim_results = []
    for col, (lbl, inc_s, sp_s) in zip(sim_cols, scenarios):
        c = classify(np.clip(inc_s, 15, 137), np.clip(sp_s, 1, 100))
        sim_results.append(c)
        mi = meta[c]
        changed = c != base_cluster
        dc = "pos" if changed else "neu"
        with col:
            st.markdown(f"""
            <div class="sim-card">
              <div class="sim-lbl">{lbl}</div>
              <div class="sim-seg" style="color:{mi['color']};">{mi['name'].split()[0]}</div>
              <div class="sim-delta {dc}">{'→ Shifted' if changed else '· Same'}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

    # Income sweep chart
    inc_range = np.arange(15, 138, 3)
    seg_sweep  = [classify(i, base_sp) for i in inc_range]
    seg_colors_sweep = [SEG_COLORS[c] for c in seg_sweep]

    fig_sweep = go.Figure()
    for i in range(5):
        mask = np.array(seg_sweep) == i
        if mask.any():
            fig_sweep.add_trace(go.Scatter(
                x=inc_range[mask], y=np.array(seg_sweep)[mask],
                mode='markers', marker=dict(color=SEG_COLORS[i], size=9, opacity=0.85),
                name=meta[i]['name'],
                hovertemplate=f'Income: %{{x}}k → {meta[i]["name"]}<extra></extra>',
            ))
    fig_sweep.add_vline(x=base_inc, line=dict(color='#C47C0A', width=1.5, dash='dot'))
    fig_sweep.update_layout(
        **CHART_DEFAULTS,
        margin=dict(l=0, r=16, t=36, b=6), height=220,
        title=dict(text="Cluster Assignment · Income Sweep (Spending Fixed)",
                   font=dict(size=11.5, color='#5C5852', family='Geist'), x=0),
        xaxis=dict(title="Annual Income (k$)", gridcolor='#EEEBE5', zeroline=False,
                   tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
        yaxis=dict(title="Cluster ID", gridcolor='#EEEBE5', zeroline=False, dtick=1,
                   tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
        legend=dict(font=dict(size=9.5), bgcolor='rgba(0,0,0,0)', borderwidth=0),
    )
    st.plotly_chart(fig_sweep, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "analytics":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Data Analytics</div>
      <div class="page-title">Cluster <em>Analytics</em></div>
      <div class="page-desc">Distribution maps, age analysis, and a 2D density map computed on the full dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: scatter + cluster sizes
    r1a, r1b = st.columns(2, gap="medium")

    with r1a:
        st.markdown('<div class="sec-label">Segmentation Map</div>', unsafe_allow_html=True)
        fig_sc = go.Figure()
        for i, mi in enumerate(meta):
            mask = df['Cluster'] == i
            fig_sc.add_trace(go.Scatter(
                x=df.loc[mask,'Income'], y=df.loc[mask,'Spending'],
                mode='markers',
                marker=dict(color=mi['color'], size=7, opacity=0.72,
                            line=dict(color='white', width=0.5)),
                name=mi['name'],
                hovertemplate='Income: %{x:.0f}k · Spending: %{y:.0f}<extra>' + mi['name'] + '</extra>',
            ))
        fig_sc.add_trace(go.Scatter(
            x=centers[:,0], y=centers[:,1], mode='markers',
            marker=dict(symbol='diamond', color='#18160F', size=12,
                        line=dict(color='white', width=1.5)),
            name='Centroids', hoverinfo='skip'
        ))
        fig_sc.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0,r=16,t=10,b=6), height=310,
            xaxis=dict(title="Income (k$)", gridcolor='#EEEBE5', zeroline=False,
                       tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
            yaxis=dict(title="Spending Score", gridcolor='#EEEBE5', zeroline=False,
                       tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
            legend=dict(font=dict(size=9.5), bgcolor='rgba(0,0,0,0)', borderwidth=0),
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})

    with r1b:
        st.markdown('<div class="sec-label">Cluster Sizes</div>', unsafe_allow_html=True)
        counts = [int((df['Cluster']==i).sum()) for i in range(5)]
        fig_bar = go.Figure(go.Bar(
            x=[mi['name'] for mi in meta], y=counts,
            marker=dict(color=SEG_COLORS, opacity=0.88, cornerradius=5, line=dict(width=0)),
            text=[str(c) for c in counts], textposition='outside',
            textfont=dict(size=11, color='#9C9890', family='Geist Mono'),
            hovertemplate='%{x}: %{y} customers<extra></extra>', showlegend=False, width=0.65,
        ))
        fig_bar.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0,r=16,t=10,b=6), height=310,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickangle=-15,
                       tickfont=dict(size=10, color='#18160F')),
            yaxis=dict(gridcolor='#EEEBE5', zeroline=False,
                       tickfont=dict(size=9.5, family='Geist Mono')),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

    # ── Row 2: age dist + heatmap
    r2a, r2b = st.columns(2, gap="medium")

    with r2a:
        st.markdown('<div class="sec-label">Age Distribution by Segment</div>', unsafe_allow_html=True)
        fig_age = go.Figure()
        for i, mi in enumerate(meta):
            ages = df[df['Cluster']==i]['Age']
            fig_age.add_trace(go.Box(
                y=ages, name=mi['name'].split()[0],
                marker=dict(color=mi['color']),
                line=dict(color=mi['color']),
                fillcolor=mi['light'],
                hovertemplate='%{y} yrs<extra>' + mi['name'] + '</extra>',
            ))
        fig_age.update_layout(
            **CHART_DEFAULTS,
            margin=dict(l=0,r=16,t=10,b=6), height=270,
            xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10, color='#18160F')),
            yaxis=dict(gridcolor='#EEEBE5', zeroline=False,
                       tickfont=dict(size=9.5, family='Geist Mono'), title="Age"),
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
                [0.0, '#FEF2F2'], [0.25, '#FDF8EC'],
                [0.5, '#EDF8F2'], [0.75, '#EEF3FD'], [1.0, '#F5F3FF']
            ],
            hovertemplate='Income: %{x}k · Spending: %{y} → Cluster %{z}<extra></extra>',
            colorbar=dict(
                tickfont=dict(size=9.5, family='Geist Mono', color='#9C9890'),
                thickness=12, outlinewidth=0, title=dict(text='Cluster', side='right',
                font=dict(size=10, family='Geist', color='#9C9890'))
            ),
            showscale=True,
        ))
        fig_hm.update_layout(
            **CHART_DEFAULTS, margin=dict(l=0,r=10,t=10,b=10), height=270,
            xaxis=dict(title="Annual Income (k$)", gridcolor='rgba(0,0,0,0)',
                       tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
            yaxis=dict(title="Spending Score", gridcolor='rgba(0,0,0,0)',
                       tickfont=dict(size=9.5, family='Geist Mono'), title_font=dict(size=10.5)),
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
      <div class="page-eyebrow">Documentation</div>
      <div class="page-title">About <em>SegmentIQ</em></div>
      <div class="page-desc">Model architecture, feature descriptions, and cluster reference guide.</div>
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
    for h, lbl in zip(hc, ["#", "Segment", "Income Range", "Spending", "Strategy"]):
        with h:
            st.markdown(f'<div style="font-size:.63rem;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.12em;padding:.4rem 0;">{lbl}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:var(--border2);margin:.25rem 0 .5rem;"></div>', unsafe_allow_html=True)

    inc_ranges  = ["15–40k",  "15–40k",  "40–70k",  "70–137k", "70–137k"]
    sp_ranges   = ["1–40",    "60–100",  "35–70",   "60–100",  "1–40"]
    for i, (mi, inc_r, sp_r) in enumerate(zip(meta, inc_ranges, sp_ranges)):
        bg = "background:#FAFAF7;" if i%2==0 else ""
        fc = st.columns([0.4, 2, 1.5, 1.2, 3])
        data = [str(i), mi['name'], inc_r, sp_r, mi['strategy']]
        for col, txt, mono in zip(fc, data, [True, False, True, True, False]):
            with col:
                ff = "font-family:'Geist Mono',monospace;font-size:.77rem;" if mono else "font-size:.79rem;"
                color = f"color:{mi['color']};" if not mono or data.index(txt)==0 else ""
                st.markdown(f'<div style="{ff}{color}color:var(--text);padding:.44rem 0;{bg}">{txt}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Segment Cards</div>', unsafe_allow_html=True)
    gcols = st.columns(5, gap="medium")
    for col, mi, i in zip(gcols, meta, range(5)):
        count = int((df['Cluster']==i).sum())
        with col:
            st.markdown(f"""
            <div style="background:{mi['light']};border:1.5px solid {mi['brd']};border-radius:16px;padding:1.75rem 1rem;text-align:center;">
              <div style="font-family:'Instrument Serif',serif;font-size:2.5rem;color:{mi['color']};line-height:1;">{i}</div>
              <div style="font-size:.76rem;font-weight:700;color:{mi['color']};margin-top:9px;">{mi['name']}</div>
              <div style="font-family:'Geist Mono',monospace;font-size:.67rem;color:var(--text3);margin-top:6px;font-weight:500;">{count} customers</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="shell" style="padding-top:0;padding-bottom:1rem;">
  <div class="app-footer">
    SegmentIQ v1.0
    <span class="footer-sep">·</span>K-Means Clustering
    <span class="footer-sep">·</span>200 customers
    <span class="footer-sep">·</span>5 segments
    <span class="footer-sep">·</span>Task 2 · ML Internship
  </div>
</div>
""", unsafe_allow_html=True)
