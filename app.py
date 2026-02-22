"""
╔══════════════════════════════════════════════════════════════╗
║     🛍️ Customer Segmentation — Streamlit App (Redesigned)   ║
║     Task 2 | Clustering | ML Internship                     ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run app1_redesigned.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — Editorial Brutalist / Ink & Paper ────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #F5F0E8;
        color: #1A1410;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1A1410 !important;
        border-right: 3px solid #E8B84B;
    }
    [data-testid="stSidebar"] * {
        color: #F5F0E8 !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #E8B84B !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Syne', sans-serif;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #E8B84B !important;
        border-bottom: 1px solid #333 !important;
        padding-bottom: 0.4rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem !important;
        color: #E8B84B !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888 !important;
    }

    /* ── Main App Background ── */
    .stApp { background-color: #F5F0E8; }
    [data-testid="stAppViewContainer"] { background-color: #F5F0E8; }

    /* ── Hero ── */
    .hero-wrap {
        border: 3px solid #1A1410;
        background: #1A1410;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: #E8B84B;
        border-radius: 50%;
        opacity: 0.15;
    }
    .hero-wrap::after {
        content: 'SEGMENTATION ENGINE';
        position: absolute;
        bottom: 10px; right: 20px;
        font-family: 'Syne', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 0.25em;
        color: #444;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #F5F0E8;
        margin: 0;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    .hero-title span { color: #E8B84B; }
    .hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.6rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 3px solid #1A1410;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888 !important;
        padding: 0.7rem 1.5rem;
        border: none;
        background: transparent;
        border-bottom: 3px solid transparent;
        margin-bottom: -3px;
    }
    .stTabs [aria-selected="true"] {
        color: #1A1410 !important;
        border-bottom: 3px solid #E8B84B !important;
        background: transparent !important;
    }

    /* ── Cards ── */
    .ink-card {
        border: 2px solid #1A1410;
        background: #FDFAF4;
        padding: 1.4rem;
        margin: 0.5rem 0;
        position: relative;
    }
    .ink-card::before {
        content: '';
        position: absolute;
        top: 4px; left: 4px; right: -4px; bottom: -4px;
        border: 2px solid #1A1410;
        z-index: -1;
    }
    .ink-card .card-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.2rem;
    }
    .ink-card .card-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #1A1410;
        line-height: 1;
    }
    .ink-card .card-sub {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.3rem;
    }

    /* ── Segment Result ── */
    .segment-result {
        border: 3px solid #1A1410;
        background: #1A1410;
        color: #F5F0E8;
        padding: 2rem;
        position: relative;
    }
    .segment-result .seg-number {
        font-family: 'Syne', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        color: #E8B84B;
        line-height: 1;
        opacity: 0.4;
        position: absolute;
        top: 1rem; right: 1.5rem;
    }
    .segment-result .seg-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: #F5F0E8;
    }
    .segment-result .seg-badge {
        display: inline-block;
        background: #E8B84B;
        color: #1A1410;
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        padding: 0.3rem 0.8rem;
        margin-bottom: 1rem;
    }
    .segment-result .seg-strategy {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #aaa;
        border-top: 1px solid #333;
        padding-top: 0.8rem;
        margin-top: 0.8rem;
    }
    .segment-result .seg-strategy b { color: #E8B84B; }

    /* ── Segment Mini Cards ── */
    .seg-mini {
        border: 2px solid #1A1410;
        padding: 0.8rem 0.6rem;
        text-align: center;
        background: #FDFAF4;
        position: relative;
    }
    .seg-mini.active {
        background: #1A1410;
    }
    .seg-mini .mini-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .seg-mini .mini-name {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .seg-mini.active .mini-name { color: #E8B84B; }
    .seg-mini .mini-count {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #1A1410;
        display: block;
        margin: 0.2rem 0;
    }
    .seg-mini.active .mini-count { color: #F5F0E8; }
    .seg-mini .mini-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .seg-mini.active .mini-label { color: #666; }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #888;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #ddd;
    }

    /* ── Button ── */
    .stButton > button {
        background: #E8B84B !important;
        color: #1A1410 !important;
        border: 2px solid #1A1410 !important;
        border-radius: 0 !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        width: 100% !important;
        box-shadow: 4px 4px 0 #1A1410 !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        box-shadow: 2px 2px 0 #1A1410 !important;
        transform: translate(2px, 2px) !important;
    }

    /* ── Misc cleanup ── */
    hr { border-color: #ddd; }
    footer { visibility: hidden; }
    [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F5F0E8; }
    ::-webkit-scrollbar-thumb { background: #1A1410; }
</style>
""", unsafe_allow_html=True)

# ── Load or Train Model ────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    if os.path.exists('kmeans_model.pkl'):
        kmeans = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('customer_scaler.pkl')
    else:
        np.random.seed(42)
        n = 200
        income = np.concatenate([
            np.random.normal(25, 8, 40), np.random.normal(25, 8, 40),
            np.random.normal(55, 10, 40), np.random.normal(85, 10, 40),
            np.random.normal(85, 10, 40)
        ])
        spending = np.concatenate([
            np.random.normal(20, 10, 40), np.random.normal(75, 10, 40),
            np.random.normal(50, 10, 40), np.random.normal(80, 10, 40),
            np.random.normal(20, 10, 40)
        ])
        X = np.column_stack([np.clip(income, 15, 137), np.clip(spending, 1, 100)])
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=15)
        kmeans.fit(X_sc)

    np.random.seed(42)
    n = 200
    income_vals = np.concatenate([np.random.normal(25,8,40), np.random.normal(25,8,40), np.random.normal(55,10,40), np.random.normal(85,10,40), np.random.normal(85,10,40)])
    spending_vals = np.concatenate([np.random.normal(20,10,40), np.random.normal(75,10,40), np.random.normal(50,10,40), np.random.normal(80,10,40), np.random.normal(20,10,40)])
    age_vals = np.random.randint(18, 70, n)
    X_disp = np.column_stack([np.clip(income_vals, 15, 137), np.clip(spending_vals, 1, 100)])
    labels = kmeans.predict(scaler.transform(X_disp))

    df_disp = pd.DataFrame({'Income': X_disp[:,0], 'Spending': X_disp[:,1], 'Age': age_vals, 'Cluster': labels})
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    cluster_meta = [
        {'color': '#E05C47', 'dot': '#E05C47', 'name': 'Budget Enthusiasts',  'tag': 'LOW INCOME · LOW SPEND',    'strategy': 'Value deals & flash sales'},
        {'color': '#E8B84B', 'dot': '#E8B84B', 'name': 'Premium Shoppers',    'tag': 'LOW INCOME · HIGH SPEND',   'strategy': 'VIP programs & loyalty rewards'},
        {'color': '#5BAD7A', 'dot': '#5BAD7A', 'name': 'Average Customers',   'tag': 'MID INCOME · MID SPEND',    'strategy': 'Mainstream promotions'},
        {'color': '#4D7FBF', 'dot': '#4D7FBF', 'name': 'Careful Spenders',    'tag': 'HIGH INCOME · HIGH SPEND',  'strategy': 'Luxury upselling campaigns'},
        {'color': '#9B6BBF', 'dot': '#9B6BBF', 'name': 'Low Engagement',      'tag': 'HIGH INCOME · LOW SPEND',   'strategy': 'Re-engagement & discounts'},
    ]

    return kmeans, scaler, df_disp, centers, cluster_meta

kmeans, scaler, df_disp, centers, cluster_meta = load_model_and_data()

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Customer<br><span>Segmentation</span></div>
    <div class="hero-sub">◈ AI-powered clustering &nbsp;·&nbsp; K-Means · 5 segments · 200 customers</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────
col_left, col_main = st.columns([1, 3], gap="large")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ Input Profile")

    income = st.slider("Annual Income (k$)", 15, 137, 65)
    spending = st.slider("Spending Score (1–100)", 1, 100, 50)
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    st.markdown("---")
    classify_btn = st.button("◈ Classify Customer")

    st.markdown("---")
    st.markdown("### ─ Dataset Stats")
    st.metric("Customers", "200")
    st.metric("Segments", "5")
    st.metric("Algorithm", "K-Means")

# ── Predict ────────────────────────────────────────────────────
user_features = np.array([[income, spending]])
user_scaled = scaler.transform(user_features)
user_cluster = int(kmeans.predict(user_scaled)[0])
meta = cluster_meta[user_cluster % len(cluster_meta)]

# ── Main ───────────────────────────────────────────────────────
with col_main:
    tab1, tab2, tab3 = st.tabs(["◈ Your Segment", "◈ Cluster Map", "◈ Analysis"])

    # ── TAB 1 ──────────────────────────────────────────────────
    with tab1:
        col_r, col_p = st.columns([1, 1], gap="medium")

        with col_r:
            st.markdown(f"""
            <div class="segment-result">
                <div class="seg-number">{user_cluster}</div>
                <div class="seg-badge">Cluster {user_cluster}</div>
                <div class="seg-name">{meta['name']}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem; 
                            color:#888; letter-spacing:0.12em; margin-top:0.4rem;">
                    {meta['tag']}
                </div>
                <div class="seg-strategy">
                    <b>Strategy →</b> {meta['strategy']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_p:
            st.markdown('<div class="section-header">Profile vs Cluster Average</div>', unsafe_allow_html=True)
            cluster_avg = df_disp[df_disp['Cluster'] == user_cluster][['Income', 'Spending']].mean()
            metrics_data = [
                ("Annual Income", f"${income}k", f"Cluster avg: ${cluster_avg['Income']:.0f}k"),
                ("Spending Score", f"{spending}/100",   f"Cluster avg: {cluster_avg['Spending']:.0f}/100"),
                ("Age",           str(age),             ""),
            ]
            for label, val, sub in metrics_data:
                st.markdown(f"""
                <div class="ink-card">
                    <div class="card-label">{label}</div>
                    <div class="card-value">{val}</div>
                    <div class="card-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        # All segments overview
        st.markdown('<div class="section-header">All Customer Segments</div>', unsafe_allow_html=True)
        seg_cols = st.columns(5)
        for i, (col, m) in enumerate(zip(seg_cols, cluster_meta)):
            count = int((df_disp['Cluster'] == i).sum())
            active_cls = "active" if i == user_cluster else ""
            with col:
                st.markdown(f"""
                <div class="seg-mini {active_cls}">
                    <span class="mini-dot" style="background:{m['color']}"></span><br>
                    <span class="mini-name">{m['name'].split()[0]}<br>{m['name'].split()[1] if len(m['name'].split()) > 1 else ''}</span>
                    <span class="mini-count">{count}</span>
                    <span class="mini-label">customers</span>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2 ──────────────────────────────────────────────────
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#F5F0E8')
        ax.set_facecolor('#FDFAF4')

        for i, m in enumerate(cluster_meta):
            mask = df_disp['Cluster'] == i
            ax.scatter(df_disp.loc[mask, 'Income'], df_disp.loc[mask, 'Spending'],
                       c=m['color'], s=55, alpha=0.8, edgecolors='#1A1410', linewidth=0.4,
                       label=m['name'], zorder=3)

        # Centroids
        ax.scatter(centers[:, 0], centers[:, 1], c='#1A1410', s=220, marker='D', zorder=6,
                   label='Centroids', edgecolors='#E8B84B', linewidth=1.5)

        # User point
        ax.scatter([income], [spending], c='#E8B84B', s=280, marker='*', zorder=7,
                   edgecolors='#1A1410', linewidth=1.5, label=f'You ({income}k, {spending})')

        ax.set_xlabel('Annual Income (k$)', color='#1A1410', fontfamily='monospace', fontsize=9)
        ax.set_ylabel('Spending Score (1–100)', color='#1A1410', fontfamily='monospace', fontsize=9)
        ax.set_title('CUSTOMER SEGMENTATION MAP', color='#1A1410', fontsize=11, fontweight='bold',
                     fontfamily='monospace', loc='left', pad=12)
        ax.tick_params(colors='#555', labelsize=8)
        ax.grid(True, alpha=0.35, color='#C8C0B0', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1A1410')
            spine.set_linewidth(1.5)
        ax.legend(loc='upper left', fontsize=7.5, facecolor='#F5F0E8', edgecolor='#1A1410',
                  labelcolor='#1A1410', framealpha=1)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ── TAB 3 ──────────────────────────────────────────────────
    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#F5F0E8')
        for ax in axes:
            ax.set_facecolor('#FDFAF4')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1A1410')
                spine.set_linewidth(1.5)

        # Cluster sizes
        counts = [int((df_disp['Cluster'] == i).sum()) for i in range(len(cluster_meta))]
        colors = [m['color'] for m in cluster_meta]
        bars = axes[0].bar(range(len(cluster_meta)), counts, color=colors,
                           edgecolor='#1A1410', linewidth=1, alpha=0.9, width=0.6)
        axes[0].set_xticks(range(len(cluster_meta)))
        axes[0].set_xticklabels([m['name'].split()[0] for m in cluster_meta],
                                 rotation=15, ha='right', color='#1A1410', fontsize=8, fontfamily='monospace')
        axes[0].set_ylabel('Count', color='#1A1410', fontfamily='monospace', fontsize=8)
        axes[0].set_title('CLUSTER SIZES', color='#1A1410', fontweight='bold',
                           fontfamily='monospace', fontsize=9, loc='left')
        for i, v in enumerate(counts):
            axes[0].text(i, v + 0.3, str(v), ha='center', color='#1A1410', fontsize=9,
                         fontfamily='monospace', fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.3, color='#C8C0B0', linewidth=0.5)
        axes[0].tick_params(colors='#555', labelsize=8)

        # Income vs Spending
        avg_inc = [df_disp[df_disp['Cluster']==i]['Income'].mean() for i in range(len(cluster_meta))]
        avg_sp  = [df_disp[df_disp['Cluster']==i]['Spending'].mean() for i in range(len(cluster_meta))]
        x = range(len(cluster_meta))
        width = 0.38
        axes[1].bar([xi - width/2 for xi in x], avg_inc, width, color='#4D7FBF',
                    alpha=0.9, label='Avg Income (k$)', edgecolor='#1A1410', linewidth=0.8)
        axes[1].bar([xi + width/2 for xi in x], avg_sp, width, color='#E8B84B',
                    alpha=0.9, label='Avg Spending Score', edgecolor='#1A1410', linewidth=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m['name'].split()[0] for m in cluster_meta],
                                  rotation=15, ha='right', color='#1A1410', fontsize=8, fontfamily='monospace')
        axes[1].set_title('INCOME vs SPENDING BY CLUSTER', color='#1A1410', fontweight='bold',
                           fontfamily='monospace', fontsize=9, loc='left')
        axes[1].legend(facecolor='#F5F0E8', edgecolor='#1A1410', labelcolor='#1A1410', fontsize=8)
        axes[1].grid(True, axis='y', alpha=0.3, color='#C8C0B0', linewidth=0.5)
        axes[1].tick_params(colors='#555', labelsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:left; font-family:'DM Mono',monospace; color:#aaa; 
            font-size:0.72rem; padding:1.5rem 0; letter-spacing:0.1em;
            border-top: 1px solid #ddd; margin-top:2rem;">
    ◈ CUSTOMER SEGMENTATION &nbsp;·&nbsp; TASK 2: CLUSTERING &nbsp;·&nbsp; ML INTERNSHIP &nbsp;·&nbsp; STREAMLIT + SCIKIT-LEARN
</div>
""", unsafe_allow_html=True)
