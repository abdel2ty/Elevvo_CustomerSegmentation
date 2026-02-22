"""
╔══════════════════════════════════════════════════════════════╗
║     🛍️ Customer Segmentation — Streamlit App                ║
║     Task 2 | Clustering | ML Internship                     ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run app.py
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
    page_title="🛍️ Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .hero {
        background: linear-gradient(135deg, #FF6B6B 0%, #C77DFF 50%, #4D96FF 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(199,125,255,0.4);
    }
    .hero h1 { color: white; font-size: 2.4rem; margin: 0; font-weight: 800; }
    .hero p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; font-size: 1.05rem; }
    
    .cluster-card {
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.2s;
    }
    .cluster-card:hover { transform: translateY(-3px); }
    .cluster-card .name { font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; }
    .cluster-card .stat { font-size: 0.82rem; color: #ccc; }
    
    .result-badge {
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 800;
        font-size: 1.3rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #C77DFF, #4D96FF);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        width: 100%;
        font-size: 1.05rem;
        box-shadow: 0 8px 24px rgba(199,125,255,0.4);
        transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-2px); }
    
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load or Train Model ────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    CLUSTER_COLORS = ['#FF6B6B', '#FFD93D', '#6BCB77', '#4D96FF', '#C77DFF']
    
    if os.path.exists('kmeans_model.pkl'):
        kmeans = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('customer_scaler.pkl')
    else:
        # Train on synthetic data
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

    # Generate display data
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
        {'color': '#FF6B6B', 'name': '💸 Budget Enthusiasts', 'strategy': 'Value deals & flash sales'},
        {'color': '#FFD93D', 'name': '🏆 Premium Shoppers', 'strategy': 'VIP programs & luxury offers'},
        {'color': '#6BCB77', 'name': '⚖️ Average Customers', 'strategy': 'Mainstream promotions'},
        {'color': '#4D96FF', 'name': '💰 Careful Spenders', 'strategy': 'Luxury upselling campaigns'},
        {'color': '#C77DFF', 'name': '📉 Low Engagement', 'strategy': 'Re-engagement & discounts'},
    ]
    
    return kmeans, scaler, df_disp, centers, cluster_meta

kmeans, scaler, df_disp, centers, cluster_meta = load_model_and_data()

# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛍️ Customer Segmentation Engine</h1>
    <p>AI-powered customer clustering for smarter marketing strategies</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────
col_sidebar_content, col_main = st.columns([1, 3], gap="large")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Profile")
    st.markdown("---")
    
    income = st.slider("💰 Annual Income (k$)", 15, 137, 65,
                        help="Customer's annual income in thousands of dollars")
    spending = st.slider("🛍️ Spending Score (1-100)", 1, 100, 50,
                          help="Mall-assigned score based on purchase behavior")
    age = st.number_input("🎂 Age", min_value=18, max_value=80, value=35)
    gender = st.radio("👤 Gender", ["Male", "Female"], horizontal=True)
    
    st.markdown("---")
    classify_btn = st.button("🔍 Find My Segment", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.metric("Total Customers", "200")
    st.metric("Clusters", "5")
    st.metric("Features Used", "Income + Spending")

# ── Predict Cluster ────────────────────────────────────────────
user_features = np.array([[income, spending]])
user_scaled = scaler.transform(user_features)
user_cluster = int(kmeans.predict(user_scaled)[0])
meta = cluster_meta[user_cluster % len(cluster_meta)]

# ── Main Viz ───────────────────────────────────────────────────
with col_main:
    tab1, tab2, tab3 = st.tabs(["🎯 Your Segment", "📊 Cluster Map", "📈 Analysis"])
    
    with tab1:
        col_result, col_profile = st.columns([1, 1])
        
        with col_result:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {meta['color']}33, {meta['color']}11); 
                        border: 2px solid {meta['color']}; border-radius: 20px; padding: 2rem; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🏷️</div>
                <div style="color: {meta['color']}; font-size: 1.4rem; font-weight: 800;">{meta['name']}</div>
                <div style="color: #aaa; margin-top: 0.8rem; font-size: 0.9rem;">Cluster {user_cluster}</div>
                <div style="background: {meta['color']}22; border-radius: 8px; padding: 0.8rem; margin-top: 1rem;">
                    <div style="color: #ccc; font-size: 0.85rem;">📣 Marketing Strategy</div>
                    <div style="color: white; font-weight: 600; margin-top: 0.3rem;">{meta['strategy']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_profile:
            st.markdown("#### 📋 Your Profile vs Cluster Average")
            cluster_avg = df_disp[df_disp['Cluster'] == user_cluster][['Income', 'Spending']].mean()
            
            metrics = [
                ("💰 Your Income", f"${income}k", f"Cluster avg: ${cluster_avg['Income']:.0f}k"),
                ("🛍️ Your Spending", f"{spending}/100", f"Cluster avg: {cluster_avg['Spending']:.0f}/100"),
                ("🎂 Your Age", str(age), ""),
            ]
            for label, val, delta in metrics:
                st.markdown(f"""
                <div style="background: #1e1e3a; border-radius: 10px; padding: 1rem; margin: 0.4rem 0; 
                            border-left: 3px solid {meta['color']};">
                    <div style="color: #aaa; font-size: 0.8rem;">{label}</div>
                    <div style="color: white; font-size: 1.5rem; font-weight: 700;">{val}</div>
                    <div style="color: #888; font-size: 0.75rem;">{delta}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # All segments overview
        st.markdown("#### 🗂️ All Customer Segments")
        seg_cols = st.columns(5)
        for i, (col, m) in enumerate(zip(seg_cols, cluster_meta)):
            count = (df_disp['Cluster'] == i).sum()
            bg = f"border: 2px solid {m['color']}; background: {m['color']}22;" if i == user_cluster else f"background: #1e1e3a; border: 1px solid #333;"
            with col:
                st.markdown(f"""
                <div style="{bg} border-radius: 12px; padding: 0.8rem; text-align: center;">
                    <div style="color: {m['color']}; font-size: 0.8rem; font-weight: 700;">{m['name'].split()[-1]}</div>
                    <div style="color: white; font-size: 1.3rem; font-weight: 800; margin-top: 0.3rem;">{count}</div>
                    <div style="color: #888; font-size: 0.7rem;">customers</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        
        for i, m in enumerate(cluster_meta):
            mask = df_disp['Cluster'] == i
            ax.scatter(df_disp.loc[mask, 'Income'], df_disp.loc[mask, 'Spending'],
                      c=m['color'], s=60, alpha=0.75, edgecolors='white', linewidth=0.3,
                      label=m['name'])
        
        # Centroids
        ax.scatter(centers[:, 0], centers[:, 1], c='white', s=300, marker='*', zorder=6, label='Centroids')
        
        # User point
        ax.scatter([income], [spending], c='#FFD93D', s=300, marker='D', zorder=7,
                   edgecolors='white', linewidth=2, label=f'You ({income}k, {spending})')
        
        ax.set_xlabel('Annual Income (k$)', color='#c9d1d9')
        ax.set_ylabel('Spending Score (1-100)', color='#c9d1d9')
        ax.set_title('Customer Segmentation Map', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#8b949e')
        ax.grid(True, alpha=0.3, color='#21262d')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', fontsize=8, facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#161b22')
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
        
        # Cluster sizes
        counts = [int((df_disp['Cluster'] == i).sum()) for i in range(len(cluster_meta))]
        colors = [m['color'] for m in cluster_meta]
        axes[0].bar(range(len(cluster_meta)), counts, color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)
        axes[0].set_xticks(range(len(cluster_meta)))
        axes[0].set_xticklabels([m['name'].split()[-1] for m in cluster_meta], rotation=15, ha='right', color='#c9d1d9')
        axes[0].set_ylabel('Count', color='#c9d1d9')
        axes[0].set_title('Cluster Sizes', color='white', fontweight='bold')
        for i, v in enumerate(counts):
            axes[0].text(i, v+0.5, str(v), ha='center', color='white', fontsize=11)
        axes[0].grid(True, axis='y', alpha=0.3, color='#21262d')
        axes[0].tick_params(colors='#8b949e')
        
        # Income vs Spending per cluster
        avg_inc = [df_disp[df_disp['Cluster']==i]['Income'].mean() for i in range(len(cluster_meta))]
        avg_sp = [df_disp[df_disp['Cluster']==i]['Spending'].mean() for i in range(len(cluster_meta))]
        x = range(len(cluster_meta))
        width = 0.4
        axes[1].bar([xi - width/2 for xi in x], avg_inc, width, color='#4D96FF', alpha=0.85, label='Avg Income (k$)', edgecolor='white', lw=0.5)
        axes[1].bar([xi + width/2 for xi in x], avg_sp, width, color='#FFD93D', alpha=0.85, label='Avg Spending Score', edgecolor='white', lw=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m['name'].split()[-1] for m in cluster_meta], rotation=15, ha='right', color='#c9d1d9')
        axes[1].set_title('Income vs Spending by Cluster', color='white', fontweight='bold')
        axes[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
        axes[1].grid(True, axis='y', alpha=0.3, color='#21262d')
        axes[1].tick_params(colors='#8b949e')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.85rem; padding: 1rem; margin-top: 2rem;">
    🛍️ Customer Segmentation · Task 2: Clustering · ML Internship · Built with Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
