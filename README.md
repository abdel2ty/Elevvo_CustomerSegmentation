# ◈ SegmentIQ — Customer Segmentation

> **Discover who your customers really are.** SegmentIQ is an interactive ML-powered web application that clusters customers into distinct behavioral segments using K-Means — enabling businesses to personalize strategies, optimize targeting, and understand spending patterns at a glance.

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-8B5CF6?style=for-the-badge)](https://elevvo-customer-segmentation.streamlit.app/)
[![Kaggle Notebook](https://img.shields.io/badge/📓_Kaggle_Notebook-abdel2ty-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/abdel2ty/elevvo-customer-segmentation)

---

## 📌 Overview

SegmentIQ applies **K-Means Clustering (k=5)** to 200 synthetic customer records, segmenting them by Annual Income and Spending Score. The app goes beyond static charts — it provides real-time segment prediction for any new customer input, scenario comparison tools, an interactive data explorer, and rich cluster analytics to drive data-informed marketing decisions.

---

## ✨ Features

### 📊 Overview Page
- Global cluster distribution chart
- Income vs. Spending scatter plot colored by segment
- Age distribution histogram per segment
- KPI strip: total records, segment count, feature count, gender split

### 🔍 Classify Page
- Input a customer's income, spending score, and age to instantly classify them into one of 5 segments
- Color-coded segment card with name, description, and recommended marketing strategy

### 🔄 What-If / Scenario Page
- Define and compare up to multiple customer scenarios simultaneously
- Visualize scenario positions on the cluster proximity map
- See cluster assignment shifts across income and spending combinations

### 📋 Data Page
- Filterable data table (by segment, gender, age range, sort order)
- Download filtered records as CSV
- Filtered overview charts: segment counts, scatter, age distribution

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | K-Means Clustering |
| Library | scikit-learn |
| Number of Clusters | 5 |
| Training Records | 200 (synthetic, seed=42) |
| Features | Annual Income (k$), Spending Score (1–100) |

---

## 🏷️ Customer Segments

| Cluster | Name | Profile |
|---|---|---|
| 0 | Budget Cautious | Low income, low spending |
| 1 | High Earner, Low Spend | High income, low spending |
| 2 | Balanced Mainstream | Mid income, mid spending |
| 3 | Impulsive Spender | Low income, high spending |
| 4 | Premium Loyalist | High income, high spending |

---

## 📥 Input Features

| Feature | Range | Type |
|---|---|---|
| Annual Income | 15k – 137k USD | Numeric |
| Spending Score | 1 – 100 | Numeric |
| Age | 18 – 80 | Numeric |
| Gender | Male / Female | Categorical |

---

## 🛠️ Tech Stack

- **Frontend / App Framework** — [Streamlit](https://streamlit.io/)
- **ML Model** — scikit-learn (KMeans, StandardScaler)
- **Visualizations** — Plotly (scatter, bar, histogram, heatmap)
- **Data Handling** — pandas, numpy
- **UI Design** — Custom glassmorphism CSS (Manrope, Sora, JetBrains Mono fonts)

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd segmentiq

# 2. Install dependencies
pip install streamlit numpy pandas plotly scikit-learn

# 3. Launch the app
streamlit run app2.py
```

---

## 🎨 UI Design

SegmentIQ shares the Elevvo glassmorphism design system featuring:
- 5 distinct segment color palettes (red, amber, green, blue, violet)
- Dark background with radial gradient ambient glows
- Frosted glass cards and sticky navigation bar
- Segment chips with colored dot indicators
- Interactive Plotly charts with custom dark theme

---

## 🔗 Links

| Resource | URL |
|---|---|
| 🚀 Live Streamlit App | [elevvo-customer-segmentation.streamlit.app](https://elevvo-customer-segmentation.streamlit.app/) |
| 📓 Kaggle Notebook | [kaggle.com/code/abdel2ty/elevvo-customer-segmentation](https://www.kaggle.com/code/abdel2ty/elevvo-customer-segmentation) |

---

## 👤 Author

**@abdel2ty** — Built as part of the Elevvo ML project series.
