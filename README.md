# SegmentIQ — Customer Segmentation

A machine learning web application that segments customers into distinct behavioral groups using K-Means clustering. Built with Streamlit and scikit-learn, featuring an interactive classifier, scenario comparison tools, and a filterable data explorer.

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-8B5CF6?style=flat-square)](https://elevvo-customer-segmentation.streamlit.app/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle_Notebook-abdel2ty-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/code/abdel2ty/elevvo-customer-segmentation)

---

## Overview

SegmentIQ applies **K-Means Clustering (k = 5)** to 200 synthetic customer records, segmenting them by Annual Income and Spending Score. The application provides real-time segment prediction for any new customer profile, scenario simulation tools, an interactive data table with export functionality, and rich cluster analytics to support data-informed marketing and targeting decisions.

---

## Application Pages

**Overview** — Global cluster distribution, Income vs. Spending scatter plot colored by segment, age distribution histogram, and key dataset statistics.

**Classify** — Input a customer's income, spending score, and age to instantly assign them to one of five segments. Results include a color-coded segment card with name, behavioral description, and recommended marketing strategy.

**What-If** — Define and compare multiple customer scenarios simultaneously. A proximity map plots scenario positions against cluster boundaries, making it easy to see how small changes in income or spending shift segment assignment.

**Data** — Filterable data table with controls for segment, gender, and age range. Supports sorting by any column and exports filtered records as a CSV file. Includes summary charts for the current filter state.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | K-Means Clustering |
| Library | scikit-learn |
| Number of Clusters | 5 |
| Training Records | 200 (synthetic, seed = 42) |
| Clustering Features | Annual Income (k$), Spending Score (1–100) |

---

## Customer Segments

| Cluster | Label | Profile |
|---|---|---|
| 0 | Budget Cautious | Low income, low spending |
| 1 | High Earner, Low Spend | High income, conservative spending |
| 2 | Balanced Mainstream | Mid income, mid spending |
| 3 | Impulsive Spender | Low income, high spending |
| 4 | Premium Loyalist | High income, high spending |

---

## Input Features

| Feature | Range | Type |
|---|---|---|
| Annual Income | 15k – 137k USD | Numeric |
| Spending Score | 1 – 100 | Numeric |
| Age | 18 – 80 | Numeric |
| Gender | Male / Female | Categorical |

---

## Tech Stack

| Component | Technology |
|---|---|
| App Framework | Streamlit |
| ML Model | scikit-learn — KMeans, StandardScaler |
| Visualizations | Plotly |
| Data Handling | pandas, numpy |

---

## Local Setup

```bash
git clone <your-repo-url>
cd segmentiq

pip install streamlit numpy pandas plotly scikit-learn

streamlit run app2.py
```

---

## Links

| | |
|---|---|
| Live Application | https://elevvo-customer-segmentation.streamlit.app/ |
| Kaggle Notebook | https://www.kaggle.com/code/abdel2ty/elevvo-customer-segmentation |

---

*Built by @abdel2ty as part of the Elevvo ML project series.*
