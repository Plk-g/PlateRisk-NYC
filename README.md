# 🍽️ PlateRisk
### ML-Powered Inspection Prioritization for NYC Restaurants

> Predicting critical food safety violations across NYC's 27,000+ restaurants — helping DOHMH allocate limited inspection resources where they matter most.

**Course:** TECH-GB.2336 — Data Science & AI for Business | NYU Stern, Spring 2026  
**Instructor:** Prof. Foster Provost  
**Team:** Palak Gupta · Tiffany Chiang · Anna Gellerman  
**Live Report:** [platerisk-nyc GitHub Pages](https://Plk-g.github.io/platerisk-nyc) 

---

## 📌 Project Overview

NYC's Department of Health and Mental Hygiene (DOHMH) conducts approximately **100,000 restaurant inspections per year** across roughly 27,000 active establishments — under a fixed weekly inspection capacity. Currently, scheduling decisions do not systematically account for the risk profile of each restaurant, meaning high-risk establishments may go undetected longer than necessary.

**PlateRisk** is a supervised machine learning system that predicts the probability of a critical violation at a restaurant's *next* inspection, enabling DOHMH analysts to generate a data-driven prioritized inspection queue each week. Given a fixed capacity K, the top-K restaurants ranked by predicted risk score are flagged for inspection — replacing ad hoc, time-based scheduling with an evidence-based prioritization policy.

This project is framed as a **proof of concept study**, following the full CRISP-DM data science process: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.

---

## 📊 Key Results (Preliminary — Status Report Stage)

| Model | Precision@K (K=500) | ROC-AUC |
|---|---|---|
| Random Baseline | 0.848 | — |
| Time-Since-Last (Current Policy) | 0.874 | 0.564 |
| Logistic Regression | 0.820 | 0.545 |
| **Random Forest (PlateRisk)** | **0.906** | **0.599** |
| **Gradient Boosting (PlateRisk)** | **0.906** | **0.595** |

> ⚠️ **Note:** Results are preliminary. A critical flag parsing issue is currently under investigation — final numbers will be updated in the full writeup after the fix is validated.

**Expected Value:** Under the Random Forest model, DOHMH catches an estimated **+16 additional critical violations per week** (+832/year) vs. the current time-based scheduling policy — at no additional inspection cost.

---

## 🏗️ Repository Structure

```
platerisk-nyc/
│
├── README.md                        ← you are here
│
├── notebook/
│   └── PlateRisk_Model.ipynb        ← full pipeline: data → features → models → evaluation
│
├── data/
│   └── README.md                    ← data source documentation (no raw data stored)
│
├── outputs/
│   ├── roc_curves.png               ← ROC curves for all models
│   ├── lift_curves.png              ← cumulative lift curves
│   ├── feature_importance.png       ← Random Forest feature importance
│   ├── fairness_borough.png         ← borough disparate impact analysis
│   └── expected_value.png           ← business value framing
│
└── report/
    └── index.html                   ← live stakeholder report (see GitHub Pages)
```

---

## 📁 Data Source

Data is pulled **live** from the NYC Open Data API — no raw files are stored in this repository.

```python
url = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv?$limit=500000"
df_raw = pd.read_csv(url)
```

**Dataset:** DOHMH New York City Restaurant Inspection Results  
**Source:** [NYC Open Data](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)  
**Size:** ~83,000 inspection-level records across 27,000+ establishments  
**Date range:** 2007 — present (live, updated daily)

---

## 🔬 Methodology

### Problem Formulation
This is a **supervised binary classification** task. For each restaurant inspection, the model predicts the probability that the restaurant's *next* inspection will result in a critical violation. A risk score is output per establishment, used to rank and select the top-K for the weekly inspection queue.

### Features
All features are constructed from pre-inspection information to prevent data leakage:

| Feature | Description |
|---|---|
| `days_since_last` | Days elapsed since previous inspection |
| `prev_score` | Inspection score from previous visit |
| `prev_critical` | Whether previous inspection had a critical violation |
| `total_past_critical` | Cumulative count of historical critical violations |
| `inspection_month` | Month of inspection (seasonality) |
| `inspection_dow` | Day of week (temporal patterns) |
| `boro_*` | One-hot encoded borough (5 NYC boroughs) |
| `cuisine_*` | One-hot encoded cuisine type (top 15 + Other) |

### Train/Test Split
An **80/20 time-based split** is used — training on historical data, testing on the most recent 20% of inspections. This simulates real deployment (train on past, predict future) and prevents leakage from random splitting.

### Models
- **Logistic Regression** — interpretable baseline
- **Random Forest** — primary model (handles non-linearity, provides feature importance)
- **Gradient Boosting** — stretch model (sequential error correction)

### Evaluation Metrics
- **Precision@K** — primary metric: fraction of top-K ranked restaurants that actually had critical violations
- **ROC-AUC** — secondary metric: overall discriminative power
- **Lift Curves** — visual comparison of violations caught per inspection slot
- **Fairness Analysis** — borough and cuisine disparate impact checks

---

## ⚖️ Ethical Considerations

PlateRisk is designed for government resource allocation — a context where fairness and transparency are critical. Three risks are explicitly monitored:

1. **Geographic disparate impact** — borough-level fairness analysis shows whether the model systematically over- or under-targets certain boroughs relative to their restaurant population share
2. **Feedback loop reinforcement** — frequently inspected areas accumulate more violation labels, which can amplify model targeting of those areas
3. **Transparency and due process** — restaurant owners affected by algorithmic prioritization have no visibility into why they were selected; this must be addressed in any real deployment

---

## 🚀 How to Run

**Requirements:**
```
pandas, numpy, scikit-learn, matplotlib, seaborn
```

**Install:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Run:**  
Open `notebook/PlateRisk_Model.ipynb` in Google Colab or Jupyter. The notebook pulls data live from the NYC Open Data API — no local data file needed.

---

## 👥 Team Contributions

### Palak Gupta
- Random Forest and Gradient Boosting model development
- Full model comparison and results table
- ROC curve and cumulative lift curve visualizations
- Feature importance analysis
- Fairness / disparate impact analysis (borough + cuisine)
- Expected value and ROI framing
- **GitHub repository architecture and documentation**
- **Live stakeholder report (`report/index.html`) — interactive HTML report with progress tracker, business proposal, results dashboard, and team overview**
- **Business framing and stakeholder narrative** — structured the project around the CRISP-DM process and Prof. Provost's proof-of-concept framework
- **Known issues log and pre-final checklist** for team coordination

### Tiffany Chiang
- Data acquisition via NYC Open Data API
- Data cleaning and column standardization
- Inspection-level aggregation (one row per inspection event)
- Target variable construction (`shift(-1)` for next-inspection labeling)
- Data validation and verification prints

### Anna Gellerman
- Feature engineering (temporal, historical risk, cumulative violation features)
- Categorical encoding (borough and cuisine one-hot encoding)
- Time-based train/test split implementation
- Heuristic and logistic regression baselines
- Precision@K and ROC-AUC baseline evaluation

---

## 📅 Project Timeline

| Deliverable | Due | Status |
|---|---|---|
| Team formation + initial ideas | Feb 21 | ✅ Complete |
| Project proposal | Mar 7 | ✅ Complete |
| Status report (this stage) | Apr 11 | ✅ Complete |
| Final writeup | TBD | 🔄 In Progress |

---

## 🔗 Links

- 📓 [Notebook (Google Colab)](https://colab.research.google.com/drive/16_OsJSPxgqnhhR4YCryfKZSMBQj-Irep)
- 🌐 [Live Report](https://yourusername.github.io/platerisk-nyc)
- 📊 [NYC Open Data Source](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j)

---

*NYU Stern School of Business · TECH-GB.2336 Data Science & AI for Business · Spring 2026*
