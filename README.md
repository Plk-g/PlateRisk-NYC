# PlateRisk 🍽️

> ML-powered restaurant inspection risk scoring for NYC's Department of Health and Mental Hygiene (DOHMH)

**NYU Stern School of Business · TECH-GB.2336 Data Science & AI for Business**
Prof. Foster Provost & Baotong Zhang · Spring 2026

**[Live Report →](https://plk-g.github.io/PlateRisk-NYC/)**

---

## Overview

NYC has 27,000+ active restaurants and a fixed weekly inspection budget of ~500 inspections. The current DOHMH scheduling heuristic (time-since-last-inspection) ignores risk signal entirely. PlateRisk replaces it with a machine learning risk score — ranking restaurants by predicted probability of a critical violation so inspectors focus effort where it matters most.

This is a proof-of-concept study structured around the CRISP-DM data science process, submitted as the term project for TECH-GB.2336.

---

## The Prediction Task

Given a restaurant's historical inspection profile, cuisine type, borough, and timing patterns — predict whether an inspection will result in a critical violation.

- **Task type:** Supervised binary classification
- **Target:** `any_critical_violation` on the current inspection (1 = critical, 0 = not critical)
- **Split:** `GroupShuffleSplit` on restaurant ID (80/20) — zero overlap between train and test
- **Primary metric:** Precision@K (K=500, the weekly inspection budget)
- **Secondary metric:** ROC-AUC

---

## Results

| Model | Precision@K (K=500) | ROC-AUC | vs Baseline |
|---|---|---|---|
| Random (lower bound) | 88.2% | 0.507 | — |
| Time-Since-Last *(current DOHMH policy)* | 89.2% | 0.483 | +1.0pp |
| Logistic Regression | 93.2% | 0.647 | +5.0pp |
| **Random Forest** ★ | **95.0%** | **0.685** | **+6.8pp** |
| Gradient Boosting | 92.0% | 0.699 | +3.8pp |

Random Forest is the primary model (best Precision@K). Gradient Boosting achieves the best ROC-AUC at 0.699 — correctly ranking a high-risk restaurant above a low-risk one ~70% of the time vs 50% for random chance.

---

## Key Modeling Decisions

**Target reformulation:** The original target (next inspection's critical flag) caused near-zero positive rates in the test set due to post-inspection remediation — restaurants that received a violation typically fixed it before the next cycle. Reformulated to predict the current inspection's critical flag, with a restaurant-based split per Prof. Provost's guidance.

**Train/test split:** Switched from temporal (80/20 by date) to `GroupShuffleSplit` on `camis` (restaurant ID). This simulates deployment to previously unseen establishments and prevents the remediation feedback loop from collapsing the test set.

**Leakage guard:** All historical features use `shift(1)` — they reflect only information available before the current inspection, never the current row's violation outcome.

---

## Repository Structure
PlateRisk-NYC/
├── index.html                  # Live stakeholder report (GitHub Pages)
├── notebook/
│   └── PlateRisk_Model.ipynb   # Full modeling pipeline
├── Outputs/
│   ├── roc_curves.png
│   ├── lift_curves.png
│   ├── feature_importance.png
│   ├── expected_value.png
│   └── fairness_borough.png
├── images/
│   ├── tiffany.jpg
│   ├── anna.jpg
│   └── palak.jpg
├── data/
│   └── README.MD               # Data source info (data not committed)
└── README.md

---

## Data Source

NYC Open Data — DOHMH New York City Restaurant Inspection Results
https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j

- ~500,000 rows loaded via Socrata API
- Each row = one violation found during one inspection visit
- Aggregated to inspection level: ~83,480 records across 27,000+ unique restaurants
- Filtered to initial cycle inspections only (re-inspections excluded)

**Note on base rate:** The dataset shows ~87% critical violation rate in the inspected sample vs 20–25% in DOHMH annual reports. This is explained by selection bias — only previously inspected, higher-risk establishments appear in the dataset with labels. The model uses `class_weight='balanced'` and is evaluated on Precision@K and ROC-AUC, not accuracy.

---

## Features Used

| Feature | Description |
|---|---|
| `prev_score` | Inspection score from the prior visit (shift(1)) |
| `prev_critical` | Critical flag from the prior visit (shift(1)) |
| `total_past_critical` | Cumulative prior critical violations (shift(1).cumsum()) |
| `days_since_last` | Days since last inspection (filled 365 for new restaurants) |
| `num_violations` | Number of violation rows at the current inspection |
| `inspection_month` | Month of inspection |
| `inspection_dow` | Day of week of inspection |
| `boro_*` | Borough dummy variables (4) |
| `cuisine_*` | Top-15 cuisine type dummies |

---

## Ethical Considerations

- **Feedback loop:** Frequently inspected restaurants accumulate more labels, reinforcing model targeting. Mitigated by reserving ~10% of weekly slots for non-model-selected restaurants.
- **Geographic disparate impact:** Manhattan is systematically under-represented in top-K outputs. Borough-proportional floors and regular fairness auditing recommended for any live deployment.
- **Transparency:** Restaurant operators have no visibility into why they were selected. Feature importance outputs from the Random Forest provide a plain-language explanation layer for inspectors.

---

## Team

| Name | Role |
|---|---|
| Tiffany Chiang | Data acquisition, cleaning, aggregation, target definition |
| Anna Gellerman | Feature engineering, baselines, train/test split, evaluation framework |
| Palak Gupta | Modeling, evaluation, fairness analysis, report, business framing |

---

## References

1. City of Chicago (2016). *Food Inspection Forecasting.* GitHub: chicago/food-inspections-evaluation
2. NYC DOHMH. *Restaurant Inspection Results.* NYC Open Data
3. Provost, F. & Fawcett, T. (2013). *Data Science for Business.* O'Reilly Media
4. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825–2830
5. Breiman, L. (2001). *Random Forests.* Machine Learning 45(1), pp. 5–32
