# Telco Customer Churn — ML + A/B Testing

> **Course project:** Statistical Methods for Machine Learning  
> **Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Research Question

**Does targeting high-risk customers using machine learning reduce churn more effectively than a random retention strategy?**

This project combines predictive modeling with a simulated A/B experiment to answer this question end-to-end.

---

## Project Structure

```
telco-churn-ab-testing/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Download from Kaggle (see below)
├── R/
│   ├── 01_eda.R            ← Exploratory data analysis & visualizations
│   ├── 02_preprocessing.R  ← Cleaning, encoding, scaling, train/test split
│   ├── 03_models.R         ← Logistic Regression, Random Forest, SVM
│   └── 04_ab_testing.R     ← A/B simulation + statistical tests
├── plots/                  ← All output plots (auto-generated)
├── .gitignore
├── telco_churn.Rproj
└── README.md
```

---

## Methodology

### Machine Learning Pipeline

| Step | Details |
|---|---|
| **EDA** | Churn by contract, tenure, internet service, payment method |
| **Preprocessing** | One-hot encoding, z-score normalization (fit on train only) |
| **Models** | Logistic Regression, Random Forest (tuned `mtry`), SVM (RBF kernel) |
| **Evaluation** | AUC-ROC, F1, Precision, Recall, Brier Score via 5-fold CV |

### A/B Testing Design

| | Group A (ML-Targeted) | Group B (Random) |
|---|---|---|
| **Selection** | Top N% by predicted churn probability | Random sample of N% |
| **Hypothesis** | H₁: p_A > p_B | H₀: p_A = p_B |
| **Tests** | Two-proportion Z-test, Chi-square, Fisher's Exact | — |
| **Effect sizes** | Relative Risk, Absolute Risk Difference, NNT, Lift | — |
| **CI** | Bootstrap 95% CI on ARD (2,000 resamples) | — |

The test set (held out from all training) serves as the simulated customer pool. The model ranks customers by churn risk; actual churn labels provide the observed outcomes.

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ChrisACr/telco-churn-ab-testing.git
cd telco-churn-ab-testing
```

### 2. Download the dataset

Go to [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and download `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Place it in the `data/` folder.

### 3. Open in RStudio

Open `telco_churn.Rproj` in RStudio. This sets the working directory correctly.

### 4. Run the scripts in order

```r
source("R/01_eda.R")
source("R/02_preprocessing.R")
source("R/03_models.R")
source("R/04_ab_testing.R")
```

Each script installs missing packages automatically.

### Required R packages

```r
install.packages(c(
  "tidyverse", "ggplot2", "scales", "gridExtra", "corrplot",
  "RColorBrewer", "patchwork", "caret", "randomForest",
  "e1071", "pROC", "boot"
))
```

---

## Key Results

Results are printed to the console after running each script. All plots are saved to `plots/` and are referenced in the accompanying presentation.

Highlights:
- **Best model:** Random Forest (AUC ≈ 0.83)
- **A/B finding:** ML targeting identifies significantly more true churners per customer contacted vs. random selection
- **Lift at 30% depth:** ~2× over random baseline

---

## Dataset

- **Source:** IBM via Kaggle (CC0: Public Domain)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` (Yes / No) — overall rate ≈ 26.5%
- **Key features:** Contract type, tenure, monthly charges, internet service, payment method

---

## License

MIT — free to use, modify, and distribute with attribution.
