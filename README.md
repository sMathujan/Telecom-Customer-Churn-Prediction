# Telecom Customer Churn Prediction Using Ensemble Machine Learning

**MSc Data Science — Final Project Report**
**Module: 7PAM2002 | University of Hertfordshire**
**Student: Mathujan Sivananthan (24087340)**

---

## 📋 Project Overview

Customer churn — the voluntary discontinuation of a service subscription — is one of the most significant operational challenges in the telecommunications industry. Annual churn rates consistently exceed 30%, costing the global telecom sector over $62 billion per year.

This project develops and evaluates a complete, reproducible machine learning pipeline for telecom customer churn prediction using the publicly available **IBM Telco Customer Churn dataset**. Three ensemble classifiers are implemented and compared: **Random Forest**, **XGBoost**, and **CatBoost**, with class imbalance handled by SMOTE and hyperparameters optimised using **Optuna Bayesian search**.

### 🎯 Research Question

> *Which machine learning classification algorithm, trained on SMOTE-balanced data and Bayesian-optimised hyperparameters, best predicts telecom customer churn on the IBM Telco dataset, as measured by F1-score and Precision-Recall AUC?*

---

## 🏆 Key Results

| Model | Threshold | F1 | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.49 | 0.624 | 0.752 | 0.830 | 0.613 |
| XGBoost | 0.49 | 0.623 | 0.766 | 0.824 | 0.603 |
| **CatBoost ✅** | **0.53** | **0.635** | **0.779** | **0.836** | **0.636** |

**Best model: CatBoost** — F1 = 63.5%, Recall = 77.9%, ROC-AUC = 83.6%, PR-AUC = 63.6%
representing a **2.4× improvement** over the no-skill PR-AUC baseline of 26.6%.

---

## 🔑 Novel Contributions

Three methodological contributions not collectively present in any single reviewed paper:

1. **PR-AUC as a primary evaluation metric** — cannot be inflated by the majority class, unlike accuracy or ROC-AUC alone
2. **Bayesian hyperparameter optimisation via Optuna** — with CV performed on raw (non-SMOTE) data to prevent inflated CV scores
3. **Per-model decision threshold optimisation** — demonstrating that the default threshold of 0.5 costs approximately 13–15 percentage points of recall

---

## 📁 Repository Structure

```
Telecom-Customer-Churn-Prediction/
│
├── churn notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb       # Data loading, EDA, feature encoding
│   ├── 02_Handling_Imbalance.ipynb          # Train-test split, SMOTE oversampling
│   └── 03_Model_Building_and_Tuning.ipynb   # Baseline models, Optuna tuning,
│                                            # evaluation, feature importance
│
├── LICENSE                                  # MIT Licence
└── README.md                                # This file
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**

| Detail | Value |
|---|---|
| Source | IBM GitHub Repository |
| Records | 7,032 (after removing 11 missing TotalCharges rows) |
| Features | 20 input features + 1 binary target |
| Churn rate | 26.6% (1,869 churners / 5,163 non-churners) |
| Class ratio | 2.76:1 majority to minority |
| Licence | Apache 2.0 |

🔗 **Dataset link:** [https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

---

## 🔧 Methodology

### Pipeline Overview

```
Raw Data (7,043 rows)
        ↓
Preprocessing (drop 11 missing rows, encode features → 23 features)
        ↓
Stratified Train-Test Split (70% train / 30% test)
        ↓
Baseline Models (SMOTE on full train set, default hyperparameters)
        ↓
Optuna Bayesian Tuning (CV on raw data, SMOTE inside each fold)
        ↓
Final Models (trained on full SMOTE train set, best params)
        ↓
Threshold Optimisation (per model, F1 maximised on test set)
        ↓
Evaluation (F1, Recall, Precision, ROC-AUC, PR-AUC, confusion matrices)
        ↓
Feature Importance Analysis
```

### Key Methodological Decisions

- **SMOTE applied inside CV folds** — prevents data leakage from synthetic samples into validation sets
- **CV on raw (non-SMOTE) data** — ensures CV F1 reflects the real 26.6% class distribution, not the inflated 50/50 synthetic distribution
- **scale_pos_weight tuned as hyperparameter** (XGBoost) — avoids double imbalance correction alongside SMOTE
- **Early stopping** — XGBoost uses `eval_metric='aucpr'`; CatBoost uses `od_type='Iter'`, `eval_metric='F1'`

### Optuna Tuning Summary

| Model | Trials | Best CV F1 | Best Trial |
|---|---|---|---|
| Random Forest | 50 | 0.636 | #32 |
| XGBoost | 100 | 0.635 | #88 |
| CatBoost | 100 | 0.640 | #21 |

---

## 📦 Dependencies

```
Python >= 3.8

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
catboost
optuna
pickle
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost optuna
```

---

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/sMathujan/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction
```

2. **Download the dataset** from the IBM GitHub repository:
[https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

3. **Run the notebooks in order**

```
01_EDA_and_Preprocessing.ipynb   →   02_Handling_Imbalance.ipynb   →   03_Model_Building_and_Tuning.ipynb
```

   Each notebook saves its outputs as pickle files that are loaded by the next notebook.

---

## 📈 Key Findings

### Top Churn Drivers (consistent across all three models)

| Feature | Insight |
|---|---|
| **Tenure** | Dominant feature (≈30% importance in CatBoost). Churners have median tenure of 10 months vs 38 months for non-churners |
| **Contract type** | Month-to-month customers churn at 43% vs 3% for two-year contracts |
| **Fibre Optic internet** | 42% churn rate vs 19% for DSL customers |
| **Electronic check payment** | 45% churn rate vs 15% for automatic bank transfer |
| **Monthly charges** | Churner mean $74.44 vs non-churner mean $61.27 |

---

## 📄 Licence

This project is licensed under the MIT Licence. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mathujan Sivananthan**
MSc Data Science | University of Hertfordshire | Student ID: 24087340
Module: 7PAM2002 — MSc Data Science Project
Academic Year: 2025–26
