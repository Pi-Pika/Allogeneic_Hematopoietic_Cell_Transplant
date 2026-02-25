# Equity Post-HCT Survival Prediction

Survival Modeling for Allogeneic Hematopoietic Cell Transplant (HCT)

------------------------------------------------------------------------

## 📌 Project Overview

This project builds a machine learning pipeline to predict
post--allogeneic Hematopoietic Cell Transplant (HCT) survival outcomes
using structured clinical data.

The solution integrates:

-   Classical survival analysis (Kaplan--Meier, Cox models)
-   Gradient boosting models
-   Neural networks
-   Random Survival Forest
-   Rank-based ensembling

The objective is to generate accurate survival probability predictions
while properly handling censored survival data.

------------------------------------------------------------------------

## 🧠 Problem Type

This is a **survival analysis** problem:

-   `efs_time` → Time-to-event
-   `efs` → Event indicator (1 = event occurred, 0 = censored)

Traditional regression approaches cannot correctly handle censoring, so
survival-specific modeling techniques are applied.

------------------------------------------------------------------------

## 🗂 Dataset

Files used:

-   train.csv
-   test.csv
-   sample_submission.csv

Target-related columns:

-   `efs_time` -- Time to event or censoring
-   `efs` -- Event indicator
-   `ID` -- Unique identifier

------------------------------------------------------------------------

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

-   Missing value analysis
-   Numerical & categorical feature separation
-   Imputation using `SimpleImputer`
-   Standard scaling for numeric features
-   One-hot encoding for categorical features
-   `ColumnTransformer` preprocessing pipeline

------------------------------------------------------------------------

### 2️⃣ Survival Probability Engineering

Kaplan--Meier estimator is used to compute survival probabilities.

Engineered variables:

-   `y` → survival probability
-   `efs_time_new` → negative time for censored samples (for ranking)

------------------------------------------------------------------------

### 3️⃣ Models Used

**Linear & Classical Models** - Ridge Regression - Cox Proportional
Hazards Model

**Tree-Based Gradient Boosting** - LightGBM - XGBoost - CatBoost

**Survival-Specific Models** - Random Survival Forest -
CoxPHSurvivalAnalysis (sksurv)

**Deep Learning** - Fully connected Neural Network (Keras) - Dense
layers - Batch Normalization - Dropout - Adam optimizer

------------------------------------------------------------------------

### 4️⃣ Cross Validation

-   KFold cross-validation
-   Out-of-fold predictions
-   Rank-based ensembling

------------------------------------------------------------------------

## 📈 Evaluation Metric

**Concordance Index (C-index)**

Measures ranking quality in survival models and properly accounts for
censored observations.

------------------------------------------------------------------------

## 🏗 Project Structure

. ├── v2-of-cibmtr-allogeneic-hct.ipynb\
├── README.md\
└── submission.csv

------------------------------------------------------------------------

## 🚀 How to Run

### Install Dependencies

``` bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost lifelines tensorflow scikit-survival
```

### Run Notebook

Execute all cells in:

    v2-of-cibmtr-allogeneic-hct.ipynb

------------------------------------------------------------------------

## 🧩 Key Features

-   Proper censoring handling
-   Multi-model ensemble strategy
-   Robust rank-based blending
-   Hybrid statistical + ML approach
-   Deep learning integration

------------------------------------------------------------------------

## 📌 Future Improvements

-   Hyperparameter tuning with Optuna
-   SHAP-based feature importance
-   DeepSurv implementation
-   Time-dependent Cox models
-   Stratified survival cross-validation

------------------------------------------------------------------------

## 👤 Author

Mahmudul Hasan Piash\
Student & Engineer\
Machine Learning \| Survival Analysis \| Biomedical AI
