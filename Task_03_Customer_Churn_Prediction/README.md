# Predictive Customer Churn Classifier

This Jupyter notebook compares Random Forest and XGBoost for predicting which telecom customers are likely to leave. Both models exceeded the target ROC-AUC of 0.80 on a held-out test set.

## Overview

Of the 7,043 customers in the dataset, 1,869 churned (26.5%). The project evaluates whether giving churners more weight during training improves each model’s ability to rank customers by churn risk.

The notebook uses ROC curves to evaluate that ranking across decision thresholds. It also uses confusion matrices to show how many churners each model detected at a threshold of 0.50 and bar charts to show the ten most important features for each model.

## Dataset and Features

The [Telco Customer Churn dataset](Telco_Customer_Churn.csv) contains one row per customer and 21 columns.

| Columns | Role |
|---|---|
| `customerID` | Unique customer identifier, excluded from the predictors |
| `Churn` | Target, encoded as 1 for `Yes` and 0 for `No` |
| `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` | Four numeric predictors, including the 0/1 senior citizen flag |
| Remaining 15 columns | Categorical predictors covering customer demographics, subscribed services, contract terms, and billing |

Converting `TotalCharges` to numeric revealed 11 missing values. Eight were in the training set, and all eight customers had zero months of tenure. These records were retained.

## Method

1. Split the data into 5,634 training customers and 1,409 test customers. Stratification preserved the 26.5% churn rate in both sets.
2. Use a `ColumnTransformer` to impute missing numeric values with the median, add a missing-value indicator for `TotalCharges`, and encode categorical predictors with `OneHotEncoder(handle_unknown="ignore")`.
3. Include preprocessing in each model’s `Pipeline` so that every cross-validation fold learns its medians and category mappings from its own training data.
4. Use `GridSearchCV` with five shuffled, stratified folds and `random_state=42`. Select the configuration with the highest mean validation ROC-AUC for each classifier.
5. Evaluate the selected models on the held-out customers using predicted churn probabilities.

| Model | Settings compared in cross-validation |
|---|---|
| Random Forest | 200 trees, `min_samples_leaf` of 1, 5, or 10, and `class_weight` set to `None` or `"balanced"` |
| XGBoost | 200 trees, learning rate of 0.05, `max_depth` of 2, 3, or 4, and `scale_pos_weight` of 1.0 or approximately 2.77 |

The XGBoost churn weight of 2.77 came from the ratio of customers who stayed to customers who churned in the training set.

## Results

### Class weighting and model selection

| Model | Class weighting | Best tree setting | Mean validation ROC-AUC | Fold standard deviation |
|---|---|---|---:|---:|
| Random Forest | Balanced | `min_samples_leaf=10` | 0.8467 | 0.0091 |
| Random Forest | Equal | `min_samples_leaf=10` | 0.8473 | 0.0094 |
| XGBoost | Churn weight ×2.77 | `max_depth=2` | 0.8492 | 0.0121 |
| XGBoost | Equal | `max_depth=2` | 0.8498 | 0.0117 |

For both classifiers, equal class weights gave a mean validation ROC-AUC about 0.0006 higher than the weighted alternative. The equal-weight configurations were selected for held-out evaluation.

### Held-out test performance

The test set contained 1,409 customers, including 374 who churned. ROC-AUC uses the predicted probabilities across thresholds. The counts, precision, and recall below use a churn threshold of 0.50.

| Model | Test ROC-AUC | Churners detected | Churners missed | False alarms | Churn precision | Churn recall |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8438 | 187 | 187 | 95 | 66.3% | 50.0% |
| XGBoost | 0.8453 | 188 | 186 | 93 | 66.9% | 50.3% |

XGBoost scored 0.0015 higher on this test split. At the 0.50 threshold, it identified one more churner and raised two fewer false alarms. Both models missed about half of the customers who churned.

### Feature importance

Customer tenure ranked first for Random Forest, followed by total charges. The month-to-month contract indicator ranked first for XGBoost, followed by the fiber-optic internet indicator. The notebook shows the top ten features for each model. Each ranking should be read within its own model because the classifiers calculate importance differently.

## Project Structure

```text
Task_03_Customer_Churn_Prediction/
├── customer_churn_prediction.ipynb
├── Telco_Customer_Churn.csv
└── README.md
```

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- pandas
- matplotlib
- scikit-learn
- xgboost

## Setup

Clone the internship repository and enter the Task 3 folder.

```bash
git clone https://github.com/opeayomide320-alt/Datheon_Machine_Learning_and_Data_Science_Internship.git
cd Datheon_Machine_Learning_and_Data_Science_Internship/Task_03_Customer_Churn_Prediction
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it using the command for your terminal.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

Install the notebook dependencies in the active environment.

```bash
python -m pip install --upgrade pip
python -m pip install jupyter pandas matplotlib scikit-learn xgboost
```

## Run the Notebook

From the Task 3 folder, launch the notebook.

```bash
jupyter notebook customer_churn_prediction.ipynb
```

Select **Run All** to load the CSV, fit both model searches, and regenerate the results table, ROC curves, confusion matrices, and feature importance charts.

In VS Code, open [customer_churn_prediction.ipynb](customer_churn_prediction.ipynb), select the `.venv` Python environment as the notebook kernel, and choose **Run All**. Keep `Telco_Customer_Churn.csv` beside the notebook so its relative file path resolves.