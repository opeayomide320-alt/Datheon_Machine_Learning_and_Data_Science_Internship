# Datheon Machine Learning & Data Science Internship

This repository contains the completed tasks for the Datheon Machine Learning
and Data Science internship.

## Completed Tasks

| Task | Folder | Status |
|---|---|---|
| Iris Species Classification | [Iris-Classification](Iris-Classification/) | Completed |
| California Housing Price Prediction | [Task 2 folder](Task_02_Housing_Price_Prediction/) | Completed |
| Predictive Customer Churn Classifier | [Task 3 folder](Task_03_Customer_Churn_Prediction/) | Completed |
| Task 4 | - | Pending |

## Task 1: Iris Species Classification

This project predicts iris flower species from sepal and petal measurements.
It compares Logistic Regression and Decision Tree classifiers and includes:

- [Jupyter notebook](Iris-Classification/iris_classification.ipynb)
- [Iris dataset](Iris-Classification/iris.csv)
- [Task documentation](Iris-Classification/README.md)

## Task 2: California Housing Price Prediction

This project compares OLS, Ridge, and Lasso regression on the California Housing dataset. On the 4,128 held-out census block groups, OLS had the lowest RMSE ($69,841.73). All three models had similar results, with only $11.26 between the lowest and highest RMSE. The notebook includes predicted-versus-actual plots and residual distributions for each model.

- [Housing prediction notebook](Task_02_Housing_Price_Prediction/housing_price_prediction.ipynb)
- [Housing dataset](Task_02_Housing_Price_Prediction/housing.csv)
- [Task documentation](Task_02_Housing_Price_Prediction/README.md)

## Task 3: Predictive Customer Churn Classifier

This project compares Random Forest and XGBoost for predicting telecom customer churn. On 1,409 held-out customers, Random Forest achieved a ROC-AUC of 0.8438 and XGBoost achieved 0.8453. Both exceeded the 0.80 target. The notebook includes ROC curves, confusion matrices, and the ten most important features for each model.

- [Customer churn prediction notebook](Task_03_Customer_Churn_Prediction/customer_churn_prediction.ipynb)
- [Telco customer churn dataset](Task_03_Customer_Churn_Prediction/Telco_Customer_Churn.csv)
- [Task documentation](Task_03_Customer_Churn_Prediction/README.md)