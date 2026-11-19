# California Housing Price Prediction

A Jupyter Notebook project that predicts the median house value of California
census block groups. The notebook compares Ordinary Least Squares (OLS), Ridge,
and Lasso regression and evaluates their performance on a held-out test set.

## Overview

The local dataset, `housing.csv`, contains 20,640 observations and 10 columns.
It represents California housing data derived from the 1990 U.S. Census.
The target variable is `median_house_value`, recorded in US dollars.

The project:

- Loads and inspects the housing data
- Creates exploratory plots and correlation analysis
- Examines missing values and extreme observations
- Engineers two additional housing features
- Uses an 80/20 training and test split
- Preprocesses numeric and categorical predictors
- Tunes Ridge and Lasso with five-fold cross-validation
- Compares the models using RMSE, MAE, and R-squared
- Visualizes predicted versus actual values and residual distributions

## Dataset and Features

The source file is kept locally as `housing.csv`.

| Variable | Type | Description |
|---|---|---|
| `longitude` | Numeric | Geographic longitude |
| `latitude` | Numeric | Geographic latitude |
| `housing_median_age` | Numeric | Median age of housing in the block group |
| `total_rooms` | Numeric | Total number of rooms |
| `total_bedrooms` | Numeric | Total number of bedrooms |
| `population` | Numeric | Block-group population |
| `households` | Numeric | Number of households |
| `median_income` | Numeric | Median household income in units of $10,000 |
| `ocean_proximity` | Categorical | Relative location to the ocean |
| `median_house_value` | Numeric target | Median house value in US dollars |

The notebook also creates:

- `rooms_per_household`: `total_rooms / households`
- `income_x_rooms_per_household`: `median_income * rooms_per_household`

The CSV contains 207 missing values in `total_bedrooms`. Under the fixed
`random_state=42` split, all 207 occur in the test set.

## Method

The data is split with 80% for training and 20% for testing:

- Training rows: 16,512
- Test rows: 4,128

Numeric predictors are median-imputed and standardized with `StandardScaler`.
`ocean_proximity` is one-hot encoded with the first category dropped.
Missing bedroom counts are imputed from the training median.

The models tested are:

- Ordinary Least Squares linear regression
- Ridge regression tuned over alpha values of `0.01`, `0.1`, `1`, `10`, `100`, and `1000`
- Lasso regression tuned over alpha values of `0.1`, `1`, `10`, `100`, `1000`, and `10000`

Both regularized models use five-fold shuffled cross-validation with
`random_state=42`. The selected values were alpha `1` for Ridge and alpha `10`
for Lasso.

## Results

The held-out test set contains 4,128 census block groups.

| Model | RMSE (US dollars) | MAE (US dollars) | R-squared |
|---|---:|---:|---:|
| OLS | 69,841.73 | 50,641.87 | 0.6278 |
| Ridge | 69,850.54 | 50,649.14 | 0.6277 |
| Lasso | 69,852.99 | 50,648.00 | 0.6276 |

OLS had the lowest test RMSE, but regularization produced very similar results.
The highest and lowest test RMSE values differed by only $11.26.

The five-fold cross-validation mean RMSE values were $68,485.50 for Ridge and
$68,490.90 for Lasso.

## Project Structure

```text
Task_02_Housing_Price_Prediction/
|-- housing.csv
|-- housing_price_prediction.ipynb
`-- README.md
```

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- pandas
- matplotlib
- seaborn
- scikit-learn

## Setup

Clone the repository and enter this task directory:

```bash
git clone https://github.com/opeayomide320-alt/Datheon_Machine_Learning_and_Data_Science_Internship.git
cd Datheon_Machine_Learning_and_Data_Science_Internship/Task_02_Housing_Price_Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter pandas matplotlib seaborn scikit-learn
```

## Run the Notebook

Launch Jupyter Notebook from the task directory:

```bash
jupyter notebook
```

Open `housing_price_prediction.ipynb`, then select **Run All**. The notebook
can also be opened in JupyterLab or an IDE with Jupyter support.
