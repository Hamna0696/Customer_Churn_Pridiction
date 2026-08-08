# Customer Churn Prediction

## Project Overview

This project focuses on predicting whether a customer will leave a telecommunications service.

The machine learning pipeline processes customer information, handles missing values, encodes categorical features, scales numerical features, and uses classification models to predict customer churn.

## Dataset

The dataset used in this project is:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information related to demographics, services, account information, and churn status.

## Objective

The main objective is to build a machine learning model that can predict customer churn based on the available customer attributes.

The target variable is:

- `Churn`

Where:

- `No` = Customer did not churn
- `Yes` = Customer churned

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook / Google Colab

## Machine Learning Models

Two classification models were implemented:

1. Logistic Regression
2. Random Forest Classifier

GridSearchCV was also used to find the best Random Forest parameters.

## Project Workflow

### 1. Import Libraries

The project uses Pandas, NumPy, Scikit-learn and Joblib for data processing, machine learning, evaluation and model saving.

### 2. Load Dataset

```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
