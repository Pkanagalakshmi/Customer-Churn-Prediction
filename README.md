

---

# Customer Churn Prediction Using XGBoost

This project aims to predict customer churn using a machine learning model based on the XGBoost algorithm. The dataset used in this project contains various features related to customer behavior and demographics, which are used to predict whether a customer will churn or not.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [License](#license)

## Project Overview

Customer churn is a critical issue for companies, especially in highly competitive industries. Predicting churn allows businesses to take proactive measures to retain customers. In this project, an XGBoost model is developed to predict churn based on customer data.

## Dataset

The dataset used in this project contains 3333 entries and 20 features, including customer account information, behavior on different platforms, transaction details, and more. The target variable is `churn`, which indicates whether a customer has churned (1) or not (0).

## Dependencies

The following Python libraries are required to run the project:

- pandas
- numpy
- xgboost
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install pandas numpy xgboost scikit-learn
```

## Data Preprocessing

The dataset undergoes several preprocessing steps:

1. **Handling Categorical Variables**: Categorical columns like `credit card info save` and `push status` are converted to binary values (1 or 0).
2. **Handling Numerical Strings**: Columns with numerical strings containing commas are converted to float values.
3. **Dummy Variables**: The `location code` column is handled using one-hot encoding.
4. **Normalization**: Selected features are normalized using the `Normalizer` from scikit-learn.
5. **Splitting the Data**: The data is split into training and testing sets with a 67-33 ratio.

## Model Training

An initial XGBoost model is trained on the training data. The model achieves an accuracy of 92% on the test dataset.

## Hyperparameter Tuning

Hyperparameter tuning is performed using `GridSearchCV` to optimize the model parameters, including `max_depth`, `learning_rate`, `gamma`, `scale_pos_weight`, and more. The best parameters are:

- `max_depth`: 5
- `learning_rate`: 0.1
- `gamma`: 1
- `scale_pos_weight`: 2
- `subsample`: 1
- `colsample_bytree`: 1

With these tuned parameters, the final model is retrained, achieving an accuracy of 92.54% on the test dataset.

## Results

The final model accuracy on the test dataset is **92.54%**. This indicates a high level of precision in predicting customer churn, making the model valuable for business applications.

## Conclusion

The XGBoost model provides a robust solution for predicting customer churn with high accuracy. The model can help businesses identify at-risk customers and take necessary actions to retain them, ultimately improving customer satisfaction and profitability.

## Usage

To run the project, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Load the dataset and execute the preprocessing steps.
4. Train the model using the provided code.
5. Evaluate the model performance.

