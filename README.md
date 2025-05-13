# Creditcard-fraud-detector
Credit Card Fraud Detection is a machine learning project that aims to detect fraudulent transactions from credit card data. It uses various data preprocessing techniques and classification models to predict the probability of fraud, enabling financial institutions to prevent potential loss.


# 💳 Credit Card Fraud Detection

This project detects fraudulent credit card transactions using machine learning. With an imbalanced dataset containing anonymized transaction features, the model learns to identify potentially fraudulent behavior.

## 📂 Project Files

- `creditcard.ipynb`: Jupyter notebook for data analysis, preprocessing, model training, and evaluation.
- `creditcard copy.csv`: Dataset containing anonymized credit card transactions.
- `app.py`: Flask application that provides an API to predict if a transaction is fraudulent.
- `test.py`: Script to test the API by sending example transaction data.
- `README.md`: Project overview and instructions.

## 📊 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent cases**: 492 (0.172%)
- **Features**:
  - V1 to V28: Principal components from PCA
  - Time: Seconds elapsed between the first and current transaction
  - Amount: Transaction amount
  - Class: Target variable (0 = not fraud, 1 = fraud)

## 🔍 Workflow Summary

1. Load and explore the dataset
2. Handle missing values and class imbalance
3. Train models (e.g., Logistic Regression, Random Forest)
4. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC-AUC
5. Deploy model with a Flask API (`app.py`)
6. Send test requests using `test.py`

## 🚀 Getting Started

### 📦 Prerequisites

Make sure you have Python 3.8+ installed. Then install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask
jupyter notebook creditcard.ipynb
python app.py
python test.py
