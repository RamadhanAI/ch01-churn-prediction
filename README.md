# Chapter 1: Churn Prediction in Banking

This project contains the source code, data, and deployment templates for building a churn prediction model in the banking sector.

## Structure

- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks with EDA and modeling
- `src/`: Core Python scripts for preprocessing and training
- `models/`: Saved models
- `deployment/`: FastAPI app, Dockerfile, CI/CD config
- `config/`: YAML or JSON configs for model and environment settings

## Objective

Predict whether a customer is likely to churn using features such as tenure, balance, complaints, and transaction frequency.

## Tools

- Python, Pandas, Scikit-learn, XGBoost
- FastAPI for serving the model
- Docker and GitHub Actions for deployment

# 📊 Churn Prediction in Production – Chapter 1 Repository


## 🚀 Project Overview

**Objective**: Predict which customers are likely to churn using structured banking data.

**Key Components:**
- Data preprocessing and feature engineering
- Model training using `RandomForestClassifier` and `XGBoost`
- Evaluation with confusion matrix, precision, recall, F1-score, and ROC AUC
- FastAPI-based model deployment
- Docker containerization



 Getting Started

1. Clone the repository
git clone https://github.com/RamadhanAI/ch01-churn-prediction.git
cd ch01-churn-prediction
2. Install dependencies
pip install -r requirements.txt
3. Train the model
python notebooks/train_model.py
Output: models/xgb_churn_model.joblib

FastAPI Inference API

Run locally:
uvicorn app.main:app --reload
Sample request:
POST /predict
{
  "feature1": 1.23,
  "feature2": 0.77
}
📈 Model Evaluation Metrics

Precision, Recall, F1-Score
Confusion Matrix (visualized)
ROC AUC Score
Docker Support

Build image:
docker build -t churn-api .
Run container:
docker run -p 8000:8000 churn-api
📚 Reference

This project is based on Chapter 1 from the book
Applied AI and MLOps: From Idea to Deployment
