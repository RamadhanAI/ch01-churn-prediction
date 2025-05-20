import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.preprocessing import preprocess_data

def train():
    # Load dataset
    df = pd.read_csv("data/bank_churn.csv")

    # Preprocess features and labels
    X, y = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    model.save_model("models/xgb_churn.model")

if __name__ == "__main__":
    train()
