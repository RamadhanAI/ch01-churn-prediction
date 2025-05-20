from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd

# Load the model
model = xgb.XGBClassifier()
model.load_model("models/xgb_churn.model")

app = FastAPI(title="Churn Prediction API")

# Define the input schema
class CustomerFeatures(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Mapping categories to integers (basic example)
geography_map = {"France": 0, "Spain": 1, "Germany": 2}
gender_map = {"Female": 0, "Male": 1}

@app.post("/predict")
def predict(features: CustomerFeatures):
    try:
        input_data = pd.DataFrame([{
            "CreditScore": features.CreditScore,
            "Geography": geography_map.get(features.Geography, 0),
            "Gender": gender_map.get(features.Gender, 0),
            "Age": features.Age,
            "Tenure": features.Tenure,
            "Balance": features.Balance,
            "NumOfProducts": features.NumOfProducts,
            "HasCrCard": features.HasCrCard,
            "IsActiveMember": features.IsActiveMember,
            "EstimatedSalary": features.EstimatedSalary
        }])
        prediction = model.predict(input_data)[0]
        return {"churn_prediction": int(prediction)}
    except Exception as e:
        return {"error": str(e)}
