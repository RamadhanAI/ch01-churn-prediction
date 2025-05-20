import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop non-numeric or irrelevant features safely
    for col in ["CustomerId", "Surname", "RowNumber"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Split into features and labels
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    return X, y

