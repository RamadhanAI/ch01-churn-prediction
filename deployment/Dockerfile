FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn pandas xgboost scikit-learn

CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]
