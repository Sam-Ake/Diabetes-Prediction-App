import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="End-to-end Machine Learning API for predicting diabetes risk.",
    version="1.0",
)

# Load saved artifacts
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# Define Pydantic request schema matching our input features
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.get("/")
def home():
    return {"message": "Welcome to the Diabetes Prediction API! Use /docs to test endpoints."}


@app.post("/predict_diabetes")
def predict_diabetes(data: DiabetesInput):
    # 1. Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # 2. Perform same feature engineering as training (BMI_Age_Interaction)
    input_data["BMI_Age_Interaction"] = (
        input_data["BMI"] * input_data["Age"]
    )

    # 3. Ensure columns match training order
    input_data = input_data[feature_columns]

    # 4. Make prediction and get probability
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    # 5. Format response
    result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"

    return {
        "prediction_code": prediction,
        "result": result,
        "probability": round(probability * 100, 2),
    }