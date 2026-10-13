import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    page_icon="🩺",
    layout="centered",
)

# Load Saved Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

# App Header
st.title("🩺 Pima Indians Diabetes Risk Predictor")
st.markdown(
    "This application uses a machine learning model trained on the Pima Indians Diabetes Dataset "
    "to assess diabetes risk based on physiological and clinical measurements."
)

st.divider()

# Input Form Layout
st.subheader("Enter Patient Medical Details:")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0
    )

with col2:
    insulin = st.number_input(
        "Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0
    )
    bmi = st.number_input(
        "BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

st.divider()

# Prediction Button
if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    # 1. Construct input DataFrame
    input_data = pd.DataFrame(
        {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age],
        }
    )

    # 2. Apply feature engineering (BMI * Age interaction)
    input_data["BMI_Age_Interaction"] = (
        input_data["BMI"] * input_data["Age"]
    )

    # 3. Align columns with training order
    input_data = input_data[feature_columns]

    # 4. Predict
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1]) * 100

    # 5. Display Results
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.error(
            f"**Result:** Positive for Diabetes Risk\n\n"
            f"**Confidence / Probability:** {probability:.2f}%"
        )
    else:
        st.success(
            f"**Result:** Negative for Diabetes Risk\n\n"
            f"**Confidence / Probability:** {100 - probability:.2f}% (Negative likelihood)"
        )