import streamlit as st
import pandas as pd
import mlflow.pyfunc
import numpy as np
import joblib

model_name = "Insurance Premium Calculator - XGBoost regressor model"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"
model = joblib.load("model.pkl")

st.title("Insurance Premium Predictor")

st.write("Please enter customer details to predict their insurance premium:")

with st.form("customer_form"):
    id = st.number_input("Id", min_value=1200000, max_value=2000000)
    
    age_missing = st.checkbox("Age is missing")
    if age_missing:
        age = np.nan
    else:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    gender = st.selectbox("Gender", ["Male", "Female"])
    
    income_missing = st.checkbox("Annual Income is missing")
    if income_missing:
        income = np.nan
    else:
        income = st.number_input("Annual Income", min_value=0, value=50000)

    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Missing"])
    
    dependents_missing = st.checkbox("Number of Dependents is missing")
    if dependents_missing:
        dependents = np.nan
    else:
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Unemployed", "Self-Employed", "Missing"])

    health_missing = st.checkbox("Health Score is missing")
    if health_missing:
        health_score = np.nan
    else:
        health_score = st.number_input("Health Score", min_value=0.0, max_value=100.0, value=50.0)

    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])

    claims_missing = st.checkbox("Previous Claims is missing")
    if claims_missing:
        previous_claims = np.nan
    else:
        previous_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=0)

    vehicle_age_missing = st.checkbox("Vehicle Age is missing")
    if vehicle_age_missing:
        vehicle_age = np.nan
    else:
        vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=50, value=5)

    credit_score_missing = st.checkbox("Credit Score is missing")
    if credit_score_missing:
        credit_score = np.nan
    else:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

    duration_missing = st.checkbox("Insurance Duration is missing")
    if duration_missing:
        insurance_duration = np.nan
    else:
        insurance_duration = st.number_input("Insurance Duration", min_value=0, max_value=20, value=2)

    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good", "Missing"])
    smoking_status = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise_freq = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    submit = st.form_submit_button("Predict Premium")
    
if submit:
    input_data = pd.DataFrame({
        "Id": [id],
        "Age": [age],
        "Gender": [gender],
        "Annual Income": [income],
        "Marital Status": [marital_status],
        "Number of Dependents": [dependents],
        "Education Level": [education],
        "Occupation": [occupation],
        "Health Score": [health_score],
        "Location": [location],
        "Policy Type": [policy_type],
        "Previous Claims": [previous_claims],
        "Vehicle Age": [vehicle_age],
        "Credit Score": [credit_score],
        "Insurance Duration": [insurance_duration],
        "Customer Feedback": [feedback],
        "Smoking Status": [smoking_status],
        "Exercise Frequency": [exercise_freq],
        "Property Type": [property_type]
    })

    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Insurance Premium: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")