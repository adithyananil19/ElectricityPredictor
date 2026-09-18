import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load saved pipeline
model_path = Path(__file__).parent / "electric_bill_ac_model.pkl"
model = joblib.load(model_path)

# App title
st.title("Electricity Bill Predictor")
st.write("Enter the number of units consumed by AC to predict the electricity bill.")

# User input
units_consumed = st.number_input(
    "Units consumed by AC",
    min_value=1.0,
    step=1.0
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "AC_Units": [units_consumed]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Electricity Bill: ₹{prediction:.2f}"
    )