import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load saved pipeline
model_path = Path(__file__).parent / "electric_bill_ac_fan_other_model.pkl"
model = joblib.load(model_path)

# App title
st.title("Electricity Bill Predictor")
st.write("Enter the number of units consumed by Appliances to predict the electricity bill.")

# User input
units_ac = st.number_input(
    "Units consumed by AC",
    min_value=10.0,
    step=1.0
)
units_fan = st.number_input(
    "Units consumed by Fan",
    min_value=20.0,
    step=1.0
)
units_other = st.number_input(
    "Units consumed by Other Appliances",
    min_value=30.0,
    step=1.0
)
# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "AC_Units": [units_ac],"Fan_Units": [units_fan],"Other_Units": [units_other]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Electricity Bill: ₹{prediction:.2f}"
    )