import streamlit as st
import pandas as pd

from pathlib import Path

model_path = Path(__file__).parent / "electric_bill_ac_fan_model.pkl"
model = joblib.load(model_path)

st.title("Electricity Bill Predictor")
st.write("Enter the number of units consumed by AC to predict the result.")

units_consumed = st.number_input("Units consumed by AC", min_value=0.0, step=0.5)

if st.button("Predict"):
	input_data = pd.DataFrame({"AC_Units": [units_consumed]})
	prediction = model.predict(input_data)[0]
	probability = model.predict_proba(input_data)[0][int(prediction)]
	st.success(f"Predicted Electricity Bill : {prediction:.2f}")

	# if prediction == 1:
	# 	st.success(f"Predicted result: Pass ({probability:.1%} confidence)")
	# else:
	# 	st.error(f"Predicted result: Fail ({probability:.1%} confidence)")