import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Dashboard Title
st.title("🌐 Disease Outbreak Prediction Dashboard")
st.write("This tool uses an **ARIMA(7,1,1)** model to forecast future global case counts.")

# 2. Load the saved model (Make sure you ran the 'joblib.dump' step earlier!)
model = joblib.load('arima_outbreak_model.pkl')

# 3. User Input
days = st.slider("Select the number of days to forecast:", 1, 30, 14)

# 4. Generate the Forecast
forecast = model.forecast(steps=days)

# 5. Visualize
st.subheader(f"Projected Outbreak for the Next {days} Days")
st.line_chart(forecast)