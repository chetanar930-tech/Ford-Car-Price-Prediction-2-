# Import Streamlit for web app
import streamlit as st

# Import pandas for DataFrame
import pandas as pd

# Import joblib to load saved model
import joblib

# Load trained model
model = joblib.load("LR_model.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load encoded columns
encoded_columns = joblib.load("columns.pkl")

# Configure Streamlit page
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

st.title("Ford Car Price Predictor")

st.write("Enter the car details below to predict its selling price.")

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    value=2020
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=10000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=55.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.5,
    value=1.5
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic","Manual","Semi-Auto"]
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

# Take car model name from user
model_name = st.text_input("Car Model")

# Create Predict Price button
predict = st.button("Predict Price")

if predict:

    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    # One-Hot Encoding
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(columns=encoded_columns, fill_value=0)

    # Scale all features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Price: £{prediction[0]:,.2f}")