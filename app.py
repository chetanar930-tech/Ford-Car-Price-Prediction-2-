import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("Ford_Model.pkl")
scaler = joblib.load("Scaler.pkl")
columns = joblib.load("Columns.pkl")

st.title("Ford Car Price Prediction")

model_name = st.selectbox("Model",
                          ["Fiesta","Focus","Kuga","EcoSport","Mondeo"])

transmission = st.selectbox(
    "Transmission",
    ["Manual","Automatic","Semi-Auto"]
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","Hybrid"]
)

year = st.number_input("Year", 2000, 2025, 2018)
mileage = st.number_input("Mileage", 0, 300000, 20000)
tax = st.number_input("Tax", 0, 500, 150)
mpg = st.number_input("MPG", 0.0, 100.0, 50.0)
engineSize = st.number_input("Engine Size", 0.0, 5.0, 1.5)

if st.button("Predict Price"):

    data = {
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize],
        "model": [model_name],
        "transmission": [transmission],
        "fuelType": [fuelType]
    }

    # Create DataFrame first
    input_df = pd.DataFrame(data)

    # Then perform encoding
    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    input_df = pd.DataFrame(
        scaler.transform(input_df),
        columns=columns
    )

    prediction = model.predict(input_df)

    st.success(f"Predicted Price: £{prediction[0]:,.2f}")