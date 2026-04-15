import streamlit as st
import pandas as pd
import pickle

# === Load Model ===
with open("model.pkl", "rb") as file:
    model, version = pickle.load(file)

# === Title ===
st.title("🏗️ Concrete Strength Prediction App")
st.write(f"Model Version: {version}")

# === User Inputs ===
cement = st.number_input("Cement", min_value=0.0)
slag = st.number_input("Blast Furnace Slag", min_value=0.0)
flyash = st.number_input("Fly Ash", min_value=0.0)
water = st.number_input("Water", min_value=0.0)
superplasticizer = st.number_input("Superplasticizer", min_value=0.0)
coarseaggregate = st.number_input("Coarse Aggregate", min_value=0.0)
fineaggregate = st.number_input("Fine Aggregate", min_value=0.0)
age = st.number_input("Age (days)", min_value=1)

# === Prediction ===
if st.button("Predict Strength"):

    input_data = pd.DataFrame({
        "cement": [cement],
        "slag": [slag],
        "flyash": [flyash],
        "water": [water],
        "superplasticizer": [superplasticizer],
        "coarseaggregate": [coarseaggregate],
        "fineaggregate": [fineaggregate],
        "age": [age]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Concrete Strength: {prediction[0]:.2f} MPa")
