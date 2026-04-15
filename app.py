import streamlit as st
import pandas as pd
import pickle
import numpy as np

# === Page Config ===
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# === Load Model ===
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model, version = pickle.load(f)
    return model, version

model, version = load_model()

# === Sidebar ===
st.sidebar.title("⚙️ Settings")
st.sidebar.write(f"Model Version: {version}")

mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Prediction"])

# === Title ===
st.title("🏗️ Concrete Strength Prediction App")
st.markdown("Predict compressive strength of concrete using ML")

# =========================================
# 🔹 SINGLE PREDICTION MODE
# =========================================
if mode == "Single Prediction":

    st.subheader("🔢 Enter Material Values")

    col1, col2 = st.columns(2)

    with col1:
        cement = st.slider("Cement", 0.0, 600.0, 300.0)
        slag = st.slider("Blast Furnace Slag", 0.0, 300.0, 0.0)
        flyash = st.slider("Fly Ash", 0.0, 300.0, 0.0)
        water = st.slider("Water", 0.0, 300.0, 150.0)

    with col2:
        superplasticizer = st.slider("Superplasticizer", 0.0, 50.0, 5.0)
        coarseaggregate = st.slider("Coarse Aggregate", 0.0, 1500.0, 1000.0)
        fineaggregate = st.slider("Fine Aggregate", 0.0, 1500.0, 800.0)
        age = st.slider("Age (days)", 1, 365, 28)

    # Prediction
    if st.button("🚀 Predict Strength"):
        try:
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

            prediction = model.predict(input_data)[0]

            st.success(f"💪 Predicted Strength: {prediction:.2f} MPa")

            # Insight
            if prediction < 20:
                st.warning("Low strength concrete ⚠️")
            elif prediction < 40:
                st.info("Moderate strength concrete ℹ️")
            else:
                st.success("High strength concrete 🔥")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================================
# 🔹 BATCH PREDICTION MODE
# =========================================
else:

    st.subheader("📂 Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)

            st.write("📊 Uploaded Data", df.head())

            predictions = model.predict(df)
            df["Predicted_Strength"] = predictions

            st.success("✅ Predictions Completed")

            st.write(df.head())

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# =========================================
# 🔹 FOOTER
# =========================================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & XGBoost")
