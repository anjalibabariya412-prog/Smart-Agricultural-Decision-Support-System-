import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model and preprocessing tools
model = load_model("ann_crop_recommendation.h5")
scaler = joblib.load("crop_scaler.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# Streamlit UI
st.set_page_config(page_title="🌿 Crop Recommendation System", layout="centered")
st.title("🌾 Smart Crop Recommendation System")
st.markdown("Provide the following soil and environmental parameters:")

# Input fields
nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1)
potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

# Prediction logic
if st.button("Predict Crop"):
    # Prepare input
    input_data = np.array([[nitrogen, phosphorus, potassium, ph, rainfall, temperature, humidity]])
    input_scaled = scaler.transform(input_data)
    
    # Predict and decode
    prediction = model.predict(input_scaled)
    predicted_index = np.argmax(prediction)
    recommended_crop = label_encoder.inverse_transform([predicted_index])[0]

    st.success(f"✅ Recommended Crop: **{recommended_crop}**")
