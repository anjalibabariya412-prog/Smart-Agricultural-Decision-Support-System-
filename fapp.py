import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

# ─────────── Load All Models and Encoders ───────────
# Yield Prediction Model
model_yield = tf.keras.models.load_model("model_yield.h5", compile=False)
with open("scaler.pkl", "rb") as f:
    yield_scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Crop Recommendation Model
with open("minmaxscaler.pkl", "rb") as scaler_file:
    crop_scaler = pickle.load(scaler_file)
with open("model.pkl", "rb") as model_file:
    crop_model = pickle.load(model_file)

crop_dict = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas', 6: 'mothbeans',
    7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango',
    13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple', 17: 'orange', 18: 'papaya',
    19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'
}

# ─────────── Page Config ───────────
st.set_page_config(page_title="🌿 Smart Agri System", layout="centered")
st.title("🌿 Smart Agricultural Decision Support System")

# ─────────── Tabs ───────────
tab1, tab2 = st.tabs(["🌾 Crop Yield Prediction", "🌱 Crop Recommendation"])

# ─────────── Tab 1: Crop Yield Prediction ───────────
with tab1:
    st.header("🌾 Crop Yield Prediction")
    st.markdown("Predict expected crop yield (tons/hectare) based on soil and environmental inputs.")

    state = st.selectbox("State Name", options=label_encoders["State_Name"].classes_)
    crop_type = st.selectbox("Crop Type", options=label_encoders["Crop_Type"].classes_)
    crop = st.selectbox("Crop", options=label_encoders["Crop"].classes_)
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=500, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=500, value=20)
    K = st.number_input("Potassium (K)", min_value=0, max_value=500, value=10)
    pH = st.slider("Soil pH", 3.0, 10.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=100.0)
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.5)
    area = st.number_input("Area (in hectares)", min_value=0.1, value=1.0)

    if st.button("🔍 Predict Yield"):
        # Encode labels
        state_encoded = label_encoders["State_Name"].transform([state])[0]
        crop_type_encoded = label_encoders["Crop_Type"].transform([crop_type])[0]
        crop_encoded = label_encoders["Crop"].transform([crop])[0]

        # Scale numerical features
        numerical_input = np.array([[N, P, K, pH, rainfall, temperature, area]])
        scaled_numerical = yield_scaler.transform(numerical_input)

        # Combine encoded and scaled inputs
        encoded_input = np.array([[state_encoded, crop_type_encoded, crop_encoded]])
        input_data = np.hstack((encoded_input, scaled_numerical))

        # Predict yield
        yield_pred = model_yield.predict(input_data)[0][0]
        st.success(f"🌾 Predicted Yield: **{yield_pred:.2f} tons**")

# ─────────── Tab 2: Crop Recommendation ───────────
with tab2:
    st.header("🌱 Crop Recommendation")
    st.markdown("Get crop suggestions based on soil nutrients and climate.")

    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1, key="N2")
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1, key="P2")
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1, key="K2")
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

    if st.button("🌿 Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_input = crop_scaler.transform(input_data)
        prediction = crop_model.predict(scaled_input)
        crop_prediction = crop_dict.get(int(prediction[0]), "Unknown crop")
        st.success(f"🌿 Recommended Crop: **{crop_prediction.capitalize()}**")
