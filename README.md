# Crop Recommendation System

A web application that recommends optimal crops based on soil and environmental conditions using Streamlit.

## Project Overview

This web application uses an Artificial Neural Network (ANN) model to predict the most suitable crops for cultivation based on soil nutrient content and environmental conditions. The application is built using Streamlit for both the frontend and backend.

## Project Structure

- `capp.py`: Streamlit application
- `ann_crop_recommendation.h5`: Trained ANN model
- `crop_label_encoder.pkl`: Label encoder for crop names
- `crop_scaler.pkl`: Scaler for input features
- `Crop_recommendation.csv`: Training dataset
- `Q2.ipynb`: Jupyter notebook with model training and analysis

## Features

- Interactive web interface built with Streamlit
- Real-time crop recommendations
- Input validation for all parameters
- Success notifications for predictions
- User-friendly input controls with proper ranges
- Visual feedback for predictions

## Input Parameters

The application accepts the following input parameters:

1. **Soil Nutrients**
   - Nitrogen (N) content in soil
   - Phosphorus (P) content in soil
   - Potassium (K) content in soil

2. **Soil Properties**
   - Soil pH (0.0 to 14.0)

3. **Environmental Conditions**
   - Rainfall (0.0 to 500.0 mm)
   - Temperature (0.0 to 50.0 °C)
   - Humidity (0.0 to 100.0 %)

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- TensorFlow/Keras
- Scikit-learn

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install flask streamlit pandas numpy tensorflow scikit-learn
   ```
3. Run the application:
   ```bash
   streamlit run capp.py
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501`
2. Enter the following parameters:
   - Soil nutrient values (NPK)
   - Soil pH
   - Environmental conditions (rainfall, temperature, humidity)
3. Click "Predict Crop" button
4. View the recommended crop in the success message

## Model Architecture

The system uses an Artificial Neural Network (ANN) implemented with TensorFlow/Keras. The model is trained on a comprehensive dataset of crop characteristics and environmental conditions.

## Dataset

The dataset (`Crop_recommendation.csv`) contains historical data about various crops and their optimal growing conditions. It includes features such as:
- N (Nitrogen)
- P (Phosphorous)
- K (Potassium)
- pH
- Rainfall
- Temperature
- Humidity
- Crop labels

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

