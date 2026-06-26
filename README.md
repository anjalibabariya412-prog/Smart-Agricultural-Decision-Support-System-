# Crop Recommendation System

A web application that recommends optimal crops based on soil and environmental conditions using Streamlit.

# Project Overview

This web application uses an Artificial Neural Network (ANN) model to predict the most suitable crop for cultivation based on soil nutrient content and environmental conditions. The application is built using Streamlit for the user interface and TensorFlow/Keras for the machine learning model.

# Project Structure

* `capp.py`: Streamlit application
* `ann_crop_recommendation.h5`: Trained ANN model
* `crop_label_encoder.pkl`: Label encoder for crop names
* `crop_scaler.pkl`: Scaler for input features
* `Crop_recommendation.csv`: Training dataset
* `Q2.ipynb`: Jupyter notebook containing model training and analysis

# Features

* Interactive web interface built with Streamlit
* Real-time crop recommendations
* Input validation for all parameters
* Instant prediction results
* User-friendly input controls with proper ranges
* Simple and responsive interface

# Input Parameters

The application accepts the following input parameters:

### Soil Nutrients

* Nitrogen (N) content in soil
* Phosphorus (P) content in soil
* Potassium (K) content in soil

### Soil Properties

* Soil pH (0.0 to 14.0)

### Environmental Conditions

* Rainfall (0.0 to 500.0 mm)
* Temperature (0.0 to 50.0 °C)
* Humidity (0.0 to 100.0%)

# Requirements

* Python 3.8+
* Streamlit
* Pandas
* NumPy
* TensorFlow/Keras
* Scikit-learn
* Joblib

# Installation

* Clone the repository.
* Install the required packages:

```bash
pip install streamlit pandas numpy tensorflow scikit-learn joblib
```

* Run the application:

```bash
streamlit run capp.py
```

# Usage

* Open your web browser and navigate to `http://localhost:8501`.
* Enter the following parameters:

  * Soil nutrient values (N, P, K)
  * Soil pH
  * Rainfall
  * Temperature
  * Humidity
* Click the **Predict Crop** button.
* View the recommended crop displayed on the screen.

# Model Architecture

The system uses an Artificial Neural Network (ANN) implemented with TensorFlow/Keras. The model is trained on a crop recommendation dataset containing soil and environmental information to predict the most suitable crop.

# Dataset

The dataset (`Crop_recommendation.csv`) contains historical data about various crops and their optimal growing conditions. It includes the following features:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Soil pH
* Rainfall
* Temperature
* Humidity
* Crop Label

# Contributing

1. Fork the repository.
2. Create your feature branch:

```bash
git checkout -b feature/AmazingFeature
```

3. Commit your changes:

```bash
git commit -m "Add some AmazingFeature"
```

4. Push to the branch:

```bash
git push origin feature/AmazingFeature
```

5. Open a Pull Request.
