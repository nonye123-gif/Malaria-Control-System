import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load model and preprocessing artifacts
MODEL_PATH = 'model_folder/malaria_severity_lstm_attention_model.keras'
PREPROCESSOR_PATH = 'model_folder/preprocessor.joblib'
ENCODER_PATH = 'model_folder/severity_encoder.joblib'

try:
    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    severity_encoder = joblib.load(ENCODER_PATH)
    print("Model and preprocessing artifacts loaded successfully")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model = preprocessor = severity_encoder = None

# Define drugs list (same as training)
DRUGS = ['Artemether', 'Lumefantrine', 'Quinine', 'Fansidar', 'Mefloquine', 
         'Doxycycline', 'Primaquine', 'Chlorproguanil', 'Amodiaquine']

# Date parsing function
def safe_parse_date(date_str):
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

def predict_severity(input_data):
    """Process input data and make prediction"""
    if model is None or preprocessor is None:
        return {"error": "Model not loaded"}, 500
        
    # Create dataframe from input
    new_df_raw = pd.DataFrame([input_data])
    
    # Add drug flags
    admin_drugs = input_data.get('Drugs Administered', '')
    for drug in DRUGS:
        new_df_raw[drug] = 1 if drug.lower() in admin_drugs.lower() else 0
    
    # Add date features
    date_val = safe_parse_date(input_data['Date'])
    if pd.isna(date_val):
        date_val = pd.Timestamp.today()
        
    new_df_raw['Date'] = date_val
    new_df_raw['Year'] = date_val.year
    new_df_raw['Month'] = date_val.month
    new_df_raw['Day'] = date_val.day
    new_df_raw['Day_of_year'] = date_val.dayofyear
    new_df_raw['Week_of_year'] = date_val.isocalendar().week
    
    # Add interaction feature
    new_df_raw['Age_BodyTemp_Interaction'] = (
        float(input_data.get('Age', 0)) * 
        float(input_data.get('Body Temp (°C)', 0)
    ))
    
    # Define expected features
    categorical_features = ['Gender', 'Genotype', 'Blood Group', 'LGA', 'Season', 'Diagnosis']
    numerical_features = ['Age', 'Body Temp (°C)', 'Latitude', 'Longitude', 
                          'Rainfall (mm)', 'Climate Temp (°C)', 
                          'Year', 'Month', 'Day', 'Day_of_year', 'Week_of_year',
                          'Age_BodyTemp_Interaction']
    
    # Create full feature set
    all_features = numerical_features + categorical_features + DRUGS
    df_for_transform = pd.DataFrame(columns=all_features)
    
    # Populate dataframe
    for col in all_features:
        if col in new_df_raw.columns:
            df_for_transform[col] = new_df_raw[col]
        else:
            # Set defaults for missing features
            if col in DRUGS:
                df_for_transform[col] = 0
            elif col in numerical_features:
                df_for_transform[col] = 0.0
            else:
                df_for_transform[col] = 'Unknown'
    
    # Transform features
    X_new = preprocessor.transform(df_for_transform)
    X_new = X_new.reshape(X_new.shape[0], 1, X_new.shape[1])
    
    # Make prediction
    probabilities = model.predict(X_new)[0]
    class_idx = np.argmax(probabilities)
    severity = severity_encoder.inverse_transform([class_idx])[0]
    
    # Format probabilities
    prob_dict = {
        cls: float(prob) 
        for cls, prob in zip(severity_encoder.classes_, probabilities)
    }
    
    return {
        'severity': severity,
        'probabilities': prob_dict
    }

@app.route('/')
def home():
    """Render main form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction"""
    if request.method == 'POST':
        try:
            # Get form data
            form_data = {
                'Age': request.form['age'],
                'Gender': request.form['gender'],
                'Genotype': request.form['genotype'],
                'Blood Group': request.form['blood_group'],
                'Body Temp (°C)': request.form['temperature'],
                'Latitude': request.form['latitude'],
                'Longitude': request.form['longitude'],
                'Rainfall (mm)': request.form['rainfall'],
                'Climate Temp (°C)': request.form['climate_temp'],
                'LGA': request.form['lga'],
                'Season': request.form['season'],
                'Date': request.form['date'],
                'Drugs Administered': request.form['drugs'],
                'Diagnosis': 'Malaria'  # Always malaria for this system
            }
            
            # Get prediction
            result = predict_severity(form_data)
            
            # Format results for display
            result['form_data'] = form_data
            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        result = predict_severity(data)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)