from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from helpers_outofsample import (reverse_minmax_scaled_value,
                                 encode_oos_data,
                                 out_of_sample_pred)


import numpy as np

import pickle

pd.set_option('display.max_columns', None)


app = Flask(__name__)

# Load models and mapping
models_dict = {}
bayesian_ridge = joblib.load('bayesian_ridge_model.pkl')
models_dict['Bayesian_Ridge'] = bayesian_ridge
mapping = joblib.load('mapping.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_salary():
    # Get input data from request
    input_data = request.form.to_dict()
    


    # Convert specific keys to integers
    keys_to_convert = ['Python', 'spark', 'aws', 'sql', 'sas', 'keras', 'pytorch', 
                       'tensor', 'hadoop', 'tableau', 'bi', 'flink', 'mongo', 
                       'google_an', 'excel', 'scikit']

    for key in keys_to_convert:
        if key in input_data:
            input_data[key] = int(input_data[key])

    # Log the input data
    print("Received input data:", input_data)


    
    # Create DataFrame from input data
    new_data = pd.DataFrame([input_data])
    
    new_data.insert(3, "Avg Salary(K)", 0)
    
    
    print("dataframe:", new_data)
    

    # Encode input data
    oosp = encode_oos_data(new_data, mapping, scaler)
    
    # Log the encoded input data
    print("Encoded input data:", oosp)
     

    # Perform prediction
    predictions_dict = out_of_sample_pred(scaler, oosp, models_dict)

    # Log the predictions dictionary
    print("Predictions:", predictions_dict)

    # Extract predicted salary
    predicted_salary = predictions_dict['Bayesian_Ridge'][0]  

    # Return predicted salary as JSON response
    return jsonify({'predicted_salary': predicted_salary})


if __name__ == "__main__":
    app.run(debug=True)
