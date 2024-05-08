import pandas as pd
import numpy as np

import matplotlib as plt
import seaborn as sns
import plotly

import category_encoders as ce
from category_encoders import TargetEncoder

import zipfile

import sklearn
import imblearn
import xgboost as xgb

import sklearn.metrics as metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, Ridge

import pymc as pm
import arviz as az







def reverse_minmax_scaled_value(scaled_value, min_val, max_val):
    """
    Reverse a MinMax scaled value to its original scale.

    Args:
    scaled_value (float): The scaled value to be reversed.
    min_val (float): The minimum value in the original scale.
    max_val (float): The maximum value in the original scale.

    Returns:
    float: The reversed value in the original scale.
    """
    original_value = scaled_value * (max_val - min_val) + min_val
    return original_value







def encode_oos_data(new_data, mapping, scaler):
    
    
    # Create a column which sums 'Programming Language'
    new_data['Programming'] = new_data['Python'] + new_data['sas']

    # Create a column which sums 'Queries'
    new_data['Queries'] = new_data['mongo'] + new_data['sql']

    # Create a column which sums 'DeepLearning'
    new_data['Machine Learning'] = new_data['keras'] + new_data['pytorch'] + new_data['tensor'] + new_data['scikit']

    # Create a column which sums 'Dashboards'
    new_data['Dashboards'] = new_data['bi'] + new_data['tableau'] + new_data['excel']

    # Create a column which sums 'Cloud'
    new_data['Cloud'] = new_data['aws'] + new_data['google_an']

    # Create a column which sums 'Apache'
    new_data['Apache'] = new_data['flink'] + new_data['spark'] + new_data['hadoop']


    new_data.drop(['Python', 'spark', 'aws', 'sql', 'sas', 'keras', 'pytorch',
                   'tensor', 'hadoop', 'tableau', 'bi', 'flink', 'mongo', 'google_an',
                   'excel', 'scikit'],
                  axis=1,
                  inplace=True)

    # Initialize empty dictionaries for each column
    type_of_own_mapping = {}
    job_title_sim_mapping = {}
    sector_mapping = {}
    job_location_mapping = {}

    # Iterate over the keys of the reverse_mapping dictionary
    for col, col_map in mapping.items():
        if col == 'Type of ownership':
            type_of_own_mapping = col_map
        elif col == 'job_title_sim':
            job_title_sim_mapping = col_map
        elif col == 'Sector':
            sector_mapping = col_map
        else:
            job_location_mapping = col_map

    # Original revenue mapping dictionary
    revenue_mapping = {'$1 to $5 million (USD)': 0,
                       '$5 to $10 million (USD)': 1,
                       '$10 to $25 million (USD)': 2,
                       '$25 to $50 million (USD)': 3,
                       '$50 to $100 million (USD)': 4,
                       '$100 to $500 million (USD)': 5,
                       '$500 million to $1 billion (USD)': 6,
                       '$1 to $2 billion (USD)': 7,
                       '$2 to $5 billion (USD)': 8,
                       '$5 to $10 billion (USD)': 9,
                       '$10+ billion (USD)': 10}

    new_data.loc[:, 'job_title_sim'] = new_data['job_title_sim'].map(job_title_sim_mapping)
    new_data.loc[:, 'Revenue'] = new_data['Revenue'].map(revenue_mapping)
    new_data.loc[:, 'Type of ownership'] = new_data['Type of ownership'].map(type_of_own_mapping)
    new_data.loc[:, 'Sector'] = new_data['Sector'].map(sector_mapping)
    new_data.loc[:, 'Job Location'] = new_data['Job Location'].map(job_location_mapping)

    # Scale the encoded features using Min-Max scaling
    columns_to_scale = ['Type of ownership', 'Sector', 'Revenue', 'Avg Salary(K)',
     'Job Location', 'job_title_sim', 'Programming', 'Queries',
      'Machine Learning', 'Dashboards', 'Cloud', 'Apache']
    new_data[columns_to_scale] = scaler.transform(new_data[columns_to_scale])
    new_data.drop('Avg Salary(K)', axis =1 , inplace=True)
    
    

    return new_data





def out_of_sample_pred(scaler, en_new, models_dict):
    # Initialize an empty dictionary to store predictions
    predictions_dict = {}

    # Iterate through the models dictionary
    for model_name, model in models_dict.items():
        # Make predictions using the model
        predictions = model.predict(en_new)
        # Store the predictions in the predictions dictionary
        predictions_dict[model_name] = predictions

    min_train = scaler.data_min_[3] #'Avg Salary(K)'
    max_train = scaler.data_max_[3] #'Avg Salary(K)'

    reversed_predictions_dict = {}
    for model_name, predictions in predictions_dict.items():
        # Reverse the MinMax scaling
        reversed_predictions = reverse_minmax_scaled_value(predictions, min_train, max_train)
        # Round each element of the NumPy array to two decimal places
        rounded_predictions = np.round(reversed_predictions * 1000, 2)
        # Store the rounded predictions in the reversed_predictions_dict
        reversed_predictions_dict[model_name] = rounded_predictions

    return reversed_predictions_dict