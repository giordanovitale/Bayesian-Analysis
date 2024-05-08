import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
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


def get_data(string):
    data = pd.read_csv(string)
    data.drop(['index',
               'Founded',
               'Rating',
               'Job Title',
               'Salary Estimate',
               'Job Description',
               'Company Name',
               'Location',
               'Headquarters',
               'Industry',
               'Competitors',
               'Hourly',
               'Employer provided',
               'company_txt',
               'seniority_by_title',
               'Lower Salary',
               'Upper Salary',
               'Degree'], axis=1, inplace=True)



    for col in data.columns:
        data[col] = data[col].replace('-1', pd.NA)
        data[col] = data[col].replace('na', pd.NA)
        data[col] = data[col].replace(-1, pd.NA)
        data[col] = data[col].replace('unknown', pd.NA)
        data[col] = data[col].replace('Unknown / Non-Applicable', pd.NA)

    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)


    return data


def inspect_data(data):
    print(data.head(3))
    print('#############################################################')
    print("Shape:\n", data.shape)
    print('#############################################################')
    print("Info:\n", data.info())
    print('#############################################################')
    print('Null values:\n', data.isnull().sum())
    print('#############################################################')





def target_encode_dfs(train_df, val_df, cat_cols, target_col):

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
                        '$10+ billion (USD)':10}

    # Copy the input DataFrames to avoid modifying them directly
    train_encoded = train_df.copy()
    val_encoded = val_df.copy()

    # Initialize the TargetEncoder
    target_encoder = TargetEncoder(cols=cat_cols)

    # Fit the encoder on the training data
    target_encoder.fit(train_encoded[cat_cols], train_encoded[target_col])

    # Transform the training and validation sets
    train_encoded[cat_cols] = target_encoder.transform(train_encoded[cat_cols])
    val_encoded[cat_cols] = target_encoder.transform(val_encoded[cat_cols])

    train_encoded[cat_cols] = train_encoded[cat_cols].round(4)
    val_encoded[cat_cols] = val_encoded[cat_cols].round(4)

    train_encoded.loc[:, 'Revenue'] = train_encoded['Revenue'].map(revenue_mapping)
    val_encoded.loc[:, 'Revenue'] = val_encoded['Revenue'].map(revenue_mapping)

    # Reverse mapping dictionary for each column
    mapping = {}
    for col in cat_cols:
        mapping[col] = dict(zip(train_df[col],train_encoded[col]))

    return train_encoded, val_encoded, mapping






def split(data):

    data_shuffled = data.sample(frac=1, random_state=42)  # Shuffle with a fixed random_state for reproducibility

    # Calculate the index to split the data (80% for training, 20% for testing)
    train_size = int(0.8 * len(data_shuffled))

    # Split the data into training and testing sets
    train = data_shuffled.iloc[:train_size]
    test = data_shuffled.iloc[train_size:]

    return train, test


def feature_eng(X_scaled):

    # Create a column which sums 'Programming Language'
    X_scaled['Programming'] = X_scaled['Python'] + X_scaled['sas']

    # Create a column which sums 'Queries'
    X_scaled['Queries'] = X_scaled['mongo'] + X_scaled['sql']

    # Create a column which sums 'DeepLearning'
    X_scaled['Machine Learning'] = X_scaled['keras'] + X_scaled['pytorch'] + X_scaled['tensor'] + X_scaled['scikit']

    # Create a column which sums 'Dashboards'
    X_scaled['Dashboards'] = X_scaled['bi'] + X_scaled['tableau'] + X_scaled['excel']

    # Create a column which sums 'Cloud'
    X_scaled['Cloud'] = X_scaled['aws'] + X_scaled['google_an']

    # Create a column which sums 'Apache'
    X_scaled['Apache'] = X_scaled['flink'] + X_scaled['spark'] + X_scaled['hadoop']


    X_scaled.drop(['Size', 'Age',
                   'Python', 'spark', 'aws', 'sql', 'sas', 'keras', 'pytorch',
                   'tensor', 'hadoop', 'tableau', 'bi', 'flink', 'mongo', 'google_an',
                   'excel', 'scikit'],
                  axis=1,
                  inplace=True)

    return X_scaled




def minmax_scale_dfs(train_df, test_df):
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler only on the training data
    scaler.fit(train_df)

    # Transform both the training and test data using the same scaler
    train_scaled = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    return train_scaled, test_scaled, scaler

      





 