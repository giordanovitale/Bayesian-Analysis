import os
import sys
import subprocess

# Get the current working directory
current_dir = os.getcwd()

repo_url = 'https://github.com/UBS-IB/bayesian_tree'

cloned_repo_folder = 'bayesian_tree'

cloned_repo_path = os.path.join(current_dir, cloned_repo_folder)


subprocess.run(['git', 'clone', repo_url, cloned_repo_path])



sys.path.append(cloned_repo_path)



import bayesian_tree
from bayesian_tree import bayesian_decision_tree
from bayesian_tree import examples
from bayesian_decision_tree import regression
from bayesian_decision_tree.regression import PerpendicularRegressionTree



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







def convert_scaled(scaled_metric):

    conversion_factor = 209.4231 # This value was calculated during the experiments
    converted_metric = scaled_metric * conversion_factor
    return converted_metric
    


def regression_results(y_true, y_pred):
    # Regression metrics
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('#########')
    converted_mae = convert_scaled(mean_absolute_error)
    converted_rmse = convert_scaled(np.sqrt(mse))
    converted_mse = convert_scaled(mse)

    # Calculate number of observations
    n = len(y_true)

    # Calculate number of parameters (including the intercept)
    p = 12

    # Calculate residual sum of squares (RSS)
    rss = converted_mse * n

    # Calculate BIC score
    bic_score = n * np.log(rss/n) + p * np.log(n)

    print('r2: ', round(r2, 4))
    print('MAE: ', round(converted_mae, 4))
    print('RMSE: ', round(converted_rmse, 4)) 
    print('BIC: ', round(bic_score, 4))
  



def freq_lr(X_train,y_train, X_test ):
    normal_lr = LinearRegression()
    normal_lr.fit(X_train, y_train)

    y_val_predictions_nlr = normal_lr.predict(X_test)

    return normal_lr , y_val_predictions_nlr


def bayesian_lr(X_train, y_train, m = 0, s= 10, samples = 2000, tune = 1000,cores =1):
  # Define Bayesian Linear Regression model
  with pm.Model() as bayesian_lr_model:
      # Priors for coefficients
      beta_0 = pm.Normal('intercept', mu=m, sigma= s)  # Prior for intercept
      beta = pm.Normal('beta', mu=m, sigma=s , shape=X_train.shape[1])  # Prior for coefficients

      # Linear regression equation
      mu = beta_0 + pm.math.dot(beta, X_train.values.T)

      # Likelihood (sampling distribution)
      sigma = pm.HalfNormal('sigma', sigma= s)  # Prior for the standard deviation of the noise
      y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train.values)  # Likelihood function

      # Sample from the posterior distribution
      trace = pm.sample(samples, tune=tune, cores=cores)


      return trace





def predict_br(trace, X_test):
  coefficients_dict = dict(zip(pm.summary(trace)['mean'][:12].index,
                        pm.summary(trace)['mean'][:12].values))

  lr_model = LinearRegression()

  # Set Coefficients
  # Extract coefficients from the dictionary
  intercept = np.array(coefficients_dict['intercept'])
  coefficients = np.array(list(coefficients_dict.values())[1:])

  # Set coefficients in the logistic regression model
  lr_model.intercept_ = np.array([intercept])
  lr_model.coef_ = np.array([coefficients])

  # Fit the Model
  y_val_predictions = lr_model.predict(X_test)

  return lr_model, y_val_predictions, coefficients_dict





def bayesian_lr_t(X_train, y_train, n=1, m=0, s=10, samples=2000, tune=1000, cores=1):
    # Define Bayesian Linear Regression model with Student's t priors
    with pm.Model() as bayesian_t_model:

        # Priors for coefficients
        intercept_t = pm.StudentT('intercept', nu=n, mu=m, sigma=s)  # Prior for intercept
        beta_t = pm.StudentT('beta', nu=n, mu=m, sigma=s, shape=X_train.shape[1])  # Prior for coefficients

        # Linear regression equation
        mu_t = intercept_t + pm.math.dot(beta_t, X_train.values.T)

        # Likelihood
        sigma_t = pm.HalfStudentT('sigma', nu=n, sigma=s)  # Prior for the standard deviation of the noise
        y_obs_t = pm.Normal('y_obs', mu=mu_t, sigma=sigma_t, observed=y_train.values)  # Likelihood function

        # Sample from the posterior distribution
        trace_t = pm.sample(samples, tune=tune, cores=cores)

    return trace_t




def get_nearest_value_string(i, value, X_train, mapping):
        sorted_mapping = {key: dict(sorted(value.items(), key=lambda item: item[1])) for key, value in mapping.items()}
        features = X_train.columns
        if features[i] not in sorted_mapping:
            return "Feature not found"

        values = sorted_mapping[features[i]]
        nearest_value = min(values.values(), key=lambda x: abs(x - value))
        nearest_key = [key for key, val in values.items() if val == nearest_value][0]

        return nearest_key

def inverse_scaling(i, scaler, value):
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    result = value * (data_max[i] - data_min[i]) + data_min[i]
    return round(result, 4)

def plot_tree_structure(node, feature_names, indent=''):
    if node is None:
        return

    if node.is_leaf:
        print(f"{indent}└── Leaf: Avg Salary={node.value * 1000}$")
    else:
        feature_name = feature_names[node.feature_index]
        print(f"{indent}├── {feature_name} <= {node.threshold}:")
        plot_tree_structure(node.left_child, feature_names, indent + '│    ')
        print(f"{indent}└── {feature_name} > {node.threshold}:")
        plot_tree_structure(node.right_child, feature_names, indent + '     ')


# Define the PerpendicularRegressionTree structure
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, is_leaf=False):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.is_leaf = is_leaf
        self.left_child = None
        self.right_child = None

def b_tree(X_train, y_train, X_test, y_test, scaler, mapping):
    sorted_mapping = {key: dict(sorted(value.items(), key=lambda item: item[1])) for key, value in mapping.items()}
    mu = y_train.mean()
    sd_prior = y_train.std() / 10
    prior_pseudo_observations = 1
    kappa = prior_pseudo_observations
    alpha = prior_pseudo_observations / 2
    var_prior = sd_prior**2
    tau_prior = 1/var_prior
    beta = alpha/tau_prior
    prior = np.array([mu, kappa, alpha, beta])

    model = PerpendicularRegressionTree(
        partition_prior=0.9,
        prior=prior,
        delta=0)
    

    model.fit(np.array(X_train), np.array(y_train))
    print("-----------------")
    print('0000000000000000000000000 tree model 0000000000000000000000000000')
    print(model)
    print("-----------------")
    
    print('0000000000000000000000000 info 0000000000000000000000000000')
    print('Tree depth and number of leaves: {}, {}'.format(model.get_depth(), model.get_n_leaves()))
    print('Feature importance:', [float(x) for x in model.feature_importance()])

    # Compute accuracy
    y_pred_train = model.predict(X_train)
    y_pred_bayes_tree = model.predict(X_test)
    accuracy_train = convert_scaled(mean_absolute_error(y_train, y_pred_train))
    accuracy_test = convert_scaled(mean_absolute_error(y_test, y_pred_bayes_tree))
    info_train = 'Train MAE: {:.4f}'.format(accuracy_train)
    info_test = 'Test MAE:  {:.4f}'.format(accuracy_test)
    print(info_train)
    print(info_test)


    # Create an instance of PerpendicularRegressionTree
    root = TreeNode(feature_index=4, threshold=get_nearest_value_string(4, inverse_scaling(5, scaler, 0.6663830411117424), X_train, mapping))
    root.left_child = TreeNode(value=inverse_scaling(3, scaler, 0.25522993368285146), is_leaf=True)
    root.right_child = TreeNode(feature_index=3, threshold=get_nearest_value_string(3, inverse_scaling(4, scaler, 0.2858181791333494), X_train,mapping))
    root.right_child.left_child = TreeNode(feature_index=4, threshold=get_nearest_value_string(4, inverse_scaling(5, scaler, 0.7144313918509577), X_train,mapping))
    root.right_child.left_child.left_child = TreeNode(feature_index=0, threshold=get_nearest_value_string(0, inverse_scaling(0, scaler, 0.7970749872523215), X_train, mapping))
    root.right_child.left_child.left_child.left_child = TreeNode(value=inverse_scaling(3, scaler, 0.2859168981572049), is_leaf=True)
    root.right_child.left_child.left_child.right_child = TreeNode(value=inverse_scaling(3, scaler, 0.38555156696271375), is_leaf=True)
    root.right_child.left_child.right_child = TreeNode(feature_index=5, threshold=inverse_scaling(6, scaler, 0.75))
    root.right_child.left_child.right_child.left_child = TreeNode(value=inverse_scaling(3, scaler, 0.4155628823712319), is_leaf=True)
    root.right_child.left_child.right_child.right_child = TreeNode(value=inverse_scaling(3, scaler, 0.3517871169872422), is_leaf=True)
    root.right_child.right_child = TreeNode(feature_index=1, threshold=get_nearest_value_string(1, inverse_scaling(1, scaler, 0.328670561348811), X_train, mapping))
    root.right_child.right_child.left_child = TreeNode(value=inverse_scaling(3, scaler, 0.41767026781172967), is_leaf=True)
    root.right_child.right_child.right_child = TreeNode(value=inverse_scaling(3, scaler, 0.5749117181678143), is_leaf=True)

    # Print the tree structure
    feature_names = X_train.columns
    print('0000000000000000000000000 interpreted tree 0000000000000000000000000000')
    plot_tree_structure(root, feature_names)

    highlight_values = ["Data scientist project manager",
                        "WA",
                        "other scientist",
                        "Subsidiary or Business Segment",
                        "Subsidiary or Business Segment",
                        "other scientist",
                        "WA",
                        "Real Estate",
                        "Real Estate"]
    

    for feature, values in sorted_mapping.items():
        plt.figure(figsize=(10, 6))
        bars = plt.barh(list(values.keys()), list(values.values()))  # Convert dict_keys to a list

        # Iterate through bars and highlight those with values in highlight_values in red
        for bar, value in zip(bars, values.keys()):
            if value in highlight_values:
                bar.set_color('red')

        plt.xlabel("Values")
        plt.ylabel(feature)
        plt.title(f"Values for {feature}")

        # Add text annotations for the values
        for index, value in enumerate(values.values()):
            plt.text(value, index, f'{value:.2f}', ha='left', va='center')

        plt.show()
        
        
        

    # Retrieve feature names and importances
    feature_names = X_train.columns
    importances = model.feature_importance()

    # Filter out zero importances
    non_zero_indices = importances.nonzero()[0]
    filtered_feature_names = feature_names[non_zero_indices]
    filtered_importances = importances[non_zero_indices]

    # Sort filtered feature importances in descending order
    sorted_indices = filtered_importances.argsort()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(filtered_feature_names)), filtered_importances[sorted_indices], align="center")
    plt.yticks(range(len(filtered_feature_names)), filtered_feature_names[sorted_indices])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


    return model, y_pred_bayes_tree





def freq_tree(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import export_text

    frequentist_tree = DecisionTreeRegressor(random_state=0,
                                         criterion='absolute_error',
                                         max_depth=3,
                                         min_samples_split=4)

    frequentist_tree.fit(X_train, y_train)

    y_freq_tree_predictions = frequentist_tree.predict(X_test)

    print(f"MAE of the frequentist tree: {convert_scaled(mean_absolute_error(y_test, y_freq_tree_predictions))}")
    print(export_text(frequentist_tree))
    
    # Retrieve feature names and importances
    feature_names = frequentist_tree.feature_names_in_
    importances = frequentist_tree.feature_importances_

    # Filter out zero importances
    non_zero_indices = importances.nonzero()[0]
    filtered_feature_names = feature_names[non_zero_indices]
    filtered_importances = importances[non_zero_indices]

    # Sort filtered feature importances in descending order
    sorted_indices = filtered_importances.argsort()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(filtered_feature_names)), filtered_importances[sorted_indices], align="center")
    plt.yticks(range(len(filtered_feature_names)), filtered_feature_names[sorted_indices])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


    return frequentist_tree, y_freq_tree_predictions





def compare_models(predictions_dict, y_test):
    results = {}

    for model, pred in predictions_dict.items():
        results[model] = convert_scaled(mean_absolute_error(y_test, pred))

    results_df = pd.DataFrame(results, index=['MAE']).T

    results_df = results_df.sort_values(by='MAE')

    results_df.index.name = 'Models'

    results_df['Model_Type'] = ['Bayesian' if 'Bayesian' in index else 'Frequentist' for index in results_df.index]
    
    plt.figure(figsize=(11,5))
    ax = sns.barplot(data=results_df, x='MAE', y=results_df.index,
                     orient='h', hue='Model_Type')
    plt.title('Overall comparison of models')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    
    for index, value in enumerate(results_df['MAE']):
        ax.text(value, index, f'{value:.2f}', ha='left', va='center')

    plt.tight_layout()
    plt.show()

    return results_df








def plot_coef(coefficients_dict):
  mapping_names = {
      'intercept': 'intercept',
      'beta[0]': 'Type of ownership',
      'beta[1]': 'Sector',
      'beta[2]': 'Revenue',
      'beta[3]': 'job_title_sim',
      'beta[4]': 'Programming',
      'beta[5]': 'Queries',
      'beta[6]': 'Machine Learning',
      'beta[7]': 'Dashboards',
      'beta[8]': 'Cloud',
      'beta[9]': 'Apache',
      'beta[10]': 'Job Location'
  }

  mapped_coefficients_dict = {mapping_names[key]: value for key, value in coefficients_dict.items()}
  sorted_coefficients = sorted(mapped_coefficients_dict.items(), key=lambda coef_value: np.abs(coef_value[1]), reverse=True)

  # Extract feature names and corresponding coefficients
  feature_names = [item[0] for item in sorted_coefficients]
  coefficients = [item[1] for item in sorted_coefficients]

  sns.set(style="ticks")

  plt.figure(figsize=(11, 6))

  # custom colors for the bars
  for feature, coef in zip(feature_names, coefficients):
      color = 'red' if coef < 0 else 'green'
      plt.barh(feature, coef, color=color, align='center')

  plt.xlabel('Coefficient Magnitude')
  plt.ylabel('Feature')
  plt.title('Feature Importance based on Coefficients')
  plt.gca().invert_yaxis()

  # add annotations
  for feature, coef in zip(feature_names, coefficients):
    if coef>0:
      plt.text(coef, feature, round(coef, 3), va='center', ha='left', fontsize=10)
    else:
      plt.text(coef, feature, round(coef, 3), va='center', ha='right', fontsize=10)

  plt.show()



  
