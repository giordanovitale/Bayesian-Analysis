# Salary Prediction for Data Jobs App

This app is designed to perform Salary Prediction using the Bayesian Ridge Model. It allows users to input data and obtain predictions based on the trained model.

## Files

- `app.py`: The main Python script containing the Flask web application.
- `bayesian_ridge_model.pkl`: Pickle file containing the trained Bayesian Ridge regression model.
- `helpers_outofsample.py`: Python script containing helper functions for out-of-sample data processing.
- `mapping.pkl`: Pickle file containing mappings used for data preprocessing.
- `requirements.txt`: Text file listing all the dependencies required to run the application.
- `scaler.pkl`: Pickle file containing the scaler used for data normalization.
- `templates/`: Directory containing HTML templates for the web interface.
  - `index.html`: HTML template for the main page of the web application.

## Usage

1. Clone this repository.
2.  Run the Flask application by executing `python app.py`.
3. Install the required dependencies by running `pip install -r requirements.txt` in case any error occured.
4. Open your web browser and navigate to `http://localhost:5000` to access the application.

## Note

Ensure that you have Python and Flask installed on your system before running the application. Additionally, make sure that all required files (`bayesian_ridge_model.pkl`, `mapping.pkl`, and `scaler.pkl`) are present in the same directory as `app.py`.
