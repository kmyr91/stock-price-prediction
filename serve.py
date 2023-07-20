#serve.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import yfinance as yf
import pandas as pd
from src import data_preprocessing
import mlflow
import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True)

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        ticker = request.json['ticker']
    else:
        ticker = request.args.get('ticker')

    # Fetch the best model from MLflow
    client = mlflow.tracking.MlflowClient(tracking_uri="sqlite:///mlflow.db")
    experiment_id = client.get_experiment_by_name("stock-price-prediction").experiment_id
    runs = client.search_runs(experiment_id)
    best_run = min(runs, key=lambda r: r.data.metrics['test_mse'])
    model_path = f"mlruns/1/{best_run.info.run_id}/artifacts/model/data/model"
    model = load_model(model_path)

    # Fetch the data
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(365)  # Get data for a year: should be enough
    data = yf.download(ticker, start=start_date, end=end_date)

    # Save the data to a CSV file
    data.to_csv('temp_data.csv')

    # Preprocess the data
    train, test, scaler, dates, train_size = data_preprocessing.load_and_preprocess_data('temp_data.csv')

    # Create dataset for prediction
    features, labels, dates_new = data_preprocessing.create_dataset(test, dates)

    # Predict the next day
    pred = model.predict(features)

    # Convert scaled prediction back to original form
    pred_transformed = scaler.inverse_transform(np.concatenate((pred[-1].reshape(-1, 1), np.zeros((len(pred[-1]), 1))), axis=1))[:, 0]
    # Fetch today's actual closing price
    todays_data = yf.download('AAPL', start=pd.to_datetime('today'), end=pd.to_datetime('today'))
    actual_price = todays_data['Close'][0]
    return jsonify({
        'prediction': pred_transformed[-1].tolist(),
        'actual': actual_price
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
