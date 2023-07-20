# main.py
from src import data_collection, data_preprocessing, model_training, model_evaluation
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import mlflow 
import mlflow.keras

ticker = 'AAPL'
start_date = '2016-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
data_path = f'Data/{ticker}_price_data.csv'

data_collection.collect_data(ticker, start_date, end_date, data_path)
train, test, scaler, dates, train_size = data_preprocessing.load_and_preprocess_data(data_path)
X_train, Y_train, dates_train = data_preprocessing.create_dataset(train, dates)
X_test, Y_test, dates_test = data_preprocessing.create_dataset(test, dates[train_size:])


model_path = f'Models/{ticker}_model.h5'
train_new_model = True  

if train_new_model:
    model, run_id = model_training.create_and_train_model(X_train, Y_train, model_path)
    # log model to mlflow here after training
    with mlflow.start_run(run_id=run_id):
        model_evaluation.evaluate_model(model, X_test, Y_test, scaler, ticker, dates_test)
else:
    # find the last run id
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name("stock-price-prediction").experiment_id
    run_infos = client.list_run_infos(experiment_id)
    last_run_id = run_infos[-1].run_id

    # load model from mlflow
    model = mlflow.keras.load_model(f"runs:/{last_run_id}/model")
    model_evaluation.evaluate_model(model, X_test, Y_test, scaler, ticker, dates_test)
