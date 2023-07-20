# data_preprocessing.py
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
    data = data[['Close', 'Volume']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    dataset = data.values
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    dates = data.index.values
    return train, test, scaler, dates, train_size

def create_dataset(dataset, dates, look_back=60):
    X, Y, dates_new = [], [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])  # Predicting 'Close' price
        dates_new.append(dates[i + look_back])
    return np.array(X), np.array(Y), np.array(dates_new)
