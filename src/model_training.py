# In model_training.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("stock-price-prediction")

def create_and_train_model(X_train, Y_train, model_path):
    mlflow.set_experiment("stock-price-prediction")
    with mlflow.start_run() as run:
        model = Sequential()
        model.add(LSTM(1000, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.3))
        model.add(LSTM(500, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(100))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam(0.001))

        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        history = model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=2, validation_split=0.2, callbacks=[early_stop])
        model.save(model_path)

        # Log parameters
        mlflow.log_params({
            'optimizer': 'Adam',
            'loss_function': 'mean_squared_error',
            'batch_size': 64,
            'epochs': 50,
            'patience': 20,
            'validation_split': 0.2
        })

        # Log metrics
        mlflow.log_metrics({
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
        })
        print(mlflow.keras)  # This should print out the keras module in mlflow, or raise an error if it doesn't exist
        # Log model
        mlflow.keras.log_model(model, "model")

    return model, run.info.run_id
