# model_evaluation.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import mlflow
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, Y_test, scaler, ticker, dates):
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(np.hstack((test_predict, np.zeros((test_predict.shape[0], 1)))))
    Y_test = scaler.inverse_transform(np.hstack((Y_test.reshape(-1,1), np.zeros((Y_test.shape[0], 1)))))

    test_mse = mean_squared_error(Y_test[:,0], test_predict[:,0])
    print(f'Test Mean Squared Error: {test_mse}')

    # Log metrics
    mlflow.log_metrics({
        'test_mse': test_mse,
    })

    plt.figure(figsize=(16,8))
    plt.plot(dates, Y_test[:,0], label='Actual')
    plt.plot(dates, test_predict[:,0], label='Predicted')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.title(f'MSE Test: {test_mse}')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.savefig(f'Plots/{ticker}_predictions_vs_actuals.png')
    plt.close()
