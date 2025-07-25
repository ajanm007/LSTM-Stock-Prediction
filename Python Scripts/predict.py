import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('results/prediction_plot.png')
    plt.close()

if __name__ == "__main__":
    import joblib
    
    # Load data and model
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    scaler = joblib.load("models/scaler.save")
    model = load_model("models/best_model.h5")
    
    # Create results directory
    import os
    os.makedirs("results", exist_ok=True)
    
    # Evaluate and plot
    evaluate_model(model, X_test, y_test, scaler)
