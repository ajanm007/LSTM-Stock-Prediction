import numpy as np
from model import build_model, train_model
import os

def load_processed_data():
    """Load preprocessed data"""
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    X_train, _, y_train, _ = load_processed_data()
    model = build_model((X_train.shape[1], 1))
    history = train_model(model, X_train, y_train)
    
    print("Training completed. Model saved to models/best_model.h5")
