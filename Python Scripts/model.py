from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def build_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=25, batch_size=32):
    """Train the model"""
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[checkpoint]
    )
    return history

if __name__ == "__main__":
    # Example usage
    model = build_model((60, 1))
    model.summary()
