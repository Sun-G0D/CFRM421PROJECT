import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class FeatureWeightedDNN:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = None
        self.feature_weights = None
        self.scaler = None

    def build_model(self, input_shape=(62,)):
        """
        Build the model architecture
        input_shape: (62,) for 60 minutes of price data + 2 weekly features
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Split input into time series and weekly features
        time_series = inputs[:, :60]  # First 60 features are time series
        weekly_features = inputs[:, 60:]  # Last 2 features are weekly data
        
        # Reshape time series for LSTM
        time_series = Reshape((60, 1))(time_series)
        
        # LSTM layers for time series
        lstm1 = LSTM(64, return_sequences=True)(time_series)
        lstm1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(32)(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Dense layers for weekly features
        weekly_dense = Dense(16, activation='relu')(weekly_features)
        weekly_dense = Dropout(0.2)(weekly_dense)
        
        # Combine LSTM and weekly features
        combined = Concatenate()([lstm2, weekly_dense])
        
        # Output layers
        dense1 = Dense(32, activation='relu')(combined)
        dense1 = Dropout(0.2)(dense1)
        outputs = Dense(1)(dense1)  # Predict a single value
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='mse',
                     metrics=['mae'])
        
        return model

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the model
        X: input features (60 minutes + 2 weekly features)
        y: target values (single value)
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss',
                                     patience=10,
                                     restore_best_weights=True)
        
        # Train the model
        history = self.model.fit(X, y,
                               validation_split=validation_split,
                               epochs=epochs,
                               batch_size=batch_size,
                               callbacks=[early_stopping],
                               verbose=1)
        
        return self.model, history

    def predict(self, model, X):
        """
        Make predictions
        X: input features (60 minutes + 2 weekly features)
        """
        return model.predict(X)

    def save(self, filepath):
        """
        Save the model
        """
        if self.model is not None:
            self.model.save(filepath)
        else:
            raise ValueError("No model to save. Train the model first.")

    def prepare_data(self, price_series, weekly_production, weekly_import):
        """
        Prepare input data with proper feature ordering based on importance
        price_series: array of shape (n_samples, 60) containing last 60 minutes of prices
        """
        # Stack weekly features
        weekly_features = np.column_stack([
            weekly_production,
            weekly_import
        ])
        # Combine time series and weekly features
        X = np.column_stack([price_series, weekly_features])
        return X 