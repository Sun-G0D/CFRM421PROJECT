import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class FeatureWeightedDNN:
    def __init__(self, input_dim=61):  # 60 minutes + 3 weekly features
        self.input_dim = input_dim
        # Feature importance weights based on our analysis
        self.feature_weights = {
            'weekly_supply': 0.4,      # Most important
            'weekly_production': 0.3,   # Second most important
            'weekly_import': 0.3        # Third most important
        }
        
    def build_model(self):
        # Input layer for time series data
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Reshape input to separate time series and weekly features
        time_series = layers.Lambda(lambda x: x[:, :-3])(inputs)  # Last 60 minutes
        weekly_features = layers.Lambda(lambda x: x[:, -3:])(inputs)  # 3 weekly features
        
        # Process time series with LSTM
        time_series = layers.Reshape((60, 1))(time_series)
        time_series = layers.LSTM(32, return_sequences=True)(time_series)
        time_series = layers.LSTM(16)(time_series)
        
        # Process weekly features with feature weights
        weighted_features = layers.Lambda(
            lambda x: tf.multiply(x, list(self.feature_weights.values()))
        )(weekly_features)
        
        # Combine time series and weekly features
        combined = layers.Concatenate()([time_series, weighted_features])
        
        # Hidden layers
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer (predicting 2 minutes)
        outputs = layers.Dense(2)(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, price_series, weekly_supply, weekly_production, weekly_import):
        """
        Prepare input data with proper feature ordering based on importance
        price_series: array of shape (n_samples, 60) containing last 60 minutes of prices
        """
        # Stack weekly features in order of importance
        weekly_features = np.column_stack([
            weekly_supply,          # Most important
            weekly_production,      # Second most important
            weekly_import           # Third most important
        ])
        
        # Combine time series and weekly features
        X = np.column_stack([price_series, weekly_features])
        return X
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the model with early stopping
        """
        model = self.build_model()
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return model, history
    
    def predict(self, model, X):
        """
        Make predictions using the trained model
        """
        return model.predict(X) 