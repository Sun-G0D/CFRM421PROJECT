import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class FeatureWeightedDNN:
    def __init__(self, input_dim=4):
        self.input_dim = input_dim
        # Feature importance weights based on our analysis
        self.feature_weights = {
            'weekly_supply': 0.4,      # Most important
            'pre_release_price': 0.3,   # Second most important
            'weekly_production': 0.2,   # Third most important
            'weekly_import': 0.1        # Least important
        }
        
    def build_model(self):
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))
        
        # Apply feature weights
        weighted_inputs = layers.Lambda(
            lambda x: tf.multiply(x, list(self.feature_weights.values()))
        )(inputs)
        
        # Hidden layers
        x = layers.Dense(64, activation='relu')(weighted_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, weekly_supply, pre_release_price, weekly_production, weekly_import):
        """
        Prepare input data with proper feature ordering based on importance
        """
        # Stack features in order of importance
        X = np.column_stack([
            weekly_supply,          # Most important
            pre_release_price,      # Second most important
            weekly_production,      # Third most important
            weekly_import           # Least important
        ])
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