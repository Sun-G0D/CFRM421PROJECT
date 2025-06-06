import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class DNN:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = None
        self.feature_weights = None
        self.scaler = None

    def build_model(self, input_shape=(7,)):
        """
        Build the model architecture
        input_shape: (7,) for 4 price features + 3 weekly features
        """
        self.reset_session()
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Split input into price and weekly features
        price_features = inputs[:, :4]  # First 4 features are price features
        weekly_features = inputs[:, 4:]  # Last 3 features are weekly data
        
        # Dense layers for price features
        price_dense = Dense(16, activation='relu')(price_features)
        price_dense = Dropout(0.2)(price_dense)
        
        # Dense layers for weekly features
        weekly_dense = Dense(16, activation='relu')(weekly_features)
        weekly_dense = Dropout(0.2)(weekly_dense)
        
        # Combine price and weekly features
        combined = Concatenate()([price_dense, weekly_dense])
        
        # Output layers
        dense1 = Dense(32, activation='relu')(combined)
        dense1 = Dropout(0.2)(dense1)
        outputs = Dense(1)(dense1)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='mse',
                     metrics=['mae'])
        
        return model

    def reset_session(self, seed=42):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        tf.keras.backend.clear_session()

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the model
        X: input features (4 price features + 3 weekly features)
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
        X: input features (4 price features + 3 weekly features)
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

    def prepare_data(self, price_features, weekly_production, weekly_import, weekly_supply):
        """
        Prepare input data with proper feature ordering
        price_features: array of shape (n_samples, 4) containing selected price features
        weekly_production, weekly_import, weekly_supply: arrays of shape (n_samples,)
        """
        # Stack weekly features
        weekly_features = np.column_stack([
            weekly_production,
            weekly_import,
            weekly_supply
        ])
        # Combine price and weekly features
        X = np.column_stack([price_features, weekly_features])
        return X 