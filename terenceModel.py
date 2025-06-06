import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.mse = tf.keras.metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mse.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return tf.sqrt(self.mse.result())

    def reset_state(self):
        self.mse.reset_state()

class DNN:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = None
        self.build_model()

    def build_model(self):
        """
        Build the model architecture
        input_shape: (7,) for 4 price features + 3 weekly features
        """
        self.reset_session()

        inputs = layers.Input(shape=(7,))
        
        price_features = inputs[:, :4]  # First 4 features are price features
        weekly_features = inputs[:, 4:]  # Last 3 features are weekly data
        
        price_dense = layers.Dense(16, activation='relu')(price_features)
        price_dense = layers.Dropout(0.2)(price_dense)
        
        weekly_dense = layers.Dense(16, activation='relu')(weekly_features)
        weekly_dense = layers.Dropout(0.2)(weekly_dense)
        
        # Combine price and weekly features
        combined = layers.Concatenate()([price_dense, weekly_dense])
        
        # Output
        dense1 = layers.Dense(32, activation='relu')(combined)
        dense1 = layers.Dropout(0.2)(dense1)
        outputs = layers.Dense(1)(dense1)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='mse',
                     metrics=[RMSE()])
        
        self.model = model

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
        return self.model.predict(X, verbose=0)

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
        weekly_features = np.column_stack([
            weekly_production,
            weekly_import,
            weekly_supply
        ])
        X = np.column_stack([price_features, weekly_features])
        return X 