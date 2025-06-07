import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class RMSE(tf.keras.metrics.Metric):
    def __init__(self, name='rmse', **kwargs):
        super(RMSE, self).__init__(name=name, **kwargs)
        self.mse = tf.keras.metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mse.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return tf.sqrt(self.mse.result())

    def reset_state(self):
        self.mse.reset_state()

class DNN:
    def __init__(self, learning_rate=0.000187):  # Updated best learning rate
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self, input_shape=(7,)):
        """
        Build the DNN model with the best hyperparameters found from tuning
        Best configuration:
        - 2 layers: [32 units, 0.2 dropout], [8 units, 0.2 dropout]
        - Learning rate: 0.000187
        """
        model = Sequential([
            Dense(32, activation='relu', input_shape=input_shape),  # Layer 1
            Dropout(0.2),
            Dense(8, activation='relu'),  # Layer 2
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=[RMSE(), 'mae']
        )
        
        self.model = model
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
            self.model = self.build_model(input_shape=(X.shape[1],))
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_rmse',
                                     patience=10,
                                     restore_best_weights=True,
                                     mode='min')
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.model, history

    def save(self, filepath):
        """
        Save the model
        """
        if self.model is not None:
            self.model.save(filepath)
        else:
            raise ValueError("No model to save. Train the model first.") 