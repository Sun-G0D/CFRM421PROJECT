import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from terenceModel import DNN
import numpy as np
from sklearn.preprocessing import StandardScaler
from terenceModel import RMSE



class DNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape=(7,)):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        
        # Tune number of layers
        n_layers = hp.Int('n_layers', 1, 4)
        
        # First layer
        model.add(layers.Dense(
            units=hp.Int('units_1', min_value=16, max_value=128, step=16),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(layers.Dropout(
            rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
        ))
        
        # Additional layers
        for i in range(1, n_layers):
            model.add(layers.Dense(
                units=hp.Int(f'units_{i+1}', min_value=8, max_value=64, step=8),
                activation='relu'
            ))
            model.add(layers.Dropout(
                rate=hp.Float(f'dropout_{i+1}', min_value=0.0, max_value=0.5, step=0.1)
            ))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Tune learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=[RMSE()]
        )
        
        return model
    def reset_session(self, seed=42):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        tf.keras.backend.clear_session()

def tune_model(X_train, y_train, X_val, y_val, max_trials=200, epochs=100):
    """
    Tune hyperparameters for the DNN model
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_val, y_val : validation data
    max_trials : maximum number of hyperparameter combinations to try
    epochs : maximum number of epochs for each trial
    """
    
    # Compute sample weights for training and validation
    epsilon = 1e-6
    sample_weights_train = np.abs(y_train) + epsilon
    sample_weights_val = np.abs(y_val) + epsilon

    # Create the hypermodel
    hypermodel = DNNHyperModel(input_shape=(X_train.shape[1],))
    
    hypermodel.reset_session()
    # Define the tuner
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective=kt.Objective("val_rmse", direction="min"),
        max_trials=max_trials,
        directory='tuner_results',
        project_name='dnn_tuning'
    )
    
    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_rmse',
        patience=10,
        restore_best_weights=True,
        mode='min'
    )
    
    # Start the search
    print("Starting hyperparameter search...")
    tuner.search(
        X_train, y_train,
        sample_weight=sample_weights_train,
        validation_data=(X_val, y_val, sample_weights_val),
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    
    print("\nBest Hyperparameters:")
    print(f"Number of layers: {best_hps.get('n_layers')}")
    print(f"Learning rate: {best_hps.get('learning_rate'):.6f}")
    print("\nLayer Configurations:")
    for i in range(best_hps.get('n_layers')):
        print(f"Layer {i+1}:")
        print(f"  Units: {best_hps.get(f'units_{i+1}')}")
        print(f"  Dropout: {best_hps.get(f'dropout_{i+1}'):.2f}")
    
    print(f"\nBest Validation Weighted RMSE: {best_trial.score:.6f}")
    
    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)
    
    return best_model, best_hps

def main():
    # Load and prepare your data here
    # This should match your data preparation in terenceTrainModel.py
    from terenceTrainModel import load_and_prepare_data
    
    # Load data
    prod_weekly, net_import_weekly, supply_weekly, price_wide, y_temp = load_and_prepare_data()
    
    price_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Create a copy of the price features to avoid SettingWithCopyWarning
    price_features = price_wide[['Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0']].copy()

    # Scale the price features in the dataframe
    for col in ['Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0']:
        price_features.loc[:, col] = price_scaler.fit_transform(price_wide[col].values.reshape(-1, 1)).flatten()

    # Scale the target values in the dataframe
    y_temp = target_scaler.fit_transform(y_temp.values.reshape(-1, 1)).flatten()

    #Scaler for weekly data
    weekly_scaler = StandardScaler()

    weekly_production_scaled = weekly_scaler.fit_transform(prod_weekly['Weekly Production'].values.reshape(-1, 1)).flatten()
    weekly_import_scaled = weekly_scaler.fit_transform(net_import_weekly['Weekly Net Import'].values.reshape(-1, 1)).flatten()
    weekly_supply_scaled = weekly_scaler.fit_transform(supply_weekly['Actual'].values.reshape(-1, 1)).flatten()

    X = []
    y = y_temp

    for idx, _ in price_features.iterrows():
        # Target: price of future 2 minutes after release (already scaled)
        
        production_value = weekly_production_scaled[idx]
        import_value = weekly_import_scaled[idx]
        supply_value = weekly_supply_scaled[idx]

        row_data = [price_features['Close_t-60'].values[idx],
                   price_features['Close_t-40'].values[idx],
                   price_features['Close_t-20'].values[idx],
                   price_features['Open_t0'].values[idx],
                   production_value,
                   import_value,
                   supply_value]
        X.append(row_data)

    X = np.array(X)
    y = np.array(y)
    
    # Split data into train and validation sets
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Tune the model
    best_model, best_hps = tune_model(X_train, y_train, X_val, y_val)
    print(best_hps)
    # Train the model with the best parameters
    best_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
    print(best_model.evaluate(X_val, y_val))


if __name__ == "__main__":
    main() 