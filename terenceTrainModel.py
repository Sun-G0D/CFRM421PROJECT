import pandas as pd
import numpy as np
from terenceModel import DNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf

def convert_to_number(value):
    """Convert string with M suffix to float"""
    if isinstance(value, str):
        return float(value.replace('M', '')) * 1000000
    return value

def load_and_prepare_data():
    # Load data
    df = pd.read_csv("full_data.csv")

    feature_cols = [col for col in df.columns if 'Close_t-60'  in col or 'Close_t-40' in col or 'Close_t-20' in col or col == 'Release Date' or col == 'Actual' or col == 'Weekly Net Import' or col == 'Weekly Production' or col == 'Open_t0']

    X_temp = df[feature_cols]
    y_temp = (df['Close_t2'] - df['Close_t0'])/df['Close_t0']

    prod_weekly = X_temp[['Release Date', 'Weekly Production']]
    net_import_weekly = X_temp[['Release Date', 'Weekly Net Import']]
    supply_weekly = X_temp[['Release Date', 'Actual']]
    price_wide = X_temp[['Release Date', 'Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0']]
    
    return prod_weekly, net_import_weekly, supply_weekly, price_wide, y_temp

def plot_predictions(predictions, actuals, save_path='terenceActualVSPredicted.png'):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"Test RMSE: {rmse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual % Change')
    plt.plot(predictions, label='Predicted % Change')
    plt.title('Actual vs Predicted Percentage Change 2 Minutes After Release')
    plt.xlabel('Test Sample')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def compute_feature_importance(model, X_test):
    """
    Compute feature importance using gradients.
    Returns the average absolute gradient for each feature.
    """
    # Convert to tensor
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    # Enable gradient computation
    with tf.GradientTape() as tape:
        tape.watch(X_test_tensor)
        predictions = model(X_test_tensor)
    
    # Compute gradients
    gradients = tape.gradient(predictions, X_test_tensor)
    
    # Take absolute value and average across samples
    feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)
    
    return feature_importance.numpy()

def plot_feature_importance(feature_importance, feature_names):
    """
    Plot feature importance as a bar chart.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance Based on Gradients')
    plt.xlabel('Features')
    plt.ylabel('Average Absolute Gradient')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def main():
    prod_weekly, net_import_weekly, supply_weekly, price_wide, y_temp = load_and_prepare_data()

    price_scaler = StandardScaler()
    target_scaler = StandardScaler()

    price_features = price_wide[['Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0']]

    # Scale the price features in the dataframe
    for col in ['Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0']:
        price_features[col] = price_scaler.fit_transform(price_wide[col].values.reshape(-1, 1)).flatten()

    # Scale the target values in the dataframe
    y_temp = target_scaler.fit_transform(y_temp.values.reshape(-1, 1)).flatten()

    #Scaler for weekly data
    weekly_scaler = StandardScaler()

    weekly_production_scaled = weekly_scaler.fit_transform(prod_weekly['Weekly Production'].values.reshape(-1, 1)).flatten()
    weekly_import_scaled = weekly_scaler.fit_transform(net_import_weekly['Weekly Net Import'].values.reshape(-1, 1)).flatten()
    weekly_supply_scaled = weekly_scaler.fit_transform(supply_weekly['Actual'].values.reshape(-1, 1)).flatten()

    X = []
    y = []

    for idx, _ in price_features.iterrows():
        # Target: price of future 2 minutes after release (already scaled)
        target_price = y_temp[idx]
        production_value = weekly_production_scaled[idx]
        import_value = weekly_import_scaled[idx]
        supply_value = weekly_supply_scaled[idx]
        row_data = [price_features['Close_t-60'].values[idx],price_features['Close_t-40'].values[idx],price_features['Close_t-20'].values[idx],price_features['Open_t0'].values[idx],production_value,import_value,supply_value]
        X.append(row_data)
        y.append(target_price)

    X = np.array(X)
    y = np.array(y)

    # Time-based 80/20 split
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Compute sample weights for training and test sets
    epsilon = 1e-6
    sample_weights_train = np.abs(y_train) + epsilon
    sample_weights_test = np.abs(y_test) + epsilon

    model = DNN()
    
    # Train model with sample weights
    trained_model, _ = model.train(X_train, y_train, sample_weight=sample_weights_train)

    # Predict on test set
    y_pred = trained_model.predict(X_test).flatten()

    # Compute feature importance
    feature_names = ['Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0', 
                    'Weekly Production', 'Weekly Net Import', 'Actual']
    feature_importance = compute_feature_importance(trained_model, X_test)
    
    # Print feature importance
    print("\nFeature Importance (based on gradients):")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.6f}")
    
    # Plot feature importance
    plot_feature_importance(feature_importance, feature_names)

    # Weighted RMSE for test set
    weighted_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=sample_weights_test))
    print(f"\nWeighted Test RMSE: {weighted_test_rmse:.4f}")

    # Baseline: predict zero change
    baseline_pred_zero = np.zeros_like(y_test)
    baseline_rmse_zero = np.sqrt(mean_squared_error(y_test, baseline_pred_zero, sample_weight=sample_weights_test))
    print(f"Baseline (Zero Prediction) Weighted RMSE: {baseline_rmse_zero:.4f}")

    if weighted_test_rmse < baseline_rmse_zero:
        print("Model outperforms baseline (predicting zero change).")
    else:
        print("Model does NOT outperform baseline (predicting zero change). Further investigation or model refinement needed.")

    plot_predictions(y_pred, y_test)

    
if __name__ == "__main__":
    main() 