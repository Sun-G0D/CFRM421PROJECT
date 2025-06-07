import pandas as pd
import numpy as np
from terenceWlessModel import DNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

def plot_predictions(predictions, actuals, save_path='terenceActualVSPredictedWless.png'):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"Test RMSE: {rmse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual % Change')
    plt.plot(predictions, label='Predicted % Change')
    plt.title('Actual vs Predicted Percentage Change 2 Minutes After Release (No Sample Weights)')
    plt.xlabel('Test Sample')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.savefig(save_path)
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

    model = DNN()
    
    # Train model without sample weights
    trained_model, _ = model.train(X_train, y_train)

    # Predict on test set
    y_pred = trained_model.predict(X_test).flatten()

    # Regular RMSE for test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.4f}")

    # Baseline: predict zero change
    baseline_pred_zero = np.zeros_like(y_test)
    baseline_rmse_zero = np.sqrt(mean_squared_error(y_test, baseline_pred_zero))
    print(f"Baseline (Zero Prediction) RMSE: {baseline_rmse_zero:.4f}")

    if test_rmse < baseline_rmse_zero:
        print("Model outperforms baseline (predicting zero change).")
    else:
        print("Model does NOT outperform baseline (predicting zero change). Further investigation or model refinement needed.")

    plot_predictions(y_pred, y_test)

    
if __name__ == "__main__":
    main() 