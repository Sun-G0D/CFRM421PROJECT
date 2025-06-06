import pandas as pd
import numpy as np
from terenceModel import DNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
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
    y_temp = df['Close_t2']

    prod_weekly = X_temp[['Release Date', 'Weekly Production']]
    net_import_weekly = X_temp[['Release Date', 'Weekly Net Import']]
    supply_weekly = X_temp[['Release Date', 'Actual']]
    price_wide = X_temp[['Release Date', 'Close_t-60', 'Close_t-40', 'Close_t-20', 'Open_t0']]
    
    return prod_weekly, net_import_weekly, supply_weekly, price_wide, y_temp

def prepare_supervised_data_wide(price_wide, weekly_production, weekly_import, weekly_supply, scaler, target_scaler, prod_weekly, net_import_weekly, supply_weekly):
    X = []
    y = []

    # Convert weekly data date columns to datetime
    prod_weekly['ï»¿Date'] = pd.to_datetime(prod_weekly['ï»¿Date'])
    net_import_weekly['ï»¿Date'] = pd.to_datetime(net_import_weekly['ï»¿Date'])
    supply_weekly['Release Date'] = pd.to_datetime(supply_weekly['Release Date'])
    price_wide['Release_Datetime'] = pd.to_datetime(price_wide['Release_Datetime'])

    for idx, row in price_wide.iterrows():
        # Use 4 price features: 60, 40, 20, 0 minutes before release
        price_features = [
            row['Close_t-60'],
            row['Close_t-40'],
            row['Close_t-20'],
            row['Close_t0']
        ]
        price_features_scaled = scaler.transform(np.array(price_features).reshape(-1, 1)).flatten()

        # Target: price of future 2 minutes after release
        target_price = row['Close_t2']
        target_price_scaled = target_scaler.transform([[target_price]])[0, 0]

        # Match weekly features by date
        report_date = row['Release_Datetime'].date()
        production_value = weekly_production[prod_weekly['ï»¿Date'].dt.date == report_date]
        import_value = weekly_import[net_import_weekly['ï»¿Date'].dt.date == report_date]
        supply_value = weekly_supply[supply_weekly['Release Date'].dt.date == report_date]
        production_value = production_value[0] if len(production_value) > 0 else 0
        import_value = import_value[0] if len(import_value) > 0 else 0
        supply_value = supply_value[0] if len(supply_value) > 0 else 0

        X.append(np.concatenate([price_features_scaled, [production_value, import_value, supply_value]]))
        y.append(target_price_scaled)
    return np.array(X), np.array(y)

def plot_predictions(predictions, actuals, scaler, save_path='terenceActualVSPredicted.png'):
    """
    Plot predictions vs actual values (unscaled) and print MAE.
    """
    # Inverse transform the predictions and actuals (We want to see the raw price comparison between actual and predicted from DNN)
    predictions_unscaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_unscaled = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Calculate MAE on unscaled data
    rmse = root_mean_squared_error(actuals_unscaled, predictions_unscaled)
    print(f"Test RMSE (unscaled): {rmse:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(actuals_unscaled, label='Actual Price (2 min after release)')
    plt.plot(predictions_unscaled, label='Predicted Price (2 min after release)')
    plt.title('Actual vs Predicted Price 2 Minutes After Release')
    plt.xlabel('Test Sample')
    plt.ylabel('Price')
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
        production_value = production_value
        import_value = import_value
        supply_value = supply_value

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
        
    # Train model
    trained_model, _ = model.train(X_train, y_train)

    # Evaluate model
    test_loss, test_rmse = trained_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.2f}")
    print(f'Test RMSE: {test_rmse:.2f}')

    # Predict on test set
    y_pred = trained_model.predict(X_test).flatten()

    # Plot and print MAE (unscaled)
    plot_predictions(y_pred, y_test, target_scaler)

    
if __name__ == "__main__":
    main() 