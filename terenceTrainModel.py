import pandas as pd
import numpy as np
from terenceModel import DNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def convert_to_number(value):
    """Convert string with M suffix to float"""
    if isinstance(value, str):
        return float(value.replace('M', '')) * 1000000
    return value

def load_and_prepare_data():
    # Load data
    prod_weekly = pd.read_csv('domestic_prod.csv', encoding='latin1')
    net_import_weekly = pd.read_csv('net_import.csv', encoding='latin1')
    supply_weekly = pd.read_csv('InvestingcomEIA.csv', encoding='latin1')
    price_wide = pd.read_csv('price_window_valid_wide.csv', encoding='latin1') 
    
    # Converting dates
    prod_weekly['ï»¿Date'] = pd.to_datetime(prod_weekly['ï»¿Date'], format='%b %d, %Y')
    net_import_weekly['ï»¿Date'] = pd.to_datetime(net_import_weekly['ï»¿Date'], format='%b %d, %Y')
    supply_weekly['Release Date'] = pd.to_datetime(supply_weekly['Release Date'], format='%d-%b-%y')
    price_wide['Release_Datetime'] = pd.to_datetime(price_wide['Release_Datetime']).dt.tz_localize(None)
    
    # Convert M to one million
    supply_weekly['Actual'] = supply_weekly['Actual'].apply(convert_to_number)
    supply_weekly['Forecast'] = supply_weekly['Forecast'].apply(convert_to_number)
    supply_weekly['Previous'] = supply_weekly['Previous'].apply(convert_to_number)
    
    # Filter date range for weekly data and price_wide data
    start_date = pd.to_datetime('2012-01-01')
    end_date = pd.to_datetime('2025-01-01')
    prod_weekly = prod_weekly[(prod_weekly['ï»¿Date'] >= start_date) & (prod_weekly['ï»¿Date'] < end_date)]
    net_import_weekly = net_import_weekly[(net_import_weekly['ï»¿Date'] >= start_date) & (net_import_weekly['ï»¿Date'] < end_date)]
    supply_weekly = supply_weekly[(supply_weekly['Release Date'] >= start_date) & (supply_weekly['Release Date'] < end_date)]
    price_wide = price_wide[(price_wide['Release_Datetime'] >= start_date) & (price_wide['Release_Datetime'] < end_date)]
    
    return prod_weekly, net_import_weekly, supply_weekly, price_wide

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
    mae = mean_absolute_error(actuals_unscaled, predictions_unscaled)
    print(f"Test MAE (unscaled): {mae:.2f}")

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
    prod_weekly, net_import_weekly, supply_weekly, price_wide = load_and_prepare_data()
    
    # For features (X): all Close prices used for features
    all_feature_prices = []
    all_target_prices = []
    for idx, row in price_wide.iterrows():
        all_feature_prices.extend([row['Close_t-60'], row['Close_t-40'], row['Close_t-20'], row['Close_t0']])
        all_target_prices.append(row['Close_t2'])
    all_feature_prices = np.array(all_feature_prices)
    all_target_prices = np.array(all_target_prices)

    price_scaler = StandardScaler()

    target_scaler = StandardScaler()

    # Fit feature scaler on all price data used for features
    price_scaler.fit(all_feature_prices.reshape(-1, 1))
    # Fit target scaler on all target prices
    target_scaler.fit(all_target_prices.reshape(-1, 1))


    # Scale weekly features
    weekly_features = np.concatenate([
        prod_weekly['US Weekly Production'].values,
        net_import_weekly['Weekly Net Import'].values,
        supply_weekly['Actual'].values
    ]).reshape(-1, 1)

    #Scaler for weekly data
    weekly_scaler = StandardScaler()

    weekly_scaler.fit(weekly_features)
    
    weekly_production_scaled = weekly_scaler.transform(prod_weekly['US Weekly Production'].values.reshape(-1, 1)).flatten()
    weekly_import_scaled = weekly_scaler.transform(net_import_weekly['Weekly Net Import'].values.reshape(-1, 1)).flatten()
    weekly_supply_scaled = weekly_scaler.transform(supply_weekly['Actual'].values.reshape(-1, 1)).flatten()
    
    # Prepare supervised data
    X, y = prepare_supervised_data_wide(
        price_wide, 
        weekly_production_scaled, 
        weekly_import_scaled,
        weekly_supply_scaled,
        price_scaler,
        target_scaler,
        prod_weekly,
        net_import_weekly,
        supply_weekly
    )
    # Time-based 80/20 split
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
 
    # Initialize model
    model = DNN()
    
    # Train model
    trained_model, _ = model.train(X_train, y_train)
    
    # Evaluate model
    test_loss, test_mae = trained_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.2f}")
    print(f'Test MAE: {test_mae:.2f}')
    
    # Predict on test set
    y_pred = trained_model.predict(X_test).flatten()

    # Plot and print MAE (unscaled)
    plot_predictions(y_pred, y_test, target_scaler)

    trained_model.save('crude_oil_price_model.h5')

    
if __name__ == "__main__":
    main() 