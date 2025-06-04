import pandas as pd
import numpy as np
from terenceModel import FeatureWeightedDNN
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
    price = pd.read_csv('price_window.csv', encoding='latin1')
    
    print("\nInitial price data columns:")
    print(price.columns.tolist())
    print("\nFirst few rows of price data:")
    print(price.head())
    
    # Convert dates
    prod_weekly['ï»¿Date'] = pd.to_datetime(prod_weekly['ï»¿Date'], format='%b %d, %Y', errors='coerce')
    net_import_weekly['ï»¿Date'] = pd.to_datetime(net_import_weekly['ï»¿Date'], format='%b %d, %Y', errors='coerce')
    supply_weekly['Release Date'] = pd.to_datetime(supply_weekly['Release Date'], format='%d-%b-%y', errors='coerce')
    price['Date'] = pd.to_datetime(price['Date'], errors='coerce')
    
    # Convert supply values from strings with M suffix to numbers
    supply_weekly['Actual'] = supply_weekly['Actual'].apply(convert_to_number)
    supply_weekly['Forecast'] = supply_weekly['Forecast'].apply(convert_to_number)
    supply_weekly['Previous'] = supply_weekly['Previous'].apply(convert_to_number)
    
    # Filter date range for weekly data
    start_date = pd.to_datetime('2012-01-01')
    end_date = pd.to_datetime('2025-01-01')
    
    prod_weekly = prod_weekly[(prod_weekly['ï»¿Date'] >= start_date) & (prod_weekly['ï»¿Date'] < end_date)]
    net_import_weekly = net_import_weekly[(net_import_weekly['ï»¿Date'] >= start_date) & (net_import_weekly['ï»¿Date'] < end_date)]
    supply_weekly = supply_weekly[(supply_weekly['Release Date'] >= start_date) & (supply_weekly['Release Date'] < end_date)]
    
    return prod_weekly, net_import_weekly, supply_weekly, price

def prepare_supervised_data(price, weekly_production, weekly_import, scaler, prod_weekly, net_import_weekly):
    grouped_prices = price.groupby('Release_Datetime')
    X = []
    y = []

    # Convert weekly data date columns to datetime
    prod_weekly['ï»¿Date'] = pd.to_datetime(prod_weekly['ï»¿Date'])
    net_import_weekly['ï»¿Date'] = pd.to_datetime(net_import_weekly['ï»¿Date'])

    for report_time, group in grouped_prices:
        group = group.sort_values('Datetime')
        group_datetimes = pd.to_datetime(group['Datetime'])
        report_time_naive = pd.to_datetime(report_time).tz_localize(None)

        pre_release = group[group_datetimes <= report_time_naive].tail(60)
        # Get the price exactly 2 minutes after release
        post_release = group[group_datetimes > report_time_naive].head(2)

        if len(pre_release) == 60 and len(post_release) == 2:
            pre_release_scaled = scaler.transform(pre_release['Close'].values.reshape(-1, 1)).flatten()
            # Target: price at exactly 2 minutes after release
            target_price = post_release['Close'].values[-1]
            target_price_scaled = scaler.transform([[target_price]])[0, 0]

            # Use only the date part for matching
            report_date = report_time_naive.date()
            production_value = weekly_production[prod_weekly['ï»¿Date'].dt.date == report_date]
            import_value = weekly_import[net_import_weekly['ï»¿Date'].dt.date == report_date]
            production_value = production_value[0] if len(production_value) > 0 else 0
            import_value = import_value[0] if len(import_value) > 0 else 0

            X.append(np.concatenate([pre_release_scaled, [production_value, import_value]]))
            y.append(target_price_scaled)

    return np.array(X), np.array(y)

def plot_predictions(predictions, actuals, scaler):
    """
    Plot predictions vs actual values
    """
    # Inverse transform the scaled values
    predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals_original = scaler.inverse_transform(actuals.reshape(-1, 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(actuals_original, label='Actual')
    plt.plot(predictions_original, label='Predicted')
    plt.title('Walk-Forward Validation: Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()

def main():
    # Load and prepare data
    prod_weekly, net_import_weekly, supply_weekly, price = load_and_prepare_data()
    
    # Scale features
    scaler = StandardScaler()
    weekly_production_scaled = scaler.fit_transform(prod_weekly['US Weekly Production'].values.reshape(-1, 1)).flatten()
    weekly_import_scaled = scaler.fit_transform(net_import_weekly['Weekly Net Import'].values.reshape(-1, 1)).flatten()
    
    # Prepare supervised data
    X, y = prepare_supervised_data(
        price, 
        weekly_production_scaled, 
        weekly_import_scaled,
        scaler,
        prod_weekly,
        net_import_weekly
    )
    
    # Time-based 80/20 split
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize model
    model = FeatureWeightedDNN()
    
    # Train model
    trained_model, history = model.train(X_train, y_train)
    
    # Evaluate model
    test_loss, test_mae = trained_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test MAE: {test_mae:.2f}')
    
    # Save model
    trained_model.save('crude_oil_price_model.h5')

if __name__ == "__main__":
    main() 