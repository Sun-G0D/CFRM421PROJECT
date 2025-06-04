import pandas as pd
import numpy as np
from terenceModel import FeatureWeightedDNN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
    print("\nSupply weekly data:")
    print(supply_weekly.head())
    
    print("\nPrice data:")
    print(price.head())
    print("\nNumber of price records:", len(price))
    
    return prod_weekly, net_import_weekly, supply_weekly, price

def prepare_features(prod_weekly, net_import_weekly, supply_weekly, price):
    # Extract features
    weekly_supply = supply_weekly['Actual'].values
    weekly_production = prod_weekly['US Weekly Production'].values
    weekly_import = net_import_weekly['Weekly Net Import'].values
    
    # Scale features
    scaler = StandardScaler()
    weekly_supply_scaled = scaler.fit_transform(weekly_supply.reshape(-1, 1)).flatten()
    weekly_production_scaled = scaler.fit_transform(weekly_production.reshape(-1, 1)).flatten()
    weekly_import_scaled = scaler.fit_transform(weekly_import.reshape(-1, 1)).flatten()
    
    # Calculate price changes
    price['price_change'] = price.groupby('report_time')['Close'].diff()
    
    # Get the price 1-2 minutes before release for each report
    pre_release_prices = []
    for report_time in price['report_time'].unique():
        report_prices = price[price['report_time'] == report_time]
        # Get the price 1-2 minutes before release
        pre_release_price = report_prices[report_prices['Date'] <= report_time].iloc[-2:]['Close'].mean()
        pre_release_prices.append(pre_release_price)
    
    # Scale pre-release prices
    pre_release_prices_scaled = scaler.fit_transform(np.array(pre_release_prices).reshape(-1, 1)).flatten()
    
    # Use the price changes as target variable
    y = price['price_change'].values
    
    return weekly_supply_scaled, weekly_production_scaled, weekly_import_scaled, pre_release_prices_scaled, y

def main():
    # Load and prepare data
    prod_weekly, net_import_weekly, supply_weekly, price = load_and_prepare_data()
    weekly_supply, weekly_production, weekly_import, pre_release_prices, y = prepare_features(
        prod_weekly, net_import_weekly, supply_weekly, price
    )
    
    # Initialize model
    model = FeatureWeightedDNN()
    
    # Prepare input data
    X = model.prepare_data(weekly_supply, pre_release_prices, weekly_production, weekly_import)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    trained_model, history = model.train(X_train, y_train)
    
    # Evaluate model
    test_loss, test_mae = trained_model.evaluate(X_test, y_test, verbose=0)
    print(f'Test MAE: {test_mae:.2f}')
    
    # Save model
    trained_model.save('crude_oil_price_model.h5')

if __name__ == "__main__":
    main() 