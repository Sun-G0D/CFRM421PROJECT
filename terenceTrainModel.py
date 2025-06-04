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

def prepare_walk_forward_data(price, weekly_supply, weekly_production, weekly_import, scaler):
    """
    Prepare data for walk-forward validation
    Returns sequences of 60 minutes for input and 2 minutes for output
    """
    # Group price data by release time
    grouped_prices = price.groupby('Release_Datetime')
    
    X_sequences = []
    y_sequences = []
    
    for report_time, group in grouped_prices:
        # Sort by date to ensure correct order
        group = group.sort_values('Date')
        
        # Get the last 60 minutes before release
        pre_release = group[group['Date'] <= report_time].tail(60)
        
        # Get the next 2 minutes after release
        post_release = group[group['Date'] > report_time].head(2)
        
        if len(pre_release) == 60 and len(post_release) == 2:
            # Scale the price sequences
            pre_release_scaled = scaler.transform(pre_release['Close'].values.reshape(-1, 1)).flatten()
            post_release_scaled = scaler.transform(post_release['Close'].values.reshape(-1, 1)).flatten()
            
            # Get the corresponding weekly features for this report time
            report_date = pd.to_datetime(report_time.date())
            supply_value = weekly_supply[supply_weekly['Release Date'] == report_date][0] if len(weekly_supply[supply_weekly['Release Date'] == report_date]) > 0 else 0
            production_value = weekly_production[prod_weekly['ï»¿Date'] == report_date][0] if len(weekly_production[prod_weekly['ï»¿Date'] == report_date]) > 0 else 0
            import_value = weekly_import[net_import_weekly['ï»¿Date'] == report_date][0] if len(weekly_import[net_import_weekly['ï»¿Date'] == report_date]) > 0 else 0
            
            # Combine price sequence with weekly features
            X_sequence = np.concatenate([pre_release_scaled, [supply_value, production_value, import_value]])
            
            X_sequences.append(X_sequence)
            y_sequences.append(post_release_scaled)
    
    return np.array(X_sequences), np.array(y_sequences)

def walk_forward_validation(X, y, model, train_size=0.8):
    """
    Perform walk-forward validation
    """
    n_samples = len(X)
    train_size = int(n_samples * train_size)
    
    # Initialize lists to store predictions and actual values
    all_predictions = []
    all_actuals = []
    
    # Train initial model on first train_size samples
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    trained_model, _ = model.train(X_train, y_train)
    
    # Walk forward
    for i in range(train_size, n_samples):
        # Make prediction
        X_test = X[i:i+1]
        y_test = y[i:i+1]
        
        prediction = model.predict(trained_model, X_test)
        
        # Store results
        all_predictions.append(prediction[0])
        all_actuals.append(y_test[0])
    
        # Update training data and retrain model
        X_train = np.vstack([X_train, X_test])
        y_train = np.vstack([y_train, y_test])
        
        trained_model, _ = model.train(X_train, y_train)
    
    return np.array(all_predictions), np.array(all_actuals)

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
    weekly_supply_scaled = scaler.fit_transform(supply_weekly['Actual'].values.reshape(-1, 1)).flatten()
    weekly_production_scaled = scaler.fit_transform(prod_weekly['US Weekly Production'].values.reshape(-1, 1)).flatten()
    weekly_import_scaled = scaler.fit_transform(net_import_weekly['Weekly Net Import'].values.reshape(-1, 1)).flatten()
    
    # Prepare walk-forward data
    X, y = prepare_walk_forward_data(
        price, 
        weekly_supply_scaled, 
        weekly_production_scaled, 
        weekly_import_scaled,
        scaler
    )
    
    # Initialize model
    model = FeatureWeightedDNN()
    
    # Perform walk-forward validation
    predictions, actuals = walk_forward_validation(X, y, model)
    
    # Calculate and print metrics
    mae = mean_absolute_error(actuals, predictions)
    print(f'Walk-Forward Validation MAE: {mae:.2f}')
    
    # Plot results
    plot_predictions(predictions, actuals, scaler)
    
    # Save the final model
    final_model, _ = model.train(X, y)
    final_model.save('crude_oil_price_model.h5')

if __name__ == "__main__":
    main() 