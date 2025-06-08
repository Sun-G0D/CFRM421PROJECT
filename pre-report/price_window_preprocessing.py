import pandas as pd
from datetime import datetime
import datetime
import pytz
import re
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go

min_WTI = pd.read_csv(
    'data/cl-1m.csv',
    sep=';',
    header=None,
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
)

icom_eia_forecasts = pd.read_csv('InvestingcomEIA.csv')

def standardize_eia_datetime(row):
    """Standardizes datetime for EIA forecast"""
    release_date = pd.to_datetime(row['Release Date'], format='%d-%b-%y', errors='coerce').date()
    release_time = pd.to_datetime(row['Time'], format='%H:%M', errors='coerce').time()

    if pd.isna(release_date) or pd.isna(release_time):
        print('nan')
        return pd.NaT

    combined_datetime = pd.Timestamp.combine(release_date, release_time)
    return combined_datetime

icom_eia_forecasts['Release_Datetime'] = icom_eia_forecasts.apply(standardize_eia_datetime, axis=1)

eastern_tz = pytz.timezone('US/Eastern')
icom_eia_forecasts['Release_Datetime_EST'] = icom_eia_forecasts['Release_Datetime'].dt.tz_localize(eastern_tz, ambiguous='infer', nonexistent='shift_forward')

chicago_tz = pytz.timezone('America/Chicago')
icom_eia_forecasts['Release_Datetime_CST'] = icom_eia_forecasts['Release_Datetime_EST'].dt.tz_convert(chicago_tz)


def standardize_wti_datetime(row):
    """Standardizes datetime for min_WTI dataframe."""
    date_str = row['Date']
    time_str = row['Time']

    combined_datetime = pd.to_datetime(date_str + ' ' + time_str, format='%d/%m/%Y %H:%M:%S', errors='coerce')

    if pd.isna(combined_datetime):
        print('nan')
        return pd.NaT
    return combined_datetime

min_WTI['Datetime'] = min_WTI.apply(standardize_wti_datetime, axis=1)
min_WTI['Datetime_CST'] = min_WTI['Datetime'].dt.tz_localize(chicago_tz, ambiguous='infer', nonexistent='shift_forward')

icom_eia_forecasts = icom_eia_forecasts.set_index('Release_Datetime_CST').sort_index()
min_WTI = min_WTI.set_index('Datetime_CST').sort_index()

min_WTI = min_WTI[~min_WTI.index.duplicated(keep='first')]


# filling in timejumps with 0 activity
min_WTI = min_WTI.resample('min').asfreq()
min_WTI['Close'] = min_WTI['Close'].ffill()
min_WTI['Open'] = min_WTI['Open'].fillna(min_WTI['Close'])
min_WTI['High'] = min_WTI['High'].fillna(min_WTI['Close'])
min_WTI['Low'] = min_WTI['Low'].fillna(min_WTI['Close'])
min_WTI['Volume'] = min_WTI['Volume'].fillna(0)
min_WTI['Date'] = min_WTI['Date'].ffill()

min_WTI = min_WTI.reset_index()

min_WTI['Datetime_CST'] = pd.to_datetime(min_WTI['Datetime_CST'])
min_WTI['Time'] = min_WTI['Datetime_CST'].dt.strftime('%H:%M:%S')
min_WTI['Datetime'] = min_WTI['Datetime_CST'].dt.strftime('%Y-%m-%d %H:%M:%S')
percentage_price_changes_1min_wti = []
previous_close_price = None

for index, row in min_WTI.iterrows():
    current_close_price = row['Close']
    if previous_close_price is not None and previous_close_price != 0:
        percentage_change = ((current_close_price - previous_close_price) / previous_close_price) * 100.0
        percentage_price_changes_1min_wti.append(percentage_change)
    else:
        percentage_price_changes_1min_wti.append(float('nan'))
    previous_close_price = current_close_price

percentage_price_changes_1min_wti_series = pd.Series(percentage_price_changes_1min_wti, index=min_WTI.index)
min_WTI['Percent_Change'] = percentage_price_changes_1min_wti_series
min_WTI.set_index('Datetime_CST', inplace=True)





def get_price_windows(eia_release_times, price_data, window_minutes_before=60, window_minutes_after=60):
    """
    Extracts price data windows around EIA report release times.

    Args:
        eia_release_times (pd.DatetimeIndex): Index of icom_eia_forecasts (release datetimes).
        price_data (pd.DataFrame): min_res_OIH dataframe with Datetime index.
        window_minutes_before (int): Minutes to include before release time.
        window_minutes_after (int): Minutes to include after release time.

    Returns:
        pd.DataFrame: A DataFrame containing price data for all events, within the specified windows.
                      Returns an empty DataFrame if no data is found within any window.
    """
    price_windows_list = []

    for release_time in eia_release_times:
        start_time = release_time - pd.Timedelta(minutes=window_minutes_before)

        end_time = release_time + pd.Timedelta(minutes=window_minutes_after)

        window_data = price_data.loc[start_time:end_time].copy()


        if not window_data.empty:
            window_data['Release_Datetime'] = release_time
            price_windows_list.append(window_data)

    if price_windows_list:
        price_windows_df = pd.concat(price_windows_list)
        return price_windows_df
    else:
        return pd.DataFrame()

price_window_60min = get_price_windows(icom_eia_forecasts.index, min_WTI, window_minutes_before=60, window_minutes_after=60)

price_window_60min = price_window_60min.reset_index()

price_window_60min['Time_to_Release_Minutes'] = (price_window_60min['Datetime_CST'] - price_window_60min['Release_Datetime']).dt.total_seconds() / 60