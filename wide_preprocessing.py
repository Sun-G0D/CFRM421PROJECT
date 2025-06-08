
valid_releases = price_window_60min.loc[price_window_60min['Time_to_Release_Minutes'] == 2]
price_window_60min = price_window_60min[price_window_60min['Date'].isin(valid_releases['Date'])]
price_window_60min = price_window_60min[price_window_60min['Time_to_Release_Minutes'] <= 2]

def pivot_market_data(df, x_minutes_before, cols_to_pivot=None):
    """
    pivot into wide formate where minutes to release from [-x, 2] are kept as columns for each feature.
    """
    if cols_to_pivot is None:
        cols_to_pivot = ['Open', 'High', 'Low', 'Close', 'Volume']

    df_filtered = df[(df['Time_to_Release_Minutes'] <= 2) & (df['Time_to_Release_Minutes'] >= -x_minutes_before)].copy()

    df_long = df_filtered.melt(
        id_vars=['Datetime', 'Time_to_Release_Minutes', 'Release_Datetime'],
        value_vars=cols_to_pivot,
        var_name='Feature',
        value_name='Value'
    )
    df_long['Feature_min'] = df_long['Feature'] + '_t' + df_long['Time_to_Release_Minutes'].astype(int).astype(str)

    df_wide = df_long.pivot_table(
        index='Release_Datetime',
        columns='Feature_min',
        values='Value'
    ).reset_index()
    cols_to_drop = [col for col in df_wide.columns if 't2' in col and col != 'Close_t2']
    df_wide.drop(columns=cols_to_drop, inplace=True)
    return df_wide

df_wide = pivot_market_data(price_window_60min, x_minutes_before=60)

def get_time_offset(col_name):
    match = re.search(r't(-?\d+)$', col_name)
    if match: return int(match.group(1))
    return float('inf')

sorted_market_cols = sorted(df_wide.columns, key=get_time_offset)
df_wide = df_wide[sorted_market_cols]
df_wide.set_index('Release_Datetime', inplace=True)
df_wide['Price_Change'] = df_wide['Close_t2'] - df_wide['Open_t0']
#df_wide is now sorted