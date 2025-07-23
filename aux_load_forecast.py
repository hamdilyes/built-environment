import pandas as pd
import streamlit as st
from statsforecast import StatsForecast
from statsforecast.models import MSTL


def mstl(df, season_length=[7], steps=7):
    """
    df with columns 'ds' and 'y' -> needed
    """
    df = df.copy()
    df["unique_id"] = 1

    sf = StatsForecast(models=[MSTL(season_length=season_length)], freq='D')
    sf = sf.fit(df=df)
    forecast = sf.forecast(df=df, h=steps)

    forecast = forecast[['ds', 'MSTL']]
        
    return forecast


def get_forecasted_load(delta_t_df):
    if delta_t_df.empty:
        return None, None, None, None

    # Resample to daily sum
    daily_df = delta_t_df.resample('D').sum()

    # Filter for current month and year
    last_date = daily_df.index.max()
    current_month = last_date.month
    current_year = last_date.year
    daily_df = daily_df[(daily_df.index.month == current_month) & (daily_df.index.year == current_year)]

    # Drop last 15 days if last entry is end of month
    if daily_df.index[-1].is_month_end:
        daily_df = daily_df.iloc[:-15] if len(daily_df) > 15 else pd.DataFrame()

    if daily_df.empty:
        return None, None, None, None

    # Combine all columns by summing if multiple exist
    if len(daily_df.columns) > 1:
        daily_df["Delta T"] = daily_df.sum(axis=1)
        daily_series = daily_df["Delta T"]
    else:
        daily_series = daily_df.iloc[:, 0]

    # Forecasting preparation
    df_forecast = pd.DataFrame({
        "ds": daily_series.index,
        "y": daily_series.values
    })

    # Calculate forecast horizon
    last_day = pd.Timestamp(current_year, current_month, 1).days_in_month
    forecast_days = last_day - daily_series.index[-1].day

    # If no future dates left
    if forecast_days <= 0:
        cumulative_actual = daily_series
        return cumulative_actual, None, cumulative_actual, None

    # Perform forecast
    forecast_df = mstl(df_forecast, season_length=[7], steps=forecast_days)
    forecast_df = forecast_df.rename(columns={"MSTL": "Delta T"})
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_index = pd.DatetimeIndex(forecast_df['ds'])
    forecast_series = pd.Series(forecast_df['Delta T'].values, index=forecast_index)

    # Cumulative actual
    cumulative_actual = daily_series

    # Cumulative forecast (starting from last actual cumulative value)
    cumulative_forecast = forecast_series

    # Combined cumulative (actual + forecast)
    combined_series = pd.concat([cumulative_actual, cumulative_forecast])

    return cumulative_actual, cumulative_forecast, combined_series, forecast_index