import pandas as pd
import streamlit as st
from statsforecast import StatsForecast
from statsforecast.models import MSTL

from aux_hvac import get_delta_t


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


def get_forecasted_cumulative_delta_t(df: pd.DataFrame):
    delta_t_df = get_delta_t(df)

    if delta_t_df.empty:
        return None, None, None, None, "No Delta T data available to plot."

    # Resample to daily mean
    daily_df = delta_t_df.resample('D').mean()

    # Filter for current month and year
    last_date = daily_df.index.max()
    current_month = last_date.month
    current_year = last_date.year
    daily_df = daily_df[(daily_df.index.month == current_month) & (daily_df.index.year == current_year)]

    if daily_df.empty:
        return None, None, None, None, "No Delta T data available for the current month."

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
        return daily_series, None, None, None, None

    # Perform forecast
    forecast_df = mstl(df_forecast, season_length=[7], steps=forecast_days)
    forecast_df = forecast_df.rename(columns={"MSTL": "Delta T"})
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_index = pd.DatetimeIndex(forecast_df['ds'])

    # Combine
    combined_series = pd.concat([
        daily_series,
        pd.Series(forecast_df['Delta T'].values, index=forecast_index)
    ])

    # Compute cumulative means
    cumulative_actual = daily_series.expanding().mean()
    cumulative_forecast = pd.Series(
        combined_series.loc[forecast_index].expanding().mean().values,
        index=forecast_index
    )

    return cumulative_actual, cumulative_forecast, combined_series, forecast_index