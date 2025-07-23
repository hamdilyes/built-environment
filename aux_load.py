import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aux_hvac_1 import get_load
from aux_load_forecast import get_forecasted_load


def plot_load(df, customer):
    # Check
    delta_t_df = get_load(df, customer)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA.")
        return

    load_overview(df, customer)
    plot_forecasted_load(df, None, customer)


def load_overview(df: pd.DataFrame, customer):
    delta_t_df = get_load(df, customer)

    # Get the latest available timestamp and sum values if multiple columns
    latest_ts = delta_t_df.index.max()
    latest_values = delta_t_df.loc[latest_ts]
    if isinstance(latest_values, pd.Series):
        live_delta_t = latest_values.sum()
    else:
        live_delta_t = float(latest_values)

    # Get current month and previous month
    latest_date = pd.to_datetime(latest_ts)
    current_month = latest_date.month
    current_year = latest_date.year

    # Filter for current month
    current_month_df = delta_t_df[
        (delta_t_df.index.month == current_month) & (delta_t_df.index.year == current_year)
    ]
    current_month_sum = current_month_df.sum(axis=1).sum() if not current_month_df.empty else None

    # Filter for previous month, handle year boundary
    prev_month = current_month - 1 if current_month > 1 else 12
    prev_year = current_year if current_month > 1 else current_year - 1
    prev_month_df = delta_t_df[
        (delta_t_df.index.month == prev_month) & (delta_t_df.index.year == prev_year)
    ]
    prev_month_sum = prev_month_df.sum(axis=1).sum() if not prev_month_df.empty else None

    # ---- Forecast ----
    cumulative_actual, cumulative_forecast, combined_series, forecast_index = get_forecasted_load(delta_t_df)
    forecasted_sum = combined_series.sum() if combined_series is not None and not combined_series.empty else None

    # Display metrics side by side
    col1, col2, col4 = st.columns(3)

    if prev_month_sum is not None:
        col1.metric(label="Previous Month", value=f"{prev_month_sum:,.0f} kWh")

    if current_month_sum is not None:
        pct = (current_month_sum / prev_month_sum - 1) * 100 if prev_month_sum else 0
        col2.metric("Month-to-Date", f"{current_month_sum:,.0f} kWh", f"{pct:.1f}%")
    
    if forecasted_sum is not None:
        pct = (forecasted_sum / prev_month_sum - 1) * 100 if prev_month_sum else 0
        col4.metric("Forecasted", f"{forecasted_sum:,.0f} kWh", f"{pct:.1f}%")


def plot_forecasted_load(df, threshold, customer):
    delta_t_df = get_load(df, customer)

    # Get forecasted cumulative load data
    load_actual, load_forecast, combined_series, forecast_index = get_forecasted_load(delta_t_df)

    if any(v is None for v in (load_actual, load_forecast, combined_series, forecast_index)):
        return
    
     # Adjust forecast to start from the last actual value
    if load_forecast is not None and not load_actual.empty:
        last_actual_ts = load_actual.index[-1]
        last_actual_val = load_actual.iloc[-1]

        # Create a new index and values for forecast starting from last actual point
        forecast_index = forecast_index.insert(0, last_actual_ts)
        load_forecast = pd.concat([
            pd.Series([last_actual_val], index=[last_actual_ts]),
            load_forecast
        ])

    # Combine actual and forecast (excluding duplicate timestamp at forecast start)
    combined_load = pd.concat([load_actual, load_forecast[1:]])

    fig = go.Figure()

    # Plot actual load
    fig.add_trace(go.Scatter(
        x=load_actual.index,
        y=load_actual,
        mode='lines',
        name="Load (Actual)",
    ))

    # Plot forecast load
    if load_forecast is not None:
        fig.add_trace(go.Scatter(
            x=load_forecast.index,
            y=load_forecast,
            mode='lines',
            name="Load (Forecast)",
            line=dict(color='#636EFA', dash='dash')
        ))

        # Forecast start indicator
        forecast_start_dt = forecast_index[0].to_pydatetime()
        fig.add_vline(
            x=forecast_start_dt,
            line=dict(color='gray', dash='dot'),
        )
        fig.add_annotation(
            x=forecast_start_dt,
            y=1,
            yref='paper',
            text="Forecast",
            showarrow=False,
            xanchor='left',
            font=dict(color='white'),
            bgcolor=None,
            bordercolor='gray',
            borderwidth=0,
            borderpad=4,
        )

    # Threshold line
    if threshold:
        fig.add_trace(go.Scatter(
            x=combined_load.index,
            y=[threshold] * len(combined_load),
            mode='lines',
            name='Threshold',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title="Month-to-Date",
        xaxis_title="Date",
        yaxis_title="Cooling Load (kWh)",
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True, key='load_' + customer)