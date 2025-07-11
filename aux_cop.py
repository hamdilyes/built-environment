import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aux_delta_t_forecast import get_forecasted_cumulative_delta_t
from aux_hvac_1 import get_cop


def plot_cop(df):
    # Check
    delta_t_df = get_cop(df)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA.")
        return

    delta_t_overview(df)
    plot_forecasted_cumulative_delta_t(df, 7)


def delta_t_overview(df: pd.DataFrame):
    delta_t_df = get_cop(df)[['COP* plant']]

    # Get the latest available timestamp and sum values if multiple columns
    latest_ts = delta_t_df.index.max()
    latest_values = delta_t_df.loc[latest_ts]
    if isinstance(latest_values, pd.Series):
        live_delta_t = latest_values.sum()
    else:
        live_delta_t = float(latest_values)

    # Get current month and previous month averages
    latest_date = pd.to_datetime(latest_ts)
    current_month = latest_date.month
    current_year = latest_date.year

    # Filter for current month
    current_month_df = delta_t_df[
        (delta_t_df.index.month == current_month) & (delta_t_df.index.year == current_year)
    ]
    current_month_avg = current_month_df.sum(axis=1).mean() if not current_month_df.empty else None

    # Filter for previous month, handle year boundary
    prev_month = current_month - 1 if current_month > 1 else 12
    prev_year = current_year if current_month > 1 else current_year - 1
    prev_month_df = delta_t_df[
        (delta_t_df.index.month == prev_month) & (delta_t_df.index.year == prev_year)
    ]
    prev_month_avg = prev_month_df.sum(axis=1).mean() if not prev_month_df.empty else None

    # ---- Forecast ----
    cumulative_actual, cumulative_forecast, combined_series, forecast_index = get_forecasted_cumulative_delta_t(delta_t_df)
    forecasted_avg = combined_series.mean()

    # Display metrics side by side
    col1, col2, col4 = st.columns(3)

    if prev_month_avg is not None:
        col1.metric(label="Previous Month COP", value=f"{prev_month_avg:.2f}")

    if current_month_avg is not None:
        pct = (current_month_avg/prev_month_avg - 1)*100
        col2.metric("Month-to-Date COP", f"{current_month_avg:.2f}", f"{pct:.1f}%")
    
    if forecasted_avg is not None:
        pct = (forecasted_avg/prev_month_avg - 1)*100
        col4.metric("Forecasted COP", f"{forecasted_avg:.2f}", f"{pct:.1f}%")


def plot_forecasted_cumulative_delta_t(df, threshold):
    delta_t_df = get_cop(df)[['COP* plant']]
    # get forecasted data
    cumulative_actual, cumulative_forecast, combined_series, forecast_index = get_forecasted_cumulative_delta_t(delta_t_df)
    
    if any(v is None for v in (cumulative_actual, cumulative_forecast, combined_series, forecast_index)):
        return

    # Adjust forecast to start from the last actual value
    if cumulative_forecast is not None and not cumulative_actual.empty:
        last_actual_ts = cumulative_actual.index[-1]
        last_actual_val = cumulative_actual.iloc[-1]

        # Create a new index and values for forecast starting from last actual point
        forecast_index = forecast_index.insert(0, last_actual_ts)
        cumulative_forecast = pd.concat([
            pd.Series([last_actual_val], index=[last_actual_ts]),
            cumulative_forecast
        ])
        
        # Update combined series index for threshold line alignment
        combined_series = pd.concat([
            cumulative_actual,
            cumulative_forecast[1:]  # exclude the duplicate timestamp
        ])

    fig = go.Figure()

    # Plot actual
    fig.add_trace(go.Scatter(
        x=cumulative_actual.index,
        y=cumulative_actual,
        mode='lines',
        name="Cumulative Avg Delta T (Actual)",
    ))

    # Plot forecast if available
    if cumulative_forecast is not None:
        fig.add_trace(go.Scatter(
            x=cumulative_forecast.index,
            y=cumulative_forecast,
            mode='lines',
            name="Cumulative Avg Delta T (Forecast)",
            line=dict(color='#636EFA', dash='dash')
        ))

        # Vertical line for forecast start
        start_forecast_dt = forecast_index[0].to_pydatetime()
        fig.add_vline(
            x=start_forecast_dt,
            line=dict(color='gray', dash='dot'),
        )
        fig.add_annotation(
            x=start_forecast_dt,
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
            x=combined_series.index,
            y=[threshold] * len(combined_series),
            mode='lines',
            name='Threshold',
            line=dict(color='red', dash='dash')
        ))

    fig.update_layout(
        title="Month-to-Date Daily Average ∆T",
        xaxis_title="Date",
        yaxis_title="∆T (°C)",
        hovermode="x unified",
        template="plotly_white",
        showlegend=False,
    )
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)