import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

from aux_hvac_get import get_setpoint, get_supply_temperature

def plot_short_cycling(df):
    # Get the supply temperature and setpoint DataFrames
    supply_temp_df = get_supply_temperature(df).min(axis=1)
    setpoint_df = get_setpoint(df).mean(axis=1)

    # Filter to last 30 days
    if df.index.max() is None:
        st.warning("No timestamp information available.")
        return

    end_time = df.index.max()
    start_time = end_time - timedelta(days=7)

    supply_temp_df = supply_temp_df.loc[start_time:end_time]
    setpoint_df = setpoint_df.loc[start_time:end_time]

    if supply_temp_df.empty or setpoint_df.empty:
        st.warning("Insufficient data for short cycling plot.")
        return

    # Create the Plotly figure
    fig = go.Figure()

    # Plot each supply temperature column
    fig.add_trace(go.Scatter(
        x=supply_temp_df.index,
        y=supply_temp_df.values,
        mode='lines',
        name=f"Supply",
        line=dict(color='green')
    ))

    # Plot each setpoint column
    fig.add_trace(go.Scatter(
        x=setpoint_df.index,
        y=setpoint_df.values,
        mode='lines',
        name=f"Setpoint",
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Chiller Supply Temperature and Setpoint",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        showlegend=False,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)