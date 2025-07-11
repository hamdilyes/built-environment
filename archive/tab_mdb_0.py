import pandas as pd
import plotly.express as px
import streamlit as st

from archive.aux_mdb_0 import get_consumption


def tab_mdb(df):
    df_consumption = get_consumption(df)
    if df_consumption.empty:
        st.warning("No 'Consumption' data available.")
        return
    
    plot_wholeelec(df_consumption)
    plot_contribution(df_consumption)
    plot_profiles(df_consumption)
    plot_monthheat(df_consumption)
    plot_loadduration(df_consumption)


def plot_wholeelec(df_consumption):
    """
    Display a stacked bar chart of electricity consumption using MOB/MDB-level data.
    Assumes df is a time-indexed DataFrame and get_consumption(df) returns only kWh columns.
    """
    df_plot = df_consumption.reset_index().melt(id_vars=df_consumption.index.name, var_name='MDB', value_name='kWh')

    fig = px.bar(df_plot, x=df_consumption.index.name, y='kWh', color='MDB', title='MDB-level consumption')

    st.plotly_chart(fig, use_container_width=True)


def plot_contribution(df_consumption: pd.DataFrame) -> None:
    """
    Displays a pie chart showing the percentage energy contribution
    of each specified MDB or meter based on interval data.

    Parameters:
    - df: DataFrame with a datetime index and energy-related columns.
    - consumption_cols: List of cumulative or interval consumption columns to include.
    """
    if len(df_consumption.columns) < 2:
        return
    
    # Compute percentage contribution
    contribution = df_consumption.sum() / df_consumption.sum().sum() * 100

    # Create pie chart
    fig = px.pie(
        values=contribution,
        names=contribution.index,
        title='Energy Contribution by MDB (%)',
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_profiles(df_consumption):
    """
    Display average 24-hour load profiles for weekdays and weekends.
    Assumes df is a time-indexed DataFrame and get_consumption(df) returns only kWh columns.
    """
    try:
        df_consumption = df_consumption.copy()

        df_consumption['Hour'] = df_consumption.index.hour
        df_consumption['Weekday'] = df_consumption.index.weekday

        weekday = df_consumption[df_consumption['Weekday'] < 5].groupby('Hour').mean()
        weekend = df_consumption[df_consumption['Weekday'] >= 5].groupby('Hour').mean()

        # Combine into long-form for Plotly Express
        df_plot = weekday.sum(axis=1).rename('Weekday').to_frame()
        df_plot['Weekend'] = weekend.sum(axis=1)
        df_plot = df_plot.reset_index().melt(id_vars='Hour', var_name='Day Type', value_name='kWh')

        fig = px.line(df_plot, x='Hour', y='kWh', color='Day Type', title='Average 24-h Load Profiles')
        fig.update_layout(xaxis_title='Hour', yaxis_title='kWh per Interval')

        st.plotly_chart(fig, use_container_width=True)
    except:
        pass


def plot_monthheat(df_consumption: pd.DataFrame) -> None:
    """
    Displays a heatmap showing the seasonal energy usage profile (Hour vs Month).
    Automatically detects or derives interval energy consumption data.

    Parameters:
    - df: DataFrame with a datetime index and energy-related columns.
    """
    # Compute total energy across all available columns
    df_consumption['total_kWh'] = df_consumption.sum(axis=1)

    # Group by hour and month
    tmp = df_consumption.copy()
    tmp['Hour'] = tmp.index.hour
    tmp['Month'] = tmp.index.month
    monthly = tmp.groupby(['Month', 'Hour'])['total_kWh'].mean().unstack(level=0)

    # Create heatmap
    fig = px.imshow(
        monthly,
        aspect='auto',
        origin='lower',
        labels=dict(x='Month', y='Hour', color='kWh/interval'),
        title='Seasonal Profile – Hour vs Month'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_loadduration(df_consumption, pctl=0.05):
    """
    Display the load duration curve with a base load reference line.
    Assumes df is a time-indexed DataFrame and get_consumption(df) returns only kWh columns.
    """
    total_load = df_consumption.sum(axis=1)

    # Sort descending for Load Duration Curve
    ldc = total_load.sort_values(ascending=False).reset_index(drop=True)

    # Estimate base load from percentile of total load
    base_kw = total_load.quantile(pctl) * 4  # convert kWh/15min to kW

    fig = px.line(ldc * 4, labels={'value': 'kW', 'index': 'Ranked Interval'},
                  title='Load Duration Curve')
    fig.add_hline(y=base_kw, line_dash='dot',
                  annotation_text=f'Base Load ≈ {base_kw:,.0f} kW')

    st.plotly_chart(fig, use_container_width=True)