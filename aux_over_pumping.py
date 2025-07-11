import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from aux_hvac_get import get_delta_t, get_flow


def plot_overpumping(df):
    over_pumping_dict = {
        'Summer': lambda: plot_overpumping_summer(df),
        'Restrictor': lambda: plot_overpumping_restrictor(df),
    }

    over_pumping_cases = list(over_pumping_dict.keys())

    for i in range(0, len(over_pumping_cases), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(over_pumping_cases):
                case = over_pumping_cases[i + j]
                with cols[j]:
                    with st.expander(case, expanded=False):
                        over_pumping_dict[case]()


def plot_overpumping_summer(df: pd.DataFrame):
    delta_t_df = get_delta_t(df).sum(axis=1)
    flow_df = get_flow(df).sum(axis=1)

    if delta_t_df.empty or flow_df.empty:
        st.warning("Insufficient data for over-pumping plot.")
        return
    
    # Align indexes (intersection only)
    common_index = delta_t_df.index.intersection(flow_df.index)
    delta_t_df = delta_t_df.loc[common_index]
    flow_df = flow_df.loc[common_index]

    # Limit to last 1 year
    max_date = common_index.max()
    one_year_ago = max_date - pd.Timedelta(days=365)

    delta_t_df = delta_t_df.loc[delta_t_df.index >= one_year_ago]
    flow_df = flow_df.loc[flow_df.index >= one_year_ago]

    # Filter to July–October only
    valid_months = [7, 8, 9, 10]
    month_filter = delta_t_df.index.month.isin(valid_months)
    delta_t_df = delta_t_df.loc[month_filter]
    flow_df = flow_df.loc[month_filter]

    # Resample both to daily means
    delta_t_daily = delta_t_df.resample('D').mean()
    flow_daily = flow_df.resample('D').mean()

    st.markdown("Operators or automation systems set chilled water flow higher to meet demand.")

    fig = go.Figure()

    # Delta T trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=delta_t_daily.index,
        y=delta_t_daily,
        name="∆T (°C)",
        yaxis="y1",
        line=dict(color="green")
    ))

    # Flow trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=flow_daily.index,
        y=flow_daily,
        name="Flow (L/s)",
        yaxis="y2",
        line=dict(color="red")
    ))

    # Layout with dual y-axes
    fig.update_layout(
        title="Daily ∆T and Flow",
        yaxis=dict(title="∆T (°C)", tickfont=dict(color="green")),
        yaxis2=dict(title="Flow (L/s)", tickfont=dict(color="red"),
                    overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=500,
    )
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)


def plot_overpumping_restrictor(df: pd.DataFrame, threshold=25):
    """
    Plot the impact of flow limiting on delta T showing monthly averages
    """
    delta_t_df = get_delta_t(df).sum(axis=1)
    flow_df = get_flow(df).sum(axis=1)
    
    if delta_t_df.empty or flow_df.empty:
        st.warning("Insufficient data for over-pumping average analysis.")
        return
    
    # Align indexes (intersection only)
    common_index = delta_t_df.index.intersection(flow_df.index)
    delta_t_df = delta_t_df.loc[common_index]
    flow_df = flow_df.loc[common_index]
    
    # Limit to last 1 year
    max_date = common_index.max()
    one_year_ago = max_date - pd.Timedelta(days=365)
    delta_t_df = delta_t_df.loc[delta_t_df.index >= one_year_ago]
    flow_df = flow_df.loc[flow_df.index >= one_year_ago]
    
    # Set flow threshold (80th percentile of flow)
    flow_threshold = flow_df.quantile(1-threshold/100)
    
    # Calculate consumption (flow * delta_t)
    consumption = flow_df * delta_t_df
    
    # Apply flow limiting
    flow_limited = np.minimum(flow_df, flow_threshold)
    
    # Calculate new delta_t to maintain same consumption
    delta_t_limited = consumption / flow_limited
    
    # Calculate monthly averages
    delta_t_monthly = delta_t_df.resample('M').mean()
    delta_t_limited_monthly = delta_t_limited.resample('M').mean()

    st.markdown("Applying a flow restrictor to prevent over-pumping can improve ∆T.")
    
    # Create the plot
    fig = go.Figure()
    
    # Current Delta T trace
    fig.add_trace(go.Scatter(
        x=delta_t_monthly.index,
        y=delta_t_monthly,
        name="Current",
        yaxis="y1",
        line=dict(color="red", width=3)
    ))
    
    # Limited Flow Delta T trace
    fig.add_trace(go.Scatter(
        x=delta_t_limited_monthly.index,
        y=delta_t_limited_monthly,
        name="Optimized",
        yaxis="y1",
        line=dict(color="green", width=3, dash='dash')
    ))
    
    # Layout with single y-axis
    fig.update_layout(
        title=f"Monthly ∆T",
        yaxis=dict(title="∆T (°C)", tickfont=dict(color="black")),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=500,
    )
    
    fig.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display summary metrics
    avg_delta_t_current = delta_t_monthly.mean()
    avg_delta_t_improved = delta_t_limited_monthly.mean()
    improvement = avg_delta_t_improved - avg_delta_t_current
    improvement_pct = (improvement / avg_delta_t_current) * 100
    
    # Calculate how often flow limiting would be active
    affected_points = (flow_df > flow_threshold).sum()
    total_points = len(flow_df)
    affected_pct = (affected_points / total_points) * 100
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active rate",
            f"{affected_pct:.0f}%",
        )
    
    with col2:
        st.metric(
            "Current ∆T",
            f"{avg_delta_t_current:.1f}°C"
        )
    
    with col3:
        st.metric(
            "Optimized ∆T",
            f"{avg_delta_t_improved:.1f}°C"
        )
    
    with col4:
        st.metric(
            "Improvement",
            f"{improvement:.2f}°C",
            f"{improvement_pct:.1f}%"
        )