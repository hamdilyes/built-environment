import streamlit as st
import plotly.express as px
import pandas as pd

from aux_hvac_get import get_capacity


def plot_chillers(df, customer):
    df_cap = get_capacity(df, customer)
    if df_cap is None or df_cap.empty:
        st.warning('NO DATA')
        return

    df_cap.index = pd.to_datetime(df_cap.index)
    last_date = df_cap.index.max()
    current_month_mask = (df_cap.index.month == last_date.month) & (df_cap.index.year == last_date.year)
    # df_cap = df_cap.loc[current_month_mask]
    if df_cap.empty:
        st.warning('NO DATA')
        return

    col_left, col_right = st.columns([2, 1])

    # # --- Average Capacity ---
    # with col_right:
    #     st.markdown("Month-to-Date Average Capacity")
    #     peak_values = df_cap[df_cap > 0].mean()
    #     for i, value in enumerate(peak_values.values, start=1):
    #         st.metric(label=f'Chiller_0{i}', value=f"{value:.0f} %")

    # # --- Usage Pie Chart ---
    # with col_left:
    #     chiller_usage = df_cap.sum()
    #     fig = px.pie(
    #         values=chiller_usage.values,
    #         names=chiller_usage.index,
    #         title="Month-to-Date Capacity Distribution",
    #         hole=0.4
    #     )
    #     fig.update_layout(showlegend=False)
    #     st.plotly_chart(fig, use_container_width=True)

    # --- Sequencing View: Used Capacity Over Time ---
    fig_seq = px.line(
        df_cap,
        x=df_cap.index,
        y=df_cap.columns,
        labels={"value": "Capacity (%)", "variable": "Chiller", "index": "Time"},
        title="Sequencing"
    )
    fig_seq.update_layout(
        xaxis_title="Time",
        yaxis_title="Capacity (%)",
        legend_title="Chiller",
        height=500
    )
    st.plotly_chart(fig_seq, use_container_width=True)

    # --- Overload Count Bar Chart ---
    overload_counts = (df_cap > 80).sum()
    if overload_counts.sum() > 0:
        fig_overload = px.bar(
            x=overload_counts.index,
            y=overload_counts.values,
            labels={"x": "Chiller", "y": "Overload Count"},
            title="Overload Events"
        )
        fig_overload.update_layout(
            xaxis_title="Chiller",
            yaxis_title="Occurrences",
            height=400
        )
        st.plotly_chart(fig_overload, use_container_width=True)