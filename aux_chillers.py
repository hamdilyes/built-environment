import streamlit as st
import plotly.express as px
import pandas as pd

from aux_hvac_get import get_capacity


def filter_current_month(df, required_columns):
    if df.empty:
        return df[required_columns] if required_columns else df
    df_valid = df.dropna(subset=required_columns)
    if df_valid.empty:
        return df[required_columns] if required_columns else df

    last_ts = pd.to_datetime(df_valid.index[-1])
    mask = (df.index.month == last_ts.month) & (df.index.year == last_ts.year)
    return df.loc[mask, required_columns]


def plot_chillers(df, customer):
    selections = st.session_state.get("selections", {})
    df = filter_current_month(
            df,
            required_columns=sorted(selections.get("Chiller Used Capacity", []))
        )

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

    # --- Sequencing View: Used Capacity Over Time ---
    fig_seq = px.line(
        df_cap,
        x=df_cap.index,
        y=df_cap.columns,
        labels={"value": "Capacity (%)", "variable": "Chiller", "index": "Time"},
        title="Month-to-Date Used Capacity"
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
        state_key = f"show_overload_{customer}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False
        explore_btn = st.button("⚠️ ALERT - Chiller Overload", key=f"btn_{customer}")
        if explore_btn:
            st.session_state[state_key] = not st.session_state[state_key]
            st.rerun()
        if st.session_state[state_key]:
            fig_overload = px.bar(
                x=overload_counts.index,
                y=overload_counts.values,
                labels={"x": "Chiller", "y": "Overload Count"},
                title="Month-to-Date Overload Events"
            )
            fig_overload.update_layout(
                xaxis_title="Chiller",
                yaxis_title="Occurrences",
                height=400
            )
            st.plotly_chart(fig_overload, use_container_width=True)