import pandas as pd
import streamlit as st

from aux_hvac_1 import get_delta_t, get_flow, get_load


def plot_live(df, customer):
    delta_t_df = get_delta_t(df, customer)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA.")
        return

    col1, col2, col3 = st.columns(3)

    try:
        df_delta_t = get_delta_t(df, customer)
        live_delta_t = None
        if not df_delta_t.dropna().empty:
            live_delta_t = float(df_delta_t.dropna().iloc[-1])
        col1.metric("∆T", f"{live_delta_t:.2f} °C")
    except:
        col1.metric("∆T", "...")

    try:
        df_flow = get_flow(df, customer)
        live_flow = None
        if not df_flow.dropna().empty:
            live_flow = round(float(df_flow.dropna().iloc[-1]),2)
        col2.metric("Flow", f"{live_flow:.2f} L/s")
    except:
        col2.metric("Flow", "...")

    try:
        df_load = get_load(df, customer)
        live_load = None
        live_load = round(float(df_load.dropna().iloc[-1]),2)
        col3.metric("Cooling Load", f"{live_load:.2f} kWh")
    except:
        col3.metric("Cooling Load", "...")