import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aux_hvac_1 import get_delta_t, get_flow, get_cop


def plot_live(df):
    delta_t_df = get_delta_t(df)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA.")
        return

    col1, col2, col3 = st.columns(3)

    try:
        df_delta_t = get_delta_t(df)
        live_delta_t = None
        if not df_delta_t.dropna().empty:
            live_delta_t = float(df_delta_t.dropna().iloc[-1])
        col1.metric("∆T", f"{live_delta_t:.2f} °C")
    except:
        col3.metric("∆T", "...")

    try:
        df_flow = get_flow(df)
        live_flow = None
        if not df_flow.dropna().empty:
            live_flow = round(float(df_flow.dropna().iloc[-1]),2)
        col2.metric("Flow", f"{live_flow:.2f} L/s")
    except:
        col3.metric("Flow", "...")

    try:
        df_cop = get_cop(df)
        live_cop = None
        live_cop = round(float(df_cop.dropna().iloc[-1]["COP* plant"]),2)
        col3.metric("COP", f"{live_cop:.2f}")
    except:
        col3.metric("COP", "...")