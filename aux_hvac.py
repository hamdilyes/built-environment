import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aux_hvac_1 import get_delta_t, get_flow, get_cop
from aux_hvac_get import get_capacity, get_chillerload
from aux_chiller_kpi import get_chiller_kpi


def plot_live(df, customer):
    delta_t_df = get_delta_t(df, customer)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

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
        df_cop = get_cop(df, customer)
        live_cop = None
        live_cop = round(float(df_cop.dropna().iloc[-1]["COP* plant"]),2)
        col3.metric("COP", f"{live_cop:.2f}")
    except:
        col3.metric("COP", "...")

    try:
        df_cap = get_capacity(df, customer)
        live_cap = None
        live_cap = round(float(df_cap.dropna().iloc[-1].sum()),2)
        col4.metric("Chiller Capacity", f"{live_cap:.0f} %")
    except:
        col4.metric("Chiller Capacity", "...")

    try:
        df_chillerload = get_chillerload(df, customer)
        live_chillerload = None
        live_chillerload = round(float(df_chillerload.dropna().iloc[-1].sum()),2)
        col5.metric("Chiller Load", f"{live_chillerload:.0f} kWh")
    except:
        col5.metric("Chiller Load", "...")


def plot_chillers_kpi(df, customer):
    with st.expander(f"CHILLER KPIs", expanded=False):
        col1, col2 = st.columns([1,1.5])
        with col1:
            chiller_name = st.selectbox("Chiller", ['CH1', 'CH2', 'CH3'])
        kpi_dict = get_chiller_kpi(df, chiller_name)
        if kpi_dict:
            plot_chiller_kpi(kpi_dict, chiller_name)


def plot_chiller_kpi(kpi_dict: dict, chiller_name: str):
    col0, col1, col2, col3, col4, col5 = st.columns(6)
    col0.metric("", chiller_name)
    col1.metric("COP", f"{kpi_dict['avg_COP_full']:.2f}")
    col2.metric("EER", f"{kpi_dict['avg_EER_full']:.2f}")
    col3.metric("kW/ton", f"{kpi_dict['avg_kW_per_ton_full']:.2f}")
    col4.metric("Avg Used Capacity", f"{kpi_dict['avg_CapacityUtil_%_full']:.0f} %")
    col5.metric("Avg Daily Runtime", f"{kpi_dict['avg_daily_hours_full']:.2f} h")
    
    # For column 6: summarize total runtime across all bands
    # hours_by_band = kpi_dict["avg_hours_by_band_full"]
    # total_hours = sum(hours_by_band.values())
    # col6.metric("Total Runtime by Band (hrs)", f"{total_hours:.1f}")
    # col6.metric("6–12hr Band (hrs)", f"{hours_by_band.get('Hrs_6_12', 0):.1f}")