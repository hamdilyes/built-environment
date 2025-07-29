import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aux_hvac_1 import get_delta_t, get_flow, get_cop
from aux_hvac_get import get_capacity, get_chillerload
from aux_chiller_kpi import get_chiller_kpi


def filter_current_month(df, required_columns):
    if df.empty:
        return df[required_columns] if required_columns else df
    df_valid = df.dropna(subset=required_columns)
    if df_valid.empty:
        return df[required_columns] if required_columns else df

    last_ts = pd.to_datetime(df_valid.index[-1])
    mask = (df.index.month == last_ts.month) & (df.index.year == last_ts.year)
    return df.loc[mask, required_columns]


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
    df = filter_current_month(
        df,
        required_columns = [
            "BTU_load",
            f'Value_Chiller_{1}_kWh_2024',
            f'CHILLER-0{1}_ChlrCap (%)',
            f'CHILLER-0{1}_RunHr (Hours)',
            "MDB-01_ActEnergyDlvd_14648",
            "MDB-02_ActEnergyDlvd_14649",
            "MDB-03_ActEnergyDlvd_14650",
        ])
    with st.expander(f"MONTH-TO-DATE CHILLER KPIs", expanded=False):
        col1, col2 = st.columns([1,1.5])
        with col1:
            chiller_name = st.selectbox("Chiller", ['CH1', 'CH2', 'CH3'])
        kpi_dict = get_chiller_kpi(df, chiller_name)
        if kpi_dict:
            plot_chiller_kpi(kpi_dict, chiller_name, customer)


def plot_chiller_kpi(kpi_dict: dict, chiller_name: str, customer):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(
    "COP",
    f"{kpi_dict['avg_COP_full']:.2f}",
    help="**Coefficient Of Performance (COP)**\n"
         "Measures chiller efficiency: cooling output ÷ electrical input.\n"
         "Higher COP means better energy efficiency."
    )
    col2.metric(
        "EER",
        f"{kpi_dict['avg_EER_full']:.2f}",
        help="**Energy Efficiency Ratio (EER)**\n"
            "Cooling output (BTU/hr) ÷ power input (W).\n"
            "Higher EER indicates lower power consumption for the same cooling."
    )
    col3.metric(
        "kW/TR",
        f"{kpi_dict['avg_kW_per_ton_full']:.2f}",
        help="**Power per Cooling Ton (kW/TR)**\n"
            "Indicates power used to produce 1 ton of cooling.\n"
            "Lower values suggest higher efficiency."
    )
    col4.metric(
        "Capacity",
        f"{kpi_dict['avg_CapacityUtil_%_full']:.0f} %",
        help="**Capacity Utilization (%)**\n"
            "Percent of total cooling capacity being used.\n"
            "Higher values show more demand on the system."
    )
    col5.metric(
        "Runtime",
        f"{kpi_dict['avg_daily_hours_full']:.2f} h",
        help="**Average Daily Runtime (h)**\n"
            "Total operating hours per day.\n"
            "Longer runtimes may indicate high demand or inefficiencies."
    )
    col6.metric(
        "Optimal Load",
        f"{kpi_dict['avg_hours_by_band_full']["Hrs_50‑75%"]:.0f} h",
        help="Average Daily Runtime operating within the optimal load band (50–75%), where efficiency is typically highest."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        explore_btn = st.button("⚠️ ALERT - Oversized Chiller", key=f"btn_kpi_1_{customer}")
    with col2:
        explore_btn = st.button("⚠️ ALERT - Inefficient Chiller", key=f"btn_kpi_2_{customer}")
    with col3:
        explore_btn = st.button("⚠️ ALERT - Short Cycling", key=f"btn_kpi_3_{customer}")