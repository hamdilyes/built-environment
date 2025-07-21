import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime


def plot_live(df):
    delta_t_df = get_delta_t(df)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA.")
        return
    building_names = ["4A", "4B", "4C", "4D", "4E"]
    cols = st.columns(5)
    for column, col, building_name in zip(delta_t_df.columns, cols, building_names):
        try:
            delta_t = delta_t_df[column]
            live_delta_t = None
            if not delta_t.dropna().empty:
                live_delta_t = float(delta_t.dropna().iloc[-1])
            col.metric(f"{building_name}", f"{live_delta_t:.2f} °C")
        except:
            col.metric(f"{building_name}", "...")


def plot_avg(df):
    delta_t_df = get_delta_t(df)
    if delta_t_df is None or delta_t_df.empty:
        st.warning("NO DATA.")
        return
    
    # Filter for current month
    now = delta_t_df.index.max()
    df_month = delta_t_df[
        (delta_t_df.index.month == now.month) &
        (delta_t_df.index.year == now.year)
    ]

    building_names = ["4A", "4B", "4C", "4D", "4E"]
    cols = st.columns(5)
    
    for column, col, building_name in zip(df_month.columns, cols, building_names):
        delta_t = df_month[column]
        avg_delta_t = None
        if not delta_t.dropna().empty:
            avg_delta_t = delta_t.dropna().mean()
            col.metric(f"{building_name}", f"{avg_delta_t:.2f} °C")
        else:
            col.metric(f"{building_name}", "...")


##### AUX #####


def get_delta_t(df) -> pd.Series:
    cols = []
    for col in df.columns:
        if 'chw_delta_temp_celsius' in col:
            cols.append(col)
    return df[cols]


def get_flow(df):
    if "BTU-01_ChwFlow_L/s" not in df.columns:
        return
    
    return df[["BTU-01_ChwFlow_L/s"]]


def get_cop(df):
    if any(col not in df.columns for col in ["Value_BTU_Meter_Data_2024", "Value_Chiller_1_kWh_2024", "Value_Chiller_2_kWh_2024", "Value_Chiller_3_kWh_2024"]):
        return

    df = df.copy()
    kWth = df["Value_BTU_Meter_Data_2024"] * 4
    ch1_kw = df["Value_Chiller_1_kWh_2024"]
    ch2_kw = df["Value_Chiller_2_kWh_2024"]
    ch3_kw = df["Value_Chiller_3_kWh_2024"]

    elec_sum = ch1_kw + ch2_kw + ch3_kw
    share1 = ch1_kw / elec_sum.replace(0, np.nan)
    share2 = ch2_kw / elec_sum.replace(0, np.nan)
    share3 = ch3_kw / elec_sum.replace(0, np.nan)

    df["COP* plant"] = kWth / elec_sum.replace(0, np.nan)
    df["COP-1"] = (kWth * share1) / ch1_kw.replace(0, np.nan)
    df["COP-2"] = (kWth * share2) / ch2_kw.replace(0, np.nan)
    df["COP-3"] = (kWth * share3) / ch3_kw.replace(0, np.nan)

    cop_real = df[["COP* plant", "COP-1", "COP-2", "COP-3"]]
    good = (cop_real > 0) & (cop_real < 10) 
    df_cop = cop_real.where(good).dropna(how='all')

    return df_cop