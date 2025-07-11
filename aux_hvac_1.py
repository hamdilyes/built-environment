import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


##### AUX #####

def get_delta_t(df) -> pd.Series:
    sup_col = "BTU-01_SupTemp_degC"
    ret_col = "BTU-01_RetTemp_degC"
    lo = 0
    hi = 25
    if any(col not in df.columns for col in [sup_col, ret_col]):
        return
    mask = (
            (df[sup_col] >= lo) & (df[sup_col] <= hi) &
            (df[ret_col] >= lo) & (df[ret_col] <= hi)
    )
    delta = (df.loc[mask, ret_col] - df.loc[mask, sup_col]).rename("ΔT_°C")
    return delta.to_frame()


def get_oat(df):
    if "OAT-SENSOR_OutAirTemp (°C)" not in df.columns:
        return
    
    return df[["OAT-SENSOR_OutAirTemp (°C)"]]


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


##### PLOTS #####

def plot_delta_t(delta_t: pd.Series):
    if delta_t is None:
        return

    df = delta_t
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour
    pivot = (
        df.pivot_table(index="hour", columns="doy", values="ΔT_°C", aggfunc="mean")
          .iloc[::-1]                                # midnight at top
    )

    fig = px.imshow(
        pivot, aspect="auto", origin="lower",
        color_continuous_scale="RdBu_r",
        labels=dict(x="Day of Year", y="Hour", color="ΔT (°C)"),
        title="Supply-Return ΔT – Hourly Mean Calendar",
    )
    fig.update_coloraxes(cmin=0, cmax=8)
    st.plotly_chart(fig, use_container_width=True)


def plot_cop_oat(df_cop, df_oat):
    if any(df is None for df in [df_cop, df_oat]):
        return
    
    scatter_df = pd.DataFrame({
        "COP*": df_cop['COP* plant'],
        "OAT": df_oat["OAT-SENSOR_OutAirTemp (°C)"]
    }).dropna()

    fig = px.scatter(
        scatter_df, x="OAT", y="COP*",
        trendline="lowess",
        title="COP* vs Outside-Air Temperature"
    )
    st.plotly_chart(fig, use_container_width=True)