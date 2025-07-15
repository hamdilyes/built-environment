import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aux_hvac_1 import (
    get_delta_t,
    get_oat,
    get_cop,

    plot_delta_t,
    plot_cop_oat,
)


def tab_hvac_1(df):
    required_columns = [
        "BTU-01_SupTemp_degC",
        "BTU-01_RetTemp_degC",
        "OAT-SENSOR_OutAirTemp (°C)",
    ]
    
    has_required_column = any(col in df.columns for col in required_columns)
    
    if not has_required_column:
        return
    
    if not st.session_state.get("run_hvac_1", False):
        st.session_state.run_hvac_1 = False
    
    if st.button("RUN / HIDE", key='button_run_hvac_1'):
        st.session_state.run_hvac_1 = not st.session_state.run_hvac_1
    
    if st.session_state.get("run_hvac_1", False):
        tab_hvac_1_run(df)


def tab_hvac_1_run(df):
    # definitions
    df_delta_t = get_delta_t(df)
    df_oat = get_oat(df)
    df_cop = get_cop(df)

    # plots
    plot_delta_t(df_delta_t)
    plot_cop_oat(df_cop, df_oat)