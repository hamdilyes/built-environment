import pandas as pd
import streamlit as st


def get_delta_t(df: pd.DataFrame) -> pd.DataFrame:
    selections = st.session_state.get("selections", {})

    supply_cols = sorted(selections.get("Chiller Supply Temperature", []))
    return_cols = sorted(selections.get("Chiller Return Temperature", []))

    # HARDCODE
    supply_temp = ['CHILLER-01_SupTemp (°C)', 'CHILLER-02_SupTemp (°C)', 'CHILLER-03_SupTemp (°C)']
    supply_cols = sorted([col for col in supply_temp if col in df.columns])

    return_temp = ['CHILLER-01_RetTemp (°C)', 'CHILLER-02_RetTemp (°C)', 'CHILLER-03_RetTemp (°C)']
    return_cols = sorted([col for col in return_temp if col in df.columns])

    if not supply_cols or not return_cols:
        return pd.DataFrame()

    delta_t_data = {}
    for supply_col, return_col in zip(supply_cols, return_cols):
        delta_col_name = f"DeltaT_{return_col[:10]}"
        delta_t_data[delta_col_name] = df[return_col] - df[supply_col]

    return pd.DataFrame(delta_t_data, index=df.index).dropna(how='all')


def get_flow(df: pd.DataFrame) -> pd.DataFrame:
    selections = st.session_state.get("selections", {})

    flow_cols = sorted(selections.get("Flow", []))

    if not flow_cols:
        return pd.DataFrame()

    flow_data = df[flow_cols].copy()
    flow_data.columns = [f"Flow_{col.replace('Flow', '').strip()}" for col in flow_cols]

    return flow_data.dropna(how='all')


def get_supply_temperature(df: pd.DataFrame) -> pd.DataFrame:
    selections = st.session_state.get("selections", {})

    return_cols = sorted(selections.get("Chiller Supply Temperature", []))

    if not return_cols:
        return pd.DataFrame()

    return_data = df[return_cols].copy()
    return_data.columns = [f"Return_{col.replace('Chiller Supply Temperature', '').strip()}" for col in return_cols]

    return return_data.dropna(how='all')


def get_setpoint(df: pd.DataFrame) -> pd.DataFrame:
    selections = st.session_state.get("selections", {})

    setpoint_cols = sorted(selections.get("Chiller Set-point Temperature", []))

    if not setpoint_cols:
        return pd.DataFrame()

    setpoint_data = df[setpoint_cols].copy()
    setpoint_data.columns = [f"Setpoint_{col.replace('Chiller Set-point Temperature', '').strip()}" for col in setpoint_cols]

    return setpoint_data.dropna(how='all')