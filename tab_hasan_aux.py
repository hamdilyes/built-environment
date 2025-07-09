import pandas as pd
import streamlit as st

def get_consumption(df: pd.DataFrame) -> pd.DataFrame:
    selections = st.session_state.get("selections", {})

    # Safely fetch selections, default to empty list
    consumption_cols = selections.get("Power Consumption", [])
    meter_cols = selections.get("Power Meter", [])

    if consumption_cols:
        return df[consumption_cols].copy()

    if meter_cols:
        # Derive consumption from meters
        return pd.concat(
            {col.replace('ActEnergyDlvd_', 'kWh_'): df[col].diff().clip(lower=0) for col in meter_cols},
            axis=1
        )

    # Nothing to return
    return pd.DataFrame()