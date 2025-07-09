import streamlit as st
import pandas as pd
import numpy as np

from aux_feature_selection import get_category_features

def get_data_granularity(df):
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return "Unknown"
    
    time_diffs = df.index.to_series().diff().dropna()
    if time_diffs.empty:
        return "Unknown"

    time_diffs_minutes = time_diffs.dt.total_seconds().div(60).round()
    most_common_diff = time_diffs_minutes.mode()
    if most_common_diff.empty:
        return "Unknown"
    
    diff_minutes = most_common_diff.iloc[0]

    if diff_minutes == 60:
        return "Hourly"
    elif diff_minutes == 1440:
        return "Daily"
    elif diff_minutes == 10080:
        return "Weekly"
    elif 40320 <= diff_minutes <= 44640:
        return "Monthly"
    elif diff_minutes < 60:
        return f"{int(diff_minutes)}min"


def show_dataset_overview(df):
    if df.empty:
        return

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Features", len(numerical_cols))
    with col2:
        granularity = get_data_granularity(df)
        st.metric("Granularity", granularity)
    with col3:
        st.metric("Start", df.index.min().strftime("%Y-%m-%d") if isinstance(df.index, pd.DatetimeIndex) else "N/A")
    with col4:
        st.metric("End", df.index.max().strftime("%Y-%m-%d") if isinstance(df.index, pd.DatetimeIndex) else "N/A")

    get_category_features(df)