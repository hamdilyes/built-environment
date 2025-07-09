import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from load_clickhouse import load_clickhouse, get_available_buildings, get_available_sensors, filter_data_by_building_and_sensor
from load_csv import process_uploaded_csvs

from tab_distributions import plot_distributions
from tab_timeseries import plot_time_series
from tab_overview import show_dataset_overview
from tab_relationships import plot_relationships
from tab_parallel import plot_parallel
from tab_hasan import plot_hasan
from tab_hvac import tab_hvac, tab_root_causes


# Set page configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sample data sources and customers
CLICKHOUSE_CUSTOMERS = {
    "Freimtech": [53],
    "DWTC": [10, 49],
}

if "show_root_causes" not in st.session_state:
    st.session_state.show_root_causes = False


def render_tabs(df, filtered_df=None):
    """
    Render all analysis tabs using either filtered_df (if provided) or df.
    """
    data_for_analysis = filtered_df if filtered_df is not None else df

    tab_functions = {
        "Overview": lambda: show_dataset_overview(df),
        "Time Series": lambda: plot_time_series(data_for_analysis),
        "Distributions": lambda: plot_distributions(data_for_analysis),
        "Parallel": lambda: plot_parallel(data_for_analysis),
        "Relationships": lambda: plot_relationships(data_for_analysis),
        "Hasan's": lambda: plot_hasan(df),
        'HVAC': lambda: tab_hvac(df),
    }

    tabs = st.tabs(tab_functions.keys())

    for tab, (label, func) in zip(tabs, tab_functions.items()):
        with tab:
            func()


def create_data_frame(source, customer, use_case="Tier 1 - Clickhouse"):
    """
    Main function to create the dataframe based on selected source and customer
    """
    if use_case == "Tier 2 - CSV":
        return pd.DataFrame()
    
    try:
        df = load_clickhouse(customer)
        if df is None:
            st.warning(f"load_clickhouse({customer}) returned None. Please check your data loading function.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error in load_clickhouse({customer}): {str(e)}")
        return pd.DataFrame()


# Sidebar
st.sidebar.title("🔧")

tier1_clickhouse = "Tier 1 - Clickhouse"
tier2_csv = "Tier 2 - CSV"
use_cases = [tier2_csv, tier1_clickhouse]
selected_use_case = st.sidebar.selectbox("Source", use_cases, index=0)

st.sidebar.markdown("---")

selected_source = None
selected_customer = None
selected_building = None
selected_sensor = None
df = pd.DataFrame()

# Handle CSV uploads
if selected_use_case == tier2_csv:
    st.sidebar.subheader("📁")
    uploaded_files = st.sidebar.file_uploader(
        "Data Upload",
        type=['csv'],
        accept_multiple_files=True,
        help="CSV files will be combined."
    )
    
    if uploaded_files:
        df = process_uploaded_csvs(uploaded_files)
        if not df.empty:
            st.sidebar.success(f"Successfully processed {len(uploaded_files)} CSV file(s)")
        else:
            st.sidebar.error("Failed to process CSV files")
    else:
        df = pd.DataFrame()

else:
    # Clickhouse flow
    selected_source = st.sidebar.selectbox("Source", list(CLICKHOUSE_CUSTOMERS.keys()), index=0)
    available_customers = CLICKHOUSE_CUSTOMERS[selected_source]
    selected_customer = st.sidebar.selectbox("Customer ID", available_customers, index=0)

    try:
        with st.spinner("Loading data..."):
            df = create_data_frame(selected_source, selected_customer, selected_use_case)

        if df is None or df.empty:
            df = pd.DataFrame()
            selected_building = None
            selected_sensor = None
            st.sidebar.warning("Data is empty or not loaded properly.")
        else:
            st.sidebar.markdown("---")
            available_buildings = get_available_buildings(df)

            if available_buildings:
                selected_building = st.sidebar.selectbox("Building ID", available_buildings, index=0)
                available_sensors = get_available_sensors(df, selected_building)

                if available_sensors:
                    sensor_options = available_sensors + ["Sum"]
                    selected_sensor = st.sidebar.selectbox("Sensor ID", sensor_options, index=len(sensor_options)-1)
                else:
                    selected_sensor = None
                    st.sidebar.warning("No sensors found for selected building")
            else:
                selected_building = None
                selected_sensor = None
                st.sidebar.warning("No building IDs found in data")

    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        selected_building = None
        selected_sensor = None
        df = pd.DataFrame()

# Main content
st.title(f"{selected_use_case}")

if not df.empty:
    if selected_use_case == tier1_clickhouse and selected_building and selected_sensor:
        st.header(f"Data Analysis - Customer {selected_customer} - Building {selected_building} - Sensor {selected_sensor}")
        filtered_df = filter_data_by_building_and_sensor(df, selected_building, selected_sensor)

        if filtered_df.empty:
            st.warning("No data available for the selected building and sensor combination.")
        else:
            render_tabs(df, filtered_df)
    else:
        if selected_use_case == tier1_clickhouse:
            st.subheader("Dataset Overview")
            show_dataset_overview(df)
            st.warning("Please select a building and sensor to view the detailed analysis.")
        else:
            render_tabs(df)
else:
    if selected_use_case == tier2_csv:
        st.info("Please upload CSV files using the sidebar to begin analysis.")
    else:
        st.warning("No data available. Please check your data source configuration.")