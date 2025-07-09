import logging
import clickhouse_connect
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil
import streamlit as st

def load_clickhouse(customer_id):
    # Setup ClickHouse connection parameters
    CLICKHOUSE_HOST: str = st.secrets["CLICKHOUSE_HOST"]
    CLICKHOUSE_PORT: str = st.secrets["CLICKHOUSE_PORT"]
    CLICKHOUSE_DB: str = st.secrets["CLICKHOUSE_DB"]
    CLICKHOUSE_USER: str = st.secrets["CLICKHOUSE_USER"]
    CLICKHOUSE_PASSWORD: str = st.secrets["CLICKHOUSE_PASSWORD"]

    # Establish connection with ClickHouse
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
        secure=True,
        verify=False
    )

    timezone = client.command('SELECT timezone()')
    print("ClickHouse connection succeeded: " + timezone)

    logger = logging.getLogger(__name__)

    def get_data(client,
                 table_name: str,
                 customer: int,
                 start_date: datetime,
                 end_date: datetime):
        try:
            start_time_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_time_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            query = f"""
            SELECT * 
            FROM {table_name} FINAL
            WHERE timestamp >= toDateTime('{start_time_str}')
            AND timestamp <= toDateTime('{end_time_str}')
            AND customer_id = '{customer}'
            """
            result = client.query(query)

            if not result.result_rows:
                logger.warning(f"No data found for customer {customer} in time range {start_time_str} - {end_time_str}")
                return pd.DataFrame()

            df = pd.DataFrame(result.result_rows)
            df.columns = result.column_names
            logger.info(f"Retrieved {len(df)} rows from {table_name} in time range {start_time_str} - {end_time_str}")
            return df
        except KeyError:
            error_msg = f"No table mapping found for utility type: {table_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error fetching data for {table_name}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # Fetch the data
    df = get_data(
        client,
        table_name="electricity_consumption_and_anomalies_tier1",
        customer=customer_id,
        start_date=dateutil.parser.isoparse("2024-01-01T00:00:00.000Z"),
        end_date=dateutil.parser.isoparse("2026-01-01T00:00:00.000Z"),
    )

    # Keep only necessary columns
    df = df[["building_id", "sensor_id", "timestamp", "power_kwh", "consumption"]]

    # Convert timestamp column to datetime and set it as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    return df

def get_available_buildings(df):
    """
    Get list of available building IDs from the dataframe
    """
    if df is not None and not df.empty and 'building_id' in df.columns:
        return sorted(df['building_id'].unique())
    return []

def get_available_sensors(df, building_id):
    """
    Get list of available sensors for a specific building
    Returns unique sensor_id values for the selected building
    """
    if df is None or df.empty:
        return []
        
    if 'sensor_id' not in df.columns:
        return []
        
    building_df = df[df['building_id'] == building_id]
    if building_df.empty:
        return []
        
    return sorted(building_df['sensor_id'].unique())

def filter_data_by_building_and_sensor(df, building_id, sensor_selection):
    """
    Filter dataframe by building and sensor selection
    """
    # Filter by building
    building_df = df[df['building_id'] == building_id].copy()

    if sensor_selection == "Sum":
        # Sum numerical columns across all sensors in this building, grouped by timestamp
        numerical_cols = building_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in ['building_id', 'sensor_id']:
            if col in numerical_cols:
                numerical_cols.remove(col)

        if numerical_cols:
            return building_df.groupby(building_df.index)[numerical_cols].sum()
        else:
            return pd.DataFrame()
    else:
        # Return data for a specific sensor
        sensor_df = building_df[building_df['sensor_id'] == sensor_selection].copy()
        result_cols = [col for col in sensor_df.columns if col not in ['building_id', 'sensor_id']]
        return sensor_df[result_cols] if result_cols else pd.DataFrame()