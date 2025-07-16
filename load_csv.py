"""
CSV Handler Module for processing and combining uploaded CSV files
Handles timestamp detection, parsing, and data combination with column differentiation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import streamlit as st
from typing import List, Optional, Tuple, Dict
import io
from pathlib import Path

# Common timestamp column names to look for
TIMESTAMP_PATTERNS = [
    'timestamp', 'time', 'datetime', 'date_time', 'dt', 'date',
    'created_at', 'updated_at', 'time_stamp', 'timeStamp',
    'Date', 'Time', 'DateTime', 'Timestamp', 'DATE', 'TIME'
]

# Common timestamp formats to try
TIMESTAMP_FORMATS = [
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d',
    '%d/%m/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M:%S',
    '%d-%m-%Y %H:%M:%S',
    '%m-%d-%Y %H:%M:%S',
    '%Y/%m/%d %H:%M:%S',
    '%d/%m/%Y',
    '%m/%d/%Y',
    '%d-%m-%Y',
    '%m-%d-%Y',
    '%Y/%m/%d',
    '%Y%m%d',
    '%Y%m%d%H%M%S',
    'ISO8601'
]

def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if col.lower().strip() in [pattern.lower() for pattern in TIMESTAMP_PATTERNS]:
            return col
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in TIMESTAMP_PATTERNS:
            if pattern.lower() in col_lower or col_lower in pattern.lower():
                return col
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                date_like_count = 0
                for val in sample_values:
                    val_str = str(val).strip()
                    if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val_str) or \
                       re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', val_str) or \
                       re.search(r'\d{4}\d{2}\d{2}', val_str):
                        date_like_count += 1
                if date_like_count >= len(sample_values) * 0.7:
                    return col
    return None

def parse_timestamp_column(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df_copy = df.copy()
    timestamp_series = df_copy[timestamp_col]
    parsed_timestamp = None
    try:
        parsed_timestamp = pd.to_datetime(timestamp_series, infer_datetime_format=True)
    except Exception:
        for fmt in TIMESTAMP_FORMATS:
            try:
                if fmt == 'ISO8601':
                    parsed_timestamp = pd.to_datetime(timestamp_series, format='ISO8601')
                else:
                    parsed_timestamp = pd.to_datetime(timestamp_series, format=fmt)
                break
            except:
                continue
        if parsed_timestamp is None:
            try:
                parsed_timestamp = pd.to_datetime(timestamp_series, errors='coerce')
                if parsed_timestamp.isna().sum() >= len(parsed_timestamp) * 0.5:
                    parsed_timestamp = None
            except:
                pass
    if parsed_timestamp is not None:
        df_copy.index = parsed_timestamp
        df_copy = df_copy.drop(columns=[timestamp_col])
        df_copy = df_copy[~df_copy.index.isna()]
        return df_copy
    else:
        raise ValueError(f"Could not parse timestamp column '{timestamp_col}' with any known format")

def get_file_identifier(filename: str) -> str:
    identifier = Path(filename).stem
    identifier = re.sub(r'[^\w\-]', '_', identifier)
    identifier = re.sub(r'_+', '_', identifier)
    return identifier.strip('_')

def differentiate_columns(dfs: List[pd.DataFrame], filenames: List[str], method: str = 'suffix') -> List[pd.DataFrame]:
    if len(dfs) <= 1:
        return dfs
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    column_counts = {}
    for df in dfs:
        for col in df.columns:
            column_counts[col] = column_counts.get(col, 0) + 1
    duplicate_columns = {col for col, count in column_counts.items() if count > 1 and col != 'source_file'}
    if not duplicate_columns:
        return dfs
    differentiated_dfs = []
    for df, filename in zip(dfs, filenames):
        df_copy = df.copy()
        file_id = get_file_identifier(filename)
        rename_map = {}
        for col in df_copy.columns:
            if col in duplicate_columns:
                new_name = f"{col}_{file_id}" if method == 'suffix' else f"{file_id}_{col}"
                rename_map[col] = new_name
        if rename_map:
            df_copy = df_copy.rename(columns=rename_map)
        differentiated_dfs.append(df_copy)
    return differentiated_dfs

def process_single_csv(uploaded_file) -> Tuple[pd.DataFrame, str]:
    try:
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                
                # Drop unnamed columns
                unnamed_cols = [col for col in df.columns if re.match(r'^Unnamed:', col)]
                if unnamed_cols:
                    df.drop(columns=unnamed_cols, inplace=True)
                    
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"Could not read {uploaded_file.name} with any encoding")
        timestamp_col = detect_timestamp_column(df)
        if timestamp_col is None:
            raise ValueError(f"No timestamp column found in {uploaded_file.name}")
        processed_df = parse_timestamp_column(df, timestamp_col)
        processed_df['source_file'] = uploaded_file.name
        return processed_df, uploaded_file.name
    except Exception as e:
        raise e

def detect_granularity(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna()
    most_common_delta = deltas.mode()
    if not most_common_delta.empty:
        return most_common_delta.iloc[0]
    return None

def check_granularity_consistency(dfs: List[pd.DataFrame], filenames: List[str]) -> bool:
    granularities = {}
    for df, filename in zip(dfs, filenames):
        granularity = detect_granularity(df.index)
        if granularity is not None:
            granularities[filename] = granularity
    if len(set(granularities.values())) > 1:
        st.info("🕒 Uploaded files have inconsistent time granularities:")
        for file, gran in granularities.items():
            st.write(f"- `{file}`: {gran}")
        st.error("Please upload files with the same timestamp granularity.")
        return False
    return True

def combine_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    try:
        combined_df = pd.concat(dfs, axis=0, sort=True)
        combined_df = combined_df.sort_index()
        if combined_df.index.duplicated().any():
            numerical_cols = combined_df.select_dtypes(include=[np.number]).columns
            non_numerical_cols = combined_df.select_dtypes(exclude=[np.number]).columns
            agg_dict = {col: 'mean' for col in numerical_cols}
            agg_dict.update({col: 'first' for col in non_numerical_cols})
            combined_df = combined_df.groupby(combined_df.index).agg(agg_dict)
        return combined_df
    except Exception as e:
        raise e

def validate_combined_data(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    if len(df.select_dtypes(include=[np.number]).columns) == 0:
        return False
    return True

def process_uploaded_csvs(uploaded_files, column_naming_method: str = 'suffix') -> pd.DataFrame:
    if not uploaded_files:
        return pd.DataFrame()
    processed_dfs = []
    filenames = []
    for uploaded_file in uploaded_files:
        try:
            df, filename = process_single_csv(uploaded_file)
            if not df.empty:
                processed_dfs.append(df)
                filenames.append(filename)
        except Exception as e:
            continue
    if len(processed_dfs) > 1:
        if not check_granularity_consistency(processed_dfs, filenames):
            return pd.DataFrame()
        processed_dfs = differentiate_columns(processed_dfs, filenames, method=column_naming_method)
    if processed_dfs:
        combined_df = combine_dataframes(processed_dfs)
        if validate_combined_data(combined_df):
            return combined_df
    return pd.DataFrame()

def preview_column_conflicts(uploaded_files) -> Dict[str, List[str]]:
    column_file_map = {}
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, nrows=0)
            for col in df.columns:
                if col not in column_file_map:
                    column_file_map[col] = []
                column_file_map[col].append(uploaded_file.name)
        except Exception as e:
            continue
    conflicts = {col: files for col, files in column_file_map.items() if len(files) > 1}
    return conflicts