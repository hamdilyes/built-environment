from pathlib import Path
from typing import Dict, List
import pandas as pd


TOOL_DEFAULTS: Dict[str, Dict[str, int]] = {
    # ── Data-quality / sensor sanity  (run every night, 24 h slice) ──
    "diagnostics_range_check"         : {"window_days": 1},
    "diagnostics_gap_check"           : {"window_days": 1},
    "diagnostics_spike_detection"     : {"window_days": 1},
    "diagnostics_flatline_detection"  : {"window_days": 1},
    "diagnostics_sensor_drift"        : {"window_days": 1},
    "diagnostics_calibration_check"   : {"window_days": 1}
}


def _load_and_index(file_path: str) -> pd.DataFrame:
    """
    Helper: read CSV, detect and parse timestamp column, then set it as the index.

    Args:
        file_path (str): Path to the CSV file containing a timestamp column.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp' parsed and set as index.

    Raises:
        ValueError: If no timestamp column is detected.
    """
    df = pd.read_csv(file_path)
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)
    if time_col is None:
        raise ValueError('No timestamp column found')
    df['timestamp'] = pd.to_datetime(df[time_col], format='mixed', dayfirst=False)
    return df.set_index('timestamp')


def _make_deltat(df: pd.DataFrame, sup: str, ret: str) -> pd.Series:
    return (df[ret] - df[sup]).rename("deltaT")


def run_all_diagnostics_nightly(toolbox, paths: Dict[str, Path]) -> Dict[str, dict]:
    """
    One-call wrapper that:
       1) loads all CSVs from *paths*,
       2) applies the look-back windows in TOOL_DEFAULTS,
       3) builds the correct payload for every tool returned by
          diagnostic_tools_collection(), and
       4) returns the nested results dict.

    Parameters
    ----------
    paths : Dict[str, pathlib.Path]
        Expected keys (min): 'sensor', 'setpoint', 'chillers',
        'oat', 'fahu'.  Add more if individual tools need extra files.

    Returns
    -------
    Dict[str, dict]
        {tool_name: diagnostic_output}
    """
    # -------- 1. load each CSV once ---------------------------------
    dfs = {k: _load_and_index(p) for k, p in paths.items()}
    END = max(df.index.max() for df in dfs.values())

    # ---------- build core series (edit column names as needed) ----
    df_sensor = dfs["sensor"]
    deltaT_plant = _make_deltat(df_sensor,
                                "BTU-01_SupTemp_degC",
                                "BTU-01_RetTemp_degC")
    flow_series = df_sensor["BTU-01_ChwFlow_L/s"]

    df_chws = dfs["setpoint"]
    deltaT_header = _make_deltat(df_chws,
                                 "CHW-SYS_SupHdrTemp (°C)",
                                 "CHW-SYS_RetHdrTemp (°C)")
    sp_series = df_chws["CHW-SYS_ChwSetpt (°C)"]

    df_fahu = dfs["fahu"]
    fahu_flag = (df_fahu["FAHU-01_SupFanRunHr (Hours)"].diff() > 0).astype(int)
    hrw_flag = (df_fahu["FAHU-01_HRWRunHr (Hours)"].diff() > 0).astype(int)

    df_chl = dfs['chillers']
    cap_cols = [c for c in df_chl.columns if c.endswith('_ChlrCap (%)')]
    ch_load_pct = df_chl[cap_cols]
    run_cols = [c for c in df_chl.columns if c.endswith("_RunHr (Hours)")]

    oat_series = dfs["oat"]["OAT-SENSOR_OutAirTemp (°C)"]

    def _slice(obj, days):
        return obj.loc[END - pd.Timedelta(days=days): END]

    # -------- 3. build payloads dict --------------------------------
    payloads: Dict[str, dict] = {}

    for name, cfg in TOOL_DEFAULTS.items():
        days = cfg["window_days"]

        if name == "diagnostics_range_check":
            # customise bounds to your sensors
            bounds = {
                "BTU-01_SupTemp_degC": (0, 60),
                "BTU-01_RetTemp_degC": (0, 60),
                "BTU-01_ChwFlow_L/s": (0, 50),
            }
            payloads[name] = {
                "dataframe": _slice(df_sensor, days),
                "columns": bounds
            }

        elif name == "diagnostics_gap_check":
            payloads[name] = {
                "dataframe": _slice(df_sensor, days),
                "max_allowed_gap_min": 60
            }

        elif name == "diagnostics_spike_detection":
            payloads[name] = {
                "dataframe": _slice(df_sensor, days),
                "sensor_cols": ["BTU-01_SupTemp_degC", "BTU-01_RetTemp_degC"]
            }

        elif name == "diagnostics_flatline_detection":
            payloads[name] = {
                "dataframe": _slice(df_sensor, days),
                "sensor_cols": ["BTU-01_SupTemp_degC", "BTU-01_RetTemp_degC"]
            }

        elif name == "diagnostics_sensor_drift":
            payloads[name] = {
                "dataframe": _slice(df_sensor, days),
                "sensor_cols": ["BTU-01_SupTemp_degC"]
            }

        elif name == "diagnostics_calibration_check":
            payloads[name] = {
                "df_setpoint": _slice(sp_series, days),
                "df_sensor"  : _slice(df_sensor,   days),
                "setpoint_cols": ["CHW-SYS_ChwSetpt (°C)"],
                "sensor_cols": ["BTU-01_SupTemp_degC"]
            }

    # -------- 4. execute every SimpleTool in this collection -------
    results = {}
    for tool in toolbox:  # <— import from same file
        if tool.name not in payloads:
            continue
        try:
            results[tool.name] = tool(**payloads[tool.name])
        except Exception as exc:
            results[tool.name] = {"success": False, "message": str(exc), "data": {}}

    return results