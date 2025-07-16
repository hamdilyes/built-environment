import pandas as pd
import numpy as np
from typing import List, Dict


def diagnostic_tools_collection():

    def diagnostics_sensor_drift(
            dataframe: pd.DataFrame,
            sensor_cols: List[str],
            window_days: int = 7
    ) -> dict:
        """
        Detect sensor drift by fitting a linear trend over the last window_days for each column.

        Args:
            dataframe (pandas.DataFrame): Slice of data already windowed by the caller.
            sensor_cols (List[str]): List of column names to analyze.
            window_days (int): Number of days of data to include in trend.

        Returns:
            dict: {
                "success" (bool): True if drift calculations succeeded,
                "message" (str): Success or error message,
                "data" (dict): Mapping column -> {"slope_per_day": float}
            }
        """
        try:
            results = {}
            cutoff = dataframe.index.max() - pd.Timedelta(days=window_days)
            for col in sensor_cols:
                if col not in dataframe.columns:
                    results[col] = {"error": "column not found"}
                    continue
                sub = dataframe.loc[cutoff:, [col]].dropna()
                t = (sub.index - sub.index.min()).total_seconds() / 86400
                slope = np.polyfit(t, sub[col], 1)[0]
                results[col] = {"slope_per_day": slope}
            return {"success": True, "message": "Drift computed", "data": results}
        except Exception as e:
            return {"success": False, "message": f"drift error: {e}", "data": {}}


    def diagnostics_spike_detection(
            dataframe: pd.DataFrame,
            sensor_cols: List[str],
            z_thresh: float = 3.0
    ) -> dict:
        """
        Detect spikes where sensor readings exceed a z-score threshold for each column.

        Args:
            dataframe (pandas.DataFrame): Slice of data already windowed by the caller.
            sensor_cols (List[str]): List of columns to check for spikes.
            z_thresh (float): Z-score threshold to flag spikes.

        Returns:
            dict: {
                "success" (bool): True if spike detection succeeded,
                "message" (str): Success or error message,
                "data" (dict): Mapping column -> list of spike timestamps
            }
        """
        results = {}
        for col in sensor_cols:
            if col not in dataframe.columns:
                results[col] = {"n_spikes": 0, "last_spike": None}
                continue
            s = dataframe[col].dropna().astype(float)
            z = (s - s.mean()) / s.std()
            spikes = z.abs() > z_thresh
            results[col] = {
                "n_spikes": int(spikes.sum()),
                "last_spike": str(s[spikes].index.max()) if spikes.any() else None
            }
        return {"success": True, "message": "Spike check done", "data": results}


    def diagnostics_flatline_detection(
            dataframe: pd.DataFrame,
            sensor_cols: List[str],
            window_hours: int = 1
    ) -> dict:
        """
        Detect flatline periods where sensor readings do not change over a time window for each column.

        Args:
            dataframe (pandas.DataFrame): Slice of data already windowed by the caller.
            sensor_cols (List[str]): List of columns to analyze for flatline.
            window_hours (int): Duration in hours to consider a flatline.

        Returns:
            dict: {
                "success" (bool): True if flatline detection succeeded,
                "message" (str): Success or error message,
                "data" (dict): Mapping column -> list of (start,end) timestamp tuples
            }
        """
        # infer sampling period (minutes)
        freq_min = int(dataframe.index.to_series()
                       .diff().dropna().mode()[0].seconds / 60)

        # number of consecutive identical samples required
        flat_len = max(1, int(window_hours * 60 / freq_min))

        results = {}
        for col in sensor_cols:
            if col not in dataframe.columns:
                results[col] = {"n_flat_sections": 0, "longest_flat_min": 0}
                continue

            s = dataframe[col].dropna()
            flat = s.diff().fillna(0) == 0
            grp = (flat != flat.shift()).cumsum()
            spans = flat.groupby(grp).sum()  # length in samples
            long = spans[spans >= flat_len]

            results[col] = {
                "n_flat_sections": int((spans >= flat_len).sum()),
                "longest_flat_min": int(long.max() * freq_min)
                if not long.empty else 0
            }

        return {"success": True, "message": "Flatline check done", "data": results}


    def diagnostics_calibration_check(
            df_setpoint: pd.DataFrame,
            df_sensor: pd.DataFrame,
            setpoint_cols: List[str],
            sensor_cols: List[str],
            tol: float = 1.0
    ) -> dict:
        """
        Compare system setpoints against measured supply temperatures to detect calibration offsets.

        Args:
            df_setpoint (pandas.DataFrame): Slice of data containing set-point columns.
            df_sensor (pandas.DataFrame): Slice of data containing sensor columns.
            setpoint_cols (List[str]): List of setpoint column names.
            sensor_cols (List[str]): List of sensor column names.
            tol (float): Allowable mean offset.

        Returns:
            dict: {
                "success" (bool): True if calibration within tolerance for all cols,
                "message" (str): Success or error message,
                "data" (dict): Mapping column -> diff_mean
            }
        """
        if isinstance(df_setpoint, pd.Series):
            df_setpoint = df_setpoint.to_frame()
        if isinstance(df_sensor, pd.Series):
            df_sensor = df_sensor.to_frame()

        results = {}
        for sp_col, sen_col in zip(setpoint_cols, sensor_cols):
            key = f"{sp_col}->{sen_col}"
            if sp_col not in df_setpoint.columns or sen_col not in df_sensor.columns:
                results[key] = None
                continue
            merged = pd.merge_asof(
                df_setpoint[[sp_col]].sort_index(),
                df_sensor[[sen_col]].sort_index(),
                left_index=True, right_index=True
            )
            results[key] = float((merged[sen_col] - merged[sp_col]).mean())
        return {"success": True, "message": "Calibration check done", "data": results}


    def diagnostics_range_check(dataframe: pd.DataFrame,
                                columns: Dict[str, tuple[float, float]]) -> dict:
        """
        Flags samples outside user-defined (min, max) bounds.

        Args:
            dataframe (pandas.DataFrame): Slice of data already windowed by the caller.
            columns   : {"col_name": (min, max), …}

        Returns:
            dict: {
                "success": True,
                "message" (str): Success or error message,
                "data": {"col": {"pct_out": 0.02, "n_out": 123}}
            }

        """
        out = {}
        for col, (lo, hi) in columns.items():
            s = dataframe[col].dropna()
            mask = (s < lo) | (s > hi)
            out[col] = {"pct_out": round(mask.mean(), 4),
                        "n_out": int(mask.sum())}
        return {"success": True, "message": "Range check done", "data": out}


    def diagnostics_gap_check(dataframe: pd.DataFrame,
                              max_allowed_gap_min: int = 60) -> dict:
        """
        Computes the longest data gap and % missing rows in the last 24 h.

        Args:
            dataframe (pandas.DataFrame): Slice of data already windowed by the caller.
            max_allowed_gap_min : int, optional
            Consecutive minutes without data tolerated before the interval is
            counted as a *gap* (default = 60 min).

        Returns:
            dict:{
                "success": True,
                "message": str,
                "data": {
                    "pct_missing"     : float,   # fraction of expected samples
                    "longest_gap_min" : int      # length of the single longest gap
                }
            }
        """
        freq = dataframe.index.inferred_freq or '15T'
        expected = pd.date_range(dataframe.index.min(),
                                 dataframe.index.max(),
                                 freq=freq)
        missing = expected.difference(dataframe.index)
        gaps = pd.Series(missing).diff().gt(pd.to_timedelta(max_allowed_gap_min, 'min')).cumsum()
        gap_lengths = (missing.to_series().groupby(gaps).count() *
                       pd.to_timedelta(freq)).dt.total_seconds() / 60
        return {"success": True,
                "message": "Gap check done",
                "data": {"pct_missing": round(len(missing) / len(expected), 3),
                         "longest_gap_min": int(gap_lengths.max() if not gap_lengths.empty else 0)}}

    return [
        diagnostics_sensor_drift,
        diagnostics_spike_detection,
        diagnostics_flatline_detection,
        diagnostics_calibration_check,
        diagnostics_range_check,
        diagnostics_gap_check,
    ]