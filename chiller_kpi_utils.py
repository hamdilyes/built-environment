"""
chiller_kpi_utils.py  –  KPI utilities for LOB‑17 chiller plant
===============================================================
A **self‑contained helper module** that wraps the most common chiller
Key‑Performance‑Indicators (KPIs).
**exact definitions based on which below functions are built**:

* **COP** – kWth_out / kWe_in
* **EER** – COP × 3.412 (BTU/hr/W)
* **kW/ton** – kWe_in / refrigeration tons delivered
* **Capacity Utilization** – Percentage of rated capacity (300 TR default) in use
* **Load Factor** – Average / peak load over a period
* **Operating Hours** – Runtime hours (plus optional breakdown by load range)
* **Start/Stop Cycles** – Count and table of cycling events

All KPI functions accept **pandas Series** indexed on timestamp.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np
from typing import List, Dict


CHILLERS = ["CH1", "CH2", "CH3"]
ROLLOVER = 1000000

def diff_cum(ser: pd.Series, span=ROLLOVER) -> pd.Series:
    d = ser.diff()
    d[d < 0] += span        # roll-over correction
    d.iloc[0] = 0.0
    return d

# ────────────────────────────────────────────────────────────────
#  Core KPI functions
# ────────────────────────────────────────────────────────────────

def _to_series(obj):
    """
    Accept Series *or* single-column DataFrame → return Series.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError("Input DataFrame has more than one column.")
        return obj.iloc[:, 0]        # the only column
    return obj

def compute_cop(cooling_kw: pd.Series, input_kw: pd.Series) -> pd.Series:
    """Coefficient of Performance – **COP = kWth_out / kWe_in**."""
    return cooling_kw / input_kw.replace(0, np.nan)

def chiller_efficiency_real(
        kWth: pd.Series | pd.DataFrame,
        ch_kw:  pd.Series | pd.DataFrame,
) -> pd.Series:
    """
    Real‑time COP* for **one chiller**.

    Parameters
    ----------
    kWth   : Series  – cooling load in kW (thermal) per 15‑min step.
    ch_kw  : Series or single‑column DataFrame – electrical kW for
             *that same* chiller on the identical timestamp grid.

    Returns
    -------
    Series of COP*  (NaN where electrical kW == 0)
    """
    # --- ensure 1‑D Series -------------------------------------------------
    if isinstance(kWth, pd.DataFrame):
        if kWth.shape[1] != 1:
            raise ValueError("kWth must be a Series or 1‑col DataFrame")
        kWth = kWth.iloc[:, 0]

    if isinstance(ch_kw, pd.DataFrame):
        if ch_kw.shape[1] != 1:
            raise ValueError("ch_kw must be a Series or 1‑col DataFrame")
        ch_kw = ch_kw.iloc[:, 0]

    # --- align indexes -----------------------------------------------------
    common_idx = kWth.index.intersection(ch_kw.index)
    kWth = kWth.reindex(common_idx)
    ch_kw = ch_kw.reindex(common_idx)

    # --- COP calculation ---------------------------------------------------
    cop_series = kWth / ch_kw.replace(0, np.nan)
    cop_series.name = "COP*"

    return cop_series


def estimate_chiller_thermal_kW(
        capacity_pct: pd.Series,
        nominal_chiller_capacity_kWth: float
) -> pd.Series:
    """
    Estimates a chiller's thermal output (kWth) based on its capacity percentage
    and nominal full-load capacity.

    Args:
        capacity_pct (pd.Series): Chiller capacity percentage (0-100%).
        nominal_chiller_capacity_kWth (float): The chiller's rated full-load
                                               cooling capacity in kW (thermal).

    Returns:
        pd.Series: Estimated thermal output (kWth) for the chiller.
    """
    # Convert capacity percentage to Part Load Ratio (0-1)
    plr = capacity_pct / 100.0

    # Ensure PLR is not negative or significantly above 1
    plr = plr.clip(lower=0.0, upper=1.0)

    # Estimated thermal output = Nominal Capacity * PLR
    estimated_kWth = nominal_chiller_capacity_kWth * plr

    # Handle cases where chiller is likely off (e.g., 0% capacity)
    # If capacity_pct is 0, then estimated_kWth will be 0, which is correct.
    return estimated_kWth


def calculate_estimated_chiller_cop(
        ch_kw_series: pd.Series,
        ch_capacity_pct_series: pd.Series,
        nominal_chiller_capacity_kWth: float,
        min_ch_kw: float = 0.5  # Minimum electrical kW to avoid division by near-zero
) -> pd.Series:
    """
    Calculates estimated COP for a single chiller using its electrical input
    and estimated thermal output based on capacity percentage.

    Args:
        ch_kw_series (pd.Series): Electrical power (kW) for the chiller.
        ch_capacity_pct_series (pd.Series): Capacity percentage (0-100%) for the chiller.
        nominal_chiller_capacity_kWth (float): The chiller's rated full-load
                                               cooling capacity in kW (thermal).
        min_ch_kw (float): Minimum electrical kW threshold. COP will be NaN if
                           ch_kw is below this to avoid inflated COPs at very low loads.

    Returns:
        pd.Series: Estimated COP for the chiller.
    """
    # 1. Align indexes
    common_idx = ch_kw_series.index.intersection(ch_capacity_pct_series.index)
    ch_kw_aligned = ch_kw_series.reindex(common_idx)
    ch_capacity_pct_aligned = ch_capacity_pct_series.reindex(common_idx)

    # 2. Estimate thermal output
    estimated_kWth = estimate_chiller_thermal_kW(
        ch_capacity_pct_aligned,
        nominal_chiller_capacity_kWth
    )

    # 3. Calculate estimated COP
    # Replace electrical kW values close to zero with NaN to avoid division by zero
    # or unrealistically high COPs when the chiller is barely consuming power.
    ch_kw_filtered = ch_kw_aligned.mask(ch_kw_aligned < min_ch_kw, np.nan)

    estimated_cop = estimated_kWth / ch_kw_filtered
    estimated_cop.name = "Estimated COP"

    return estimated_cop


def compute_eer(cop: pd.Series) -> pd.Series:
    """Energy‑Efficiency‑Ratio – **EER = COP × 3.412** (IP units)."""
    return cop * 3.412


def kw_per_ton(input_kw: pd.Series, cooling_tons: pd.Series) -> pd.Series:
    """**kW per refrigeration ton** – lower means better (0.45‑0.65 modern)."""
    return input_kw / cooling_tons.replace(0, np.nan)


def capacity_utilization_pct(
    cooling_tons: pd.Series,
    rated_tons: float = 300.0,
) -> pd.Series:
    """**Capacity Utilization (%)** – delivered tons ÷ rated tons × 100."""
    return 100.0 * cooling_tons / rated_tons


def load_factor_v2(
    load_kw: pd.Series,
    window: str = "D",
) -> pd.Series:
    """Average / peak load over the resample *window* (daily by default)."""
    avg = load_kw.resample(window).mean()
    peak = load_kw.resample(window).max()
    return avg / peak.replace(0, np.nan)

def load_factor(
    kwh_series: pd.Series | pd.DataFrame,
    interval_minutes: int = 15,
    by: str | None = None,
) -> pd.Series | float:
    """
    Load Factor = (average kW) / (peak kW)

    Parameters
    ----------
    kwh_series : Series or single-column DataFrame
        Interval kWh (not kW!).  Must have a DateTimeIndex.
    interval_minutes : int, default 15
        Resolution of the data in minutes.  60 / interval_minutes is the
        multiplier to convert kWh → kW.
    by : str, optional
        If None → return one load factor for the entire span.
        Pass a pandas offset alias ('M' for month, 'W' for week, 'D' for day)
        to get a grouped result.

    Returns
    -------
    float  (or Series if `by` given)
    """

    # Make sure we are working with a Series
    if isinstance(kwh_series, pd.DataFrame):
        if kwh_series.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column")
        kwh_series = kwh_series.iloc[:, 0]

    kwh_series = kwh_series.dropna()
    kw_series = kwh_series * (60 / interval_minutes)  # convert to kW

    if by is None:
        total_kwh = kwh_series.sum()
        hours = len(kwh_series) * interval_minutes / 60
        avg_kw = total_kwh / hours
        peak_kw = kw_series.max()
        return round(avg_kw / peak_kw, 3)

    # group-by (month, week, etc.)
    grouped = (
        pd.DataFrame({"kWh": kwh_series, "kW": kw_series})
        .groupby(pd.Grouper(freq=by))
        .agg(total_kWh=("kWh", "sum"),
             peak_kW =("kW",  "max"),
             intervals=("kWh", "count"))
    )
    grouped["avg_kW"] = grouped["total_kWh"] * (60 / interval_minutes) / grouped["intervals"]
    grouped["load_factor"] = (grouped["avg_kW"] / grouped["peak_kW"]).round(3)
    return grouped["load_factor"]

def ch_runtime_diff(runhr: pd.Series, minutes: int = 15) -> pd.Series:
    """
    Run-hours counter → ON/OFF minutes per interval.
    Works for the typical BMS tag that *increments while ON* and is flat while OFF.
    """
    # 1) coarse diff in *hours*
    dh = runhr.diff().clip(lower=0)          # negative jumps = counter reset
    # 2) convert to minutes actually ON in the interval
    on_minutes = dh * 60                     # 1h increment means 60 minutes ON
    # 3) quantise to the sampling period (handles rounding noise)
    return (on_minutes.round().clip(upper=minutes))


def daily_runtime(on_minutes: pd.Series) -> pd.DataFrame:
    on_flag      = on_minutes > 0
    cycle_start  = on_flag & (~on_flag.shift(fill_value=False))

    daily = pd.DataFrame({
        "Run_min" : on_minutes.resample('D').sum(),
        "Cycles"  : cycle_start.resample('D').sum(),
    })

    # --- short cycles ----------------------------------------------------
    mask_short   = (on_minutes > 0) & (on_minutes < 20)
    short        = on_minutes[mask_short]
    daily["ShortCycles"] = short.groupby(short.index.normalize()).count()

    daily["AvgCycleMin"] = daily["Run_min"] / daily["Cycles"].replace(0, pd.NA)
    return daily


def detect_cycles_minutes(run_minutes: pd.Series,
                          cap_pct: pd.Series,
                          min_on: int,
                          min_off: int) -> pd.DataFrame:
    """
    Returns a table of ON→OFF and OFF→ON events with duration & capacity.
    """
    # ON flag per 15-min interval
    on = run_minutes > 0
    chg = on.astype(int).diff().fillna(0)

    events = []
    last_start = None
    for ts, delta in chg.items():
        if delta == +1:          # OFF → ON
            last_start = ts
        elif delta == -1 and last_start is not None:  # ON → OFF
            dur_min = run_minutes.loc[last_start:ts].sum()
            if dur_min >= min_on:
                max_cap = cap_pct.loc[last_start:ts].max()
                events.append(
                    {"Start": last_start,
                     "Stop":  ts,
                     "Duration_min": dur_min,
                     "MaxCap_%": max_cap}
                )
            last_start = None

    cycles = pd.DataFrame(events)
    # Mark *short* ON or OFF periods
    cycles["ShortCycle"] = cycles["Duration_min"] < min_off
    return cycles

def operating_hours(run_minutes: pd.Series, window: str = "D") -> pd.Series:
    """Total operating **hours** in each resample *window*."""
    return run_minutes.resample(window).sum() / 60.0


def operating_hours_by_load_range(
    run_minutes: pd.Series,
    cap_pct: pd.Series,
    bins: Sequence[int] = (0, 25, 50, 75, 100),
    window: str = "D",
) -> pd.DataFrame:
    """Return hours‑run and share‑of‑runtime for each load band."""
    labels = [f"{bins[i]}‑{bins[i+1]}%" for i in range(len(bins) - 1)]
    cat = pd.cut(cap_pct, bins=bins, labels=labels, right=False)

    df = pd.DataFrame({"Minutes": run_minutes, "Band": cat}).dropna()

    # use the 'Minutes' column directly so the result has a simple Index
    minutes = (
        df.groupby([pd.Grouper(freq=window), "Band"], observed=False)["Minutes"]
          .sum()
          .unstack(fill_value=0)
    )

    hours = minutes / 60.0
    perc  = hours.div(hours.sum(axis=1), axis=0).mul(100.0)

    # prepend clear prefixes
    hours.columns = [f"Hrs_{c}" for c in hours.columns]
    perc.columns  = [f"%_{c}"  for c in perc.columns]

    return pd.concat([hours, perc], axis=1)


def start_stop_cycles(
    run_minutes: pd.Series,
    cap_pct: pd.Series,
    min_on: int,
    min_off: int,
) -> pd.DataFrame:
    """Return detailed cycle table using *detect_cycles_minutes*."""
    return detect_cycles_minutes(run_minutes, cap_pct, min_on, min_off)


def chiller_performance_index(
    actual_cop: pd.Series,
    design_cop: float = 5.5,
) -> pd.Series:
    """Simple **CPI = Actual COP / Design COP** (≥ 1 ⇒ better than design)."""
    return actual_cop / design_cop

# ----------------------------------------------------------------
#  Convenience wrapper – compute everything in one shot
# ----------------------------------------------------------------

def compute_kpi_suite(
    *,
    cooling_kw: pd.Series,
    cooling_tons: pd.Series,
    input_kw: pd.Series,
    input_ton,
    runhr_series: pd.Series,
    cap_pct_series: pd.Series,
    mdb_kw: pd.Series,
    resample_window: str = "D",
) -> Dict[str, pd.Series | pd.DataFrame]:
    """Compute and return all KPIs in a dict – **definitions‑accurate**.

    Parameters
    ----------
    cooling_kw        : Cooling load in **kWth**.
    cooling_tons      : Cooling load in **tons**.
    input_kw          : Electrical input **kWe**.
    runhr_series      : Cumulative run‑hours from BMS.
    cap_pct_series    : Instantaneous %‑load.
    design_cop        : Nameplate COP for CPI.
    resample_window   : Aggregation window (daily default).
    """

    # 1) Convert run‑hours counter → minutes per interval
    run_minutes = ch_runtime_diff(runhr_series)

    # 2) Primary KPIs (intrinsic timestamp granularity)
    cop = chiller_efficiency_real(cooling_kw, input_kw)
    #cop = chiller_efficiency_real(cooling_kw, input_kw, all_chillrs)
    # Assume Chiller 1 is a 300 Ton chiller
    NOMINAL_CHILLER_CAPACITY_TONS = 300
    NOMINAL_CHILLER_CAPACITY_KWTH = NOMINAL_CHILLER_CAPACITY_TONS * 3.516
    cop = calculate_estimated_chiller_cop(
        input_kw,
        cap_pct_series,
        NOMINAL_CHILLER_CAPACITY_KWTH
    )

    good = (cop > 0) & (cop < 100)
    cop_clean = cop.where(good).dropna(how='all')
    eer = compute_eer(cop_clean)
    kw_ton = kw_per_ton(input_kw, cooling_tons)

    # cap_util = capacity_utilization_pct(cooling_tons)
    cap_util = capacity_utilization_pct(input_ton)

    #lf = load_factor(input_kw, window=resample_window)
    lf = load_factor(mdb_kw, by='D')
    op_hours = operating_hours(run_minutes, window=resample_window)
    op_hrs_by_band = operating_hours_by_load_range(run_minutes, cap_pct_series, window=resample_window)
    cycles_tbl = start_stop_cycles(run_minutes, cap_pct_series, min_on=0, min_off=16)

    return {
        "COP": cop,
        "EER": eer,
        "kW_per_ton": kw_ton,
        "CapacityUtil_%": cap_util,
        "LoadFactor": lf,
        "OperatingHours": op_hours,
        "OperatingHoursByBand": op_hrs_by_band,
        "CyclesTable": cycles_tbl,
    }