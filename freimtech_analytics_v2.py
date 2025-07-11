#!/usr/bin/env python3
"""
freimtech_analytics_v2.py
Author : Hasan
Date   : 2025-05-16

Analyse 15-min data from Freimtech
Outputs Plotly figures in HTML format
"""

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from dateutil import tz
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from pandas.tseries.frequencies import to_offset
import freimtech_rootcause as rc
import plots as fig

# ───────────────────────────────────  folders / files  ─────────────────────────
DATA_DIR  = pathlib.Path("freimtech_data")  #this is current path where all CSV files are placed.
PLOTS_DIR = pathlib.Path("plots_v2")  #plots will be stored in a new directory "plots_v2".
PLOTS_DIR.mkdir(exist_ok=True)

CSV_FILES = {
    "btu":   "LOB17_BTUMtr.csv",
    "chws":  "LOB17_ChwSys.csv",
    "chiller":  "LOB17_Chillers.csv",
    "mdb":   "LOB17_MDB-kWh.csv",
    "oat":   "LOB17_OAT.csv",
    "fahu": "LOB17_FAHU.csv",
}

#TZ = tz.gettz("Asia/Dubai")
ROLLOVER = 1000000                          # kWh counter wrap value

# ─────────────────────────────  generic helpers  ───────────────────────────────
def make_index_naive(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Strip timezone information from the index in-place.
    Works for both tz-aware DatetimeIndex and tz-naive.
    """
    idx = obj.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        obj.index = idx.tz_convert(None)   # faster than tz_localize(None)
    return obj

def read_csv_ts(path: pathlib.Path, time_col: str, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[time_col], **kw)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    return df

def build_df_btu(data_dir: pathlib.Path,
                 file_name="LOB17_BTUMtr.csv") -> pd.DataFrame:
    """
    Returns a DataFrame indexed by local timestamp (15-min)
    with numeric columns:
        BTU-01_ChwActEnergyDlvd_kWh
        BTU-01_SupTemp_degC
        BTU-01_RetTemp_degC
        (and whatever extra columns we may decide to keep)
    """
    path = data_dir / file_name
    use = ["DateTime",
           "BTU-01_ChwActEnergyDlvd_kWh",
           "BTU-01_ChwFlow_L/s",
           "BTU-01_SupTemp_degC",
           "BTU-01_RetTemp_degC"]

    df = pd.read_csv(path, usecols=use, parse_dates=["DateTime"])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index(['DateTime'])

    # Localise to Gulf Standard Time
    #df.index = df.index.tz_localize(tz_str, nonexistent="shift_forward")

    # coerce any stray text strings to numeric, turn errors into NaN
    for c in ["BTU-01_ChwActEnergyDlvd_kWh",
              "BTU-01_SupTemp_degC",
              "BTU-01_RetTemp_degC"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def diff_cum(ser: pd.Series, span=ROLLOVER) -> pd.Series:
    d = ser.diff()
    d[d < 0] += span        # roll-over correction
    d.iloc[0] = 0.0
    return d


df_oat = make_index_naive(read_csv_ts(              #read OAT csv file
    DATA_DIR / CSV_FILES["oat"],
    time_col="Timestamp",
    usecols=["Timestamp", "OAT-SENSOR_OutAirTemp (°C)"],
))


df_fahu = make_index_naive(read_csv_ts(
    DATA_DIR / CSV_FILES["fahu"],         #  Read "LOB17_FAHU.csv"
    time_col="Timestamp",
    usecols=[
        "Timestamp",
        "FAHU-01_SupFanRunHr (Hours)",
        "FAHU-01_ExFanRunHr (Hours)",
        "FAHU-01_HRWRunHr (Hours)",
        "FAHU-02_SupFanRunHr (Hours)",
        "FAHU-02_ExFanRunHr (Hours)",
        "FAHU-02_HRWRunHr (Hours)",
    ],
))

df_chws = make_index_naive(read_csv_ts(
    DATA_DIR / CSV_FILES["chws"],          # Read "LOB17_ChwSys.csv"
    time_col="Timestamp",
    usecols=[
        "Timestamp",
        "CHW-SYS_SupHdrTemp (°C)",
        "CHW-SYS_RetHdrTemp (°C)",
        "CHW-SYS_ChwSetpt (°C)",
        "CHW-SYS_ChwSetptDay (°C)",
        "CHW-SYS_ChwSetptNight (°C)",
    ],
))

def align_naive(a, b):
    """
    Return two Series/DataFrames with tz-naive, aligned indices.
    """
    a_ = a.copy()
    b_ = b.copy()
    a_.index = a_.index.tz_localize(None)
    b_.index = b_.index.tz_localize(None)
    return a_.align(b_, join="inner", axis=0)


# ─────────────────────────────  Energy Analytics helpers  ─────────────────────────────
def total_energy_consumption(df):
    daily   = df.resample('D').sum()
    weekly  = df.resample('W').sum()
    monthly = df.resample('ME').sum()
    annual  = df.resample('YE').sum()
    contribution = df.sum() / df.sum().sum() * 100
    trend   = daily.sum(axis=1)
    return daily, weekly, monthly, annual, contribution, trend


def load_profile_analysis(df):
    tmp = df.copy()
    tmp['Hour']    = tmp.index.hour
    tmp['Weekday'] = tmp.index.weekday
    weekday = tmp[tmp['Weekday'] < 5].groupby('Hour').mean()
    weekend = tmp[tmp['Weekday'] >= 5].groupby('Hour').mean()
    monthly = tmp.groupby([tmp.index.month, tmp.index.hour]).mean()
    return weekday, weekend, monthly


def peak_demand_analysis(df):
    peak15   = df.max()
    daily_pk = df.resample('D').max()
    ldc      = df.sum(axis=1).sort_values(ascending=False).reset_index(drop=True)
    return peak15, daily_pk, ldc


def daily_peak_demand(df_kwh: pd.DataFrame):
    """
    Return series of *daily* peak kW (total of all MDBs).
    """
    daily_peak_kw = (df_kwh.sum(axis=1) * 4).resample("D").max()
    return daily_peak_kw


def base_load_estimation(df, night_start='22:00', night_end='05:00', pctl=0.05):
    night  = df.between_time(night_start, night_end)
    min_n  = night.min()
    p05    = df.sum(axis=1).quantile(pctl)
    return min_n, p05


def load_balancing_analysis(df):
    """
    tot   : lifetime kWh per MDB
    ratio : share per interval (row-normalised)
    imb   : σ of those shares (imbalance index)
    """
    tot = df.sum()
    ratio = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
    imb = ratio.std(axis=1).fillna(0)

    return tot, ratio, imb


def seasonal_energy_patterns(df):
    tmp = df.copy()
    tmp['Month'] = tmp.index.month
    monthly  = tmp.groupby('Month').sum()
    season   = tmp.groupby([tmp.index.month, tmp.index.hour]).mean()
    return monthly, season


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


def top_peak_intervals(df_kwh: pd.DataFrame, top_n=20):
    """
    Return the *top-n* 15-min intervals ranked by total kWh.
    """
    total = df_kwh.sum(axis=1)
    return total.nlargest(top_n).sort_values(ascending=False)


def print_top_peaks(peaks: pd.Series):
    print("\nTop-{} 15-min demand spikes (kW):".format(len(peaks)))
    for ts, kwh in peaks.items():
        print(f"  {ts:%Y-%m-%d %H:%M}  →  {kwh*4:6.0f} kW")


def overloaded_mdb(df_kwh: pd.DataFrame, threshold_share=0.65) -> pd.DataFrame:
    """
    Returns a mask dataframe with True where any MDB’s share > threshold_share.
    """
    share = df_kwh.div(df_kwh.sum(axis=1), axis=0)
    return share.gt(threshold_share)


def print_overload_stats(mask: pd.DataFrame):
    pct = mask.mean()*100
    print("\nOverload incidence (share >65 % of total load):")
    for col, p in pct.items():
        print(f"  {col:20s}: {p:5.1f} % of intervals")


def rolling_mdb_share(df_kwh: pd.DataFrame, window="30D") -> pd.DataFrame:
    """
    30-day centred rolling mean share per MDB.
    """
    share = df_kwh.div(df_kwh.sum(axis=1), axis=0)
    return share.rolling(window, center=True, min_periods=10).mean()


def detect_trend(series: pd.Series):
    """Mann-Kendall test (null: no trend)."""
    clean = series.dropna()
    if len(clean) < 30:
        return None
    tau, p = kendalltau(np.arange(len(clean)), clean)
    return {"tau": tau, "p": p}


def print_trend_results(roll_share: pd.DataFrame):
    print("\n30-day share trend (Mann-Kendall τ, p-value):")
    for col in roll_share:
        res = detect_trend(roll_share[col])
        if res:
            direction = "↑" if res["tau"] > 0 else "↓"
            print(f"  {col}: τ={res['tau']:+.2f} {direction}  p={res['p']:.3f}")


# ─────────────────────────────  HVAC and Delta-T helpers  ─────────────────────────────
def delta_t_series(
        df_btu: pd.DataFrame,
        sup_name: str = "BTU-01_SupTemp_degC",
        ret_name: str = "BTU-01_RetTemp_degC",
        lo: float = 0.0,
        hi: float = 25.0,
) -> pd.Series:
    """
    Clean ΔT = return – supply.
    Coerces to numeric and discards implausible values (<0 °C or >25 °C).
    """
    sup_col = "BTU-01_SupTemp_degC"
    ret_col = "BTU-01_RetTemp_degC"

    for c in (sup_col, ret_col):
        df_btu[c] = pd.to_numeric(df_btu[c], errors="coerce")

    mask = (
            (df_btu[sup_name] >= lo) & (df_btu[sup_name] <= hi) &
            (df_btu[ret_name] >= lo) & (df_btu[ret_name] <= hi)
    )
    delta = (df_btu.loc[mask, ret_col] - df_btu.loc[mask, sup_col]).rename("ΔT_°C")
    return delta


def delta_t_stats(delta_t: pd.Series, target=5.0):
    """
    Prints mean, std, 75th & 99th percentiles and flags chronic low-ΔT.
    """
    desc = delta_t.describe(percentiles=[0.75, 0.99])
    mean, std = desc['mean'], desc['std']
    q75, q99   = desc['75%'], desc['99%']

    print("\nΔT summary (°C)")
    print(f"  Mean  : {mean:4.1f}")
    print(f"  Std   : {std:4.1f}")
    print(f"  75 %  : {q75:4.1f}")
    print(f"  99 %  : {q99:4.1f}")

    if mean < 0.8*target:
        print("  ⚠  Mean ΔT is <80 % of design – low-ΔT syndrome likely.")


def detect_deltaT_mismatch(
    df_dt: pd.DataFrame,
    col_btu: str = "BTU_ΔT",
    col_hdr: str = "Header_ΔT",
    abs_tol: float = 0.8,  #previously it was 0.5
    iqr_mult: float = 2.5,
) -> pd.DataFrame:
    """
    Flag intervals where BTU ΔT differs abnormally from header ΔT.

    Parameters
    ----------
    df_dt : DataFrame with the two ΔT columns aligned on the index.
    abs_tol : float - handles instrument/sensor measurement errors. 0.3% is typical compunded error
        Always flag if |ΔT_hdr - ΔT_btu| exceeds this °C.
    iqr_mult : float - handles data variability. 2-3xIQR is typical for energy meter data
        Robust threshold: flag if residual exceeds (IQR * iqr_mult)

    Returns
    -------
    DataFrame with original columns plus:
        • residual   = hdr - btu
        • anomaly    = True / False
    """
    df = df_dt[[col_btu, col_hdr]].dropna().copy()
    df["residual"] = df[col_hdr] - df[col_btu]

    # robust spread estimate
    iqr = np.subtract(*np.percentile(df["residual"], [75, 25]))
    lim = max(abs_tol, iqr_mult * iqr)

    df["anomaly"] = df["residual"].abs() >= lim
    return df


def chiller_efficiency(
        df_chiller: pd.DataFrame,
        df_btu: pd.DataFrame,
        energy_col: str | None = None,   #  ← add this default
):
    """
    Very coarse COP proxy:
        COP* ≈ Cooling_kWth / (MDB-2 + MDB-3 kW)
    Requires diff_cum() already applied to MDB boards.
    """

    if energy_col is None:
        # pick the first column that contains 'ActEnergy' and 'kWh'
        hits = [c for c in df_btu.columns if 'ActEnergy' in c and 'kWh' in c]
        if not hits:
            raise ValueError("Cooling-energy column not found in df_btu")
        energy_col = hits[0]

    cooling_kw = diff_cum(df_btu[energy_col]) * 4

    #cooling_kw = diff_cum(df_btu["BTU-01_ChwActEnergyDlvd_kWh"])*4
    elec_kw    = (df_chiller.sum(axis=1))*4      # already kWh → kW
    mask = elec_kw >= 5  # 5 kW threshold filters zero / glitch values
    cop_proxy = (cooling_kw / elec_kw).where(mask)

    # 2) Optional: clip absurdly high COP*
    #cop_proxy = cop_proxy.clip(upper=10)
    #cop_proxy = cop_proxy.clip(lower=0)
    return cop_proxy


def detect_cop_anomalies(cop, lo=0.01, hi=10.0):
    """
    Flag intervals with COP* < lo  OR  COP* > hi.
    Returns DataFrame with 'COP*' and 'anomaly' bool.
    """
    df = cop.to_frame("COP*").copy()
    df["anomaly"] = (df["COP*"] < lo) | (df["COP*"] > hi)
    return df


def check_zero_cop_sensors(cop_df, flow, btu_kW):
    """
    Returns subset of rows where COP*≈0 but flow or BTU kW
    still registers, suggesting sensor or meter issue.
    """
    zero = cop_df[cop_df["COP*"] < 0.01].copy()
    zero["Flow_Ls"] = flow.reindex(zero.index)
    zero["Cooling_kW"] = btu_kW.reindex(zero.index)
    sus = zero[(zero["Flow_Ls"] > 1) | (zero["Cooling_kW"] > 1)]
    return sus


def rank_low_cop_days(cop, n=5):
    """
    Return DataFrame of the n worst days (lowest daily-mean COP*).
    """
    daily = cop.resample('D').mean()
    return daily.nsmallest(n)


def redundant_chiller_op(df_chiller_pct: pd.DataFrame, low=0.3, high=0.8):
    """
    Flags intervals where ≥2 chillers run between `low` and `high`
    (e.g., both at ~40 % each).
    """
    # make sure the dataframe is numeric
    df = df_chiller_pct.apply(pd.to_numeric, errors="coerce")

    # element-wise test:   low*100 ≤ %load ≤ high*100
    mask_low = (df >= low * 100) & (df <= high * 100)

    # redundant if ≥ 2 chillers satisfy the test in the same 15-min slot
    redundant = mask_low.sum(axis=1) >= 2
    return redundant


# ─────────────────────────────────  main  ───────────────────────────────────────
def main():
    # load MDB cumulative and convert to interval kWh
    cols = ["MDB-01_ActEnergyDlvd_14648",
            "MDB-02_ActEnergyDlvd_14649",
            "MDB-03_ActEnergyDlvd_14650"]

    csv_path = DATA_DIR / CSV_FILES['mdb']
    if not csv_path.exists():
        sys.exit(f"ERROR – CSV not found: {csv_path}")
    mdb_cum = read_csv_ts(csv_path, 'Date Time', usecols=['Date Time'] + cols)
    mdb_int = pd.concat({c.replace('ActEnergyDlvd_','kWh_'): diff_cum(mdb_cum[c])
                         for c in cols}, axis=1)
    mdb_int['total_kWh'] = mdb_int.sum(axis=1)    #total MDB load (summed all 3 MDBs)

    df_btu = make_index_naive(build_df_btu(DATA_DIR))   #


    csv_chiller = DATA_DIR / CSV_FILES['chiller']
    if not csv_chiller.exists():
        sys.exit(f"ERROR – CSV not found: {csv_chiller}")
    df_chillers = read_csv_ts(
        csv_chiller,
        time_col="Timestamp"
        #usecols=["Timestamp", "OAT-SENSOR_OutAirTemp (°C)"],
    )
    cap_cols = [c for c in df_chillers.columns if c.endswith("_ChlrCap (%)")]

    # Ensure numeric & clip to 0–100
    df_load_pct = (
        df_chillers[cap_cols]
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=0, upper=100)
        .rename(columns=lambda x: x.replace("_ChlrCap (%)", "_LoadPct"))
    )


    # ── Sanity check: simultaneous chillers? ─────────────────────────
    # count how many ≥ 10 % at each timestamp
    n_on = (df_load_pct > 10)
    n_multi = (n_on.sum(axis=1) > 1)  #number of chillers on simultaneously
    n_multi_index = n_multi[n_multi == True].index

    print("\nChiller loads during multi-on intervals:")

    share = n_multi.sum() / len(n_on)

    print(f"\nSimultaneous-chiller check → {n_multi} intervals "
          f"({share:.2%}) have 2 or more machines above 10 % load.")

    if n_multi.sum() == 0:
        print("✓  Plant runs one chiller at a time → drop n_on from later plots.")
    else:
        print("⚠  Multiple chillers do overlap – keep n_on in analysis.")


    # Whole-building electricity plot
    mdb_cols_only = mdb_int.iloc[:, :3]  # i.e. drop total_kWh
    daily, weekly, monthly, annual, contrib, trend = (total_energy_consumption(mdb_cols_only))
    fig.save(fig.fig_whole_elec(daily), 'electricity_daily')
    fig.save(fig.fig_contribution(contrib), 'electricity_contribution')


    # Load-profile analysis
    weekday, weekend, month_hour = load_profile_analysis(mdb_int[['total_kWh']])
    fig.save(fig.fig_profiles(weekday, weekend), 'profiles_weekday_weekend')
    fig.save(fig.fig_month_heat(month_hour), 'profiles_season_heat')

    # Peak-demand and Load Duration Curve (LDC)
    peak_15min, daily_peak, ldc = peak_demand_analysis(mdb_int[['total_kWh']])

    # Building Load Factor
    print ("Building Load Factor is: ", load_factor(mdb_int['total_kWh']))  # calculated as 15.9%, which is very low --> peak demand is higher than average load

    # base-load numbers for annotation
    (night_min, p05) = base_load_estimation(mdb_int[['total_kWh']])
    base_kw = p05 * 4          # convert  kWh/15-min → kW
    fig.save(fig.fig_load_duration(ldc, base_kw), 'peak_ldc')

    # Base-load visuals
    night_sum_kw = mdb_int.between_time('22:00','05:00').sum(axis=1) * 4
    daily_min_kw = mdb_int.resample('D').min()['total_kWh'] * 4
    fig.save(fig.fig_base_hist(night_sum_kw, base_kw), 'base_histogram')
    fig.save(fig.fig_daily_min(daily_min_kw), 'base_daily_min')


    # Load balancing
    tot, ratio, imb = load_balancing_analysis(mdb_int.iloc[:,:3])
    fig1, fig2, fig3 = fig.fig_load_balance(ratio)
    fig.save(fig1, "share_area")
    fig.save(fig2, "share_24h_line")
    fig.save(fig3, "imbalance_24h")


    # Peak-demand visuals
    peaks = top_peak_intervals(mdb_int.iloc[:, :3])  # only MDB cols
    print_top_peaks(peaks)
    fig.save(fig.fig_top_peaks(peaks), "peak_15min_spikes")

    daily_kw = daily_peak_demand(mdb_int.iloc[:, :3])
    fig.save(fig.fig_daily_peaks(daily_kw), "daily_peak_kw")

    # Overload stats
    over_mask = overloaded_mdb(mdb_int.iloc[:, :3])
    print_overload_stats(over_mask)

    # Pattern shift
    roll_share = rolling_mdb_share(mdb_int.iloc[:, :3])
    print_trend_results(roll_share)
    fig.save(fig.fig_rolling_share(roll_share), "rolling_share")
    

    # 10) Chiller OP
    redundant = redundant_chiller_op(df_load_pct)
    print(f"\nRedundant chiller intervals: {redundant.mean()*100:.1f}%")


    # ΔT visual
    delta_t = delta_t_series(df_btu)
    delta_t_stats(delta_t)
    fig.save(fig.fig_delta_t(delta_t), "delta_t_fullcalendar")


    df_chiller_elec = mdb_int.loc[:, ["MDB-02_kWh_14649", "MDB-03_kWh_14650"]]

    # Align indices just in case there are mismatched time-stamps
    df_chiller_elec, df_btu = df_chiller_elec.align(df_btu, join="inner", axis=0)

    # Calculate Coefficient of Performance (COP) Proxy
    cop_proxy = chiller_efficiency(df_chiller_elec, df_btu)

    # Optional clipping of COP values (to remove outliers)
    cop_proxy = cop_proxy.clip(upper=10)
    cop_proxy = cop_proxy.clip(lower=0)

    # 2) Print headline KPIs
    print("\nChiller-plant COP* proxy:")
    print(f"  Mean  : {cop_proxy.mean():.2f}")
    print(f"  25 %  : {cop_proxy.quantile(0.25):.2f}")
    print(f"  75 %  : {cop_proxy.quantile(0.75):.2f}")
    print(f"  Min   : {cop_proxy.min():.2f}")
    print(f"  Max   : {cop_proxy.max():.2f}")

    fig.save(fig.plot_cop_line(cop_proxy), "cop_proxy_line")
    fig.save(fig.plot_cop_oat(cop_proxy, df_oat), "cop_vs_oat")

    # ---- COP* anomalies -------------------------------------------
    cop_series = cop_proxy.dropna()  # from earlier block

    cop_df = detect_cop_anomalies(cop_series)
    fig_cop_anom = fig.plot_cop_anomaly_scatter(cop_df)
    fig.save(fig_cop_anom, "cop_anomaly_scatter_postprocessed")

    # smoothed curves
    fig.save(fig.plot_cop_timeseries(cop_series, 'D'), "cop_daily_postprocessed")
    fig.save(fig.plot_cop_timeseries(cop_series, 'W'), "cop_weekly_postprocessed")

    # sensor sanity during zero-COP
    suspect = check_zero_cop_sensors(
        cop_df, df_btu["BTU-01_ChwFlow_L/s"],  # flow L/s
        diff_cum(df_btu["BTU-01_ChwActEnergyDlvd_kWh"]) * 4  # kWth
    )
    suspect.to_csv(PLOTS_DIR / "zero_cop_suspects_postprocessed.csv")
    print(f"Zero-COP but non-zero flow/Btu rows: {len(suspect)}   "
          f"saved to zero_cop_suspects_postprocessed.csv")

    # top-5 low-COP days
    low5 = rank_low_cop_days(cop_series, 5)
    print("\n=== 5 lowest-COP days (mean COP*) ===")
    print(low5.round(2))

    # quick equipment state snapshot for each low-COP day
    for d in low5.index:
        span = slice(d, d + pd.Timedelta(days=1))
        print(f"\n--- {d.date()}  ---")
        print(df_load_pct.loc[span].describe())


    # Plant-header vs BTU ΔT
    # build the two Series with their DateTimeIndex intact
    btu_dt = delta_t_series(df_btu)  # index = BTU timestamps
    hdr_dt = (
            df_chws["CHW-SYS_RetHdrTemp (°C)"]
            - df_chws["CHW-SYS_SupHdrTemp (°C)"]
    )  # index = CHW-SYS timestamps
    btu_dt, hdr_dt = btu_dt.align(hdr_dt, join="inner", axis=0)
    df_dt = pd.DataFrame({"BTU_ΔT": btu_dt, "Header_ΔT": hdr_dt}).dropna()
    fig.save(fig.plot_deltaT_comparison(df_dt), "deltaT_compare")


    # detect mismatch anomalies
    mismatch = detect_deltaT_mismatch(df_dt)
    n_anom = mismatch["anomaly"].sum()
    print(f"\nΔT mismatch anomalies: {n_anom} of {len(mismatch):,} intervals "
          f"({n_anom / len(mismatch):.2%})  |  threshold {mismatch['residual'].abs().quantile(0.75) * 2.5:.2f} °C")

    fig.save(fig.plot_deltaT_anomalies(mismatch), "plant_vs_btu_deltaT_anomalies")
    fig.save(fig.plot_deltaT_anomaly_timeline(mismatch, max_anom=100, pick="largest"), "plant_vs_btu_deltaT_anomaly_timeline")

    # 2) Flow vs ΔT (over-pumping check)
    df_flow = pd.DataFrame({
        "ΔT": df_dt["BTU_ΔT"],
        "Flow_Ls": df_btu["BTU-01_ChwFlow_L/s"]
    }).dropna()
    fig.save(fig.plot_flow_vs_deltaT(df_flow), "flow_vs_deltaT")


    # MDB-01 vs FAHU overlay
    fahu_on_1 = (diff_cum(df_fahu["FAHU-01_SupFanRunHr (Hours)"]) > 0).astype(int)
    fahu_on_2 = (diff_cum(df_fahu["FAHU-02_SupFanRunHr (Hours)"]) > 0).astype(int)
    df_overlay = pd.DataFrame({
        "MDB1_kW": mdb_int["MDB-01_kWh_14648"] * 4,
        "MDB2_kW": mdb_int["MDB-02_kWh_14649"] * 4,
        "MDB3_kW": mdb_int["MDB-03_kWh_14650"] * 4,
        "FAHU_1_on": fahu_on_1,
        "FAHU_2_on": fahu_on_2
    }).dropna()
    fig.save(fig.plot_mdb1_fahu_overlay(df_overlay), "mdb1_fahu_overlay5")


    # 4) ΔT vs OAT
    df_oat_align = df_oat.align(df_dt["BTU_ΔT"], axis=0, join="inner")[0]
    oat_series = df_oat_align["OAT-SENSOR_OutAirTemp (°C)"]  # 1-D

    # build the plotting frame
    df_plot = pd.DataFrame({"OAT": oat_series, "ΔT": df_dt["BTU_ΔT"]}).dropna()
    fig.save(fig.plot_deltaT_vs_OAT(df_plot), "deltaT_vs_oat")


    # ΔT vs chiller load
    load_pct = df_load_pct.sum(axis=1)  # total %
    n_on = (df_load_pct > 10).sum(axis=1)  # chiller count
    df_load = pd.DataFrame({"Load_pct": load_pct, "N_on": n_on, "ΔT": df_dt["BTU_ΔT"]}).dropna()
    #fig.save(fig.plot_deltaT_vs_chillerload(df_load), "deltaT_vs_load")

    # align
    ch_pct_aligned, delta_aligned = df_load_pct.align(
        delta_t_series(df_btu), join="inner", axis=0
    )

    fig_all = fig.plot_all_chillers_vs_deltaT(ch_pct_aligned, delta_aligned)
    fig.save(fig_all, "chiller_load_vs_deltaT_all")


    # Daily ΔT vs HRW fraction
    hrw_on = (diff_cum(df_fahu["FAHU-01_HRWRunHr (Hours)"]) > 0).astype(int)

    #force numeric & single timezone
    hrw_on = pd.to_numeric(hrw_on, errors="coerce")
    hrw_on.index = hrw_on.index.tz_localize(None)

    #ensure a continuous 15-min grid (optional but fool-proof)
    full_15min = pd.date_range(hrw_on.index[0], hrw_on.index[-1], freq="15min")
    hrw_on = hrw_on.reindex(full_15min, fill_value=0)

    #df_dt["BTU_ΔT"].index = df_dt["BTU_ΔT"].index.tz_localize(None)
    df_btu_deltat = df_dt["BTU_ΔT"].resample("D").mean()

    # now resample
    hrw_daily = hrw_on.resample("D").mean()
    df_daily = pd.DataFrame({
        "DailyMean_ΔT": df_btu_deltat,
        "HRW_frac": hrw_daily
    }).dropna()
    fig.save(fig.plot_daily_deltaT_vs_HRW(df_daily), "deltaT_vs_hrw")


    # 7) ΔT recovery during shutdowns (find ≥6-h FAHU-off stretches)
    mask_off = (fahu_on_1 == 0)
    g = (mask_off != mask_off.shift()).cumsum()
    events = []
    hrw_on, df_dt = align_naive(hrw_on, df_dt)

    for grp, chunk in mask_off.groupby(g):
        # keep only rows where supply fan is OFF
        if chunk.sum() * 0.25 >= 6:  # 0.25 h per 15-min slot
            # intersect indices so we don't request missing timestamps
            idx = chunk.index.tz_localize(None).intersection(df_dt.index)
            if idx.empty:
                continue
            win = df_dt.loc[idx, "BTU_ΔT"]
            events.append(pd.DataFrame({
                "Time": idx,
                "ΔT": win.values,
                "Event_ID": f"Ev{grp}"
            }))
    if events:
        events_df = pd.concat(events)
        fig.save(fig.plot_deltaT_recovery(events_df), "deltaT_recovery")
    

    # ── root-cause correlation scan ─────────────────────────────────
    features = rc.build_feature_frame(
        residual=btu_dt, #mismatch["residual"],  #added BTU delta-T instead of Header bs BTU delta-T residual
        flow=df_btu["BTU-01_ChwFlow_L/s"],
        chiller_pct=load_pct,
        chiller1_pct=df_load_pct.iloc[:,0],
        chiller2_pct=df_load_pct.iloc[:, 1],
        chiller3_pct=df_load_pct.iloc[:, 2],
        n_chillers_on=n_on,
        fahu1_on=fahu_on_1,
        fahu2_on=fahu_on_2,
        hrw_on=hrw_on.astype(int),
        oat=df_oat["OAT-SENSOR_OutAirTemp (°C)"],
        setpt=df_chws["CHW-SYS_ChwSetpt (°C)"],
        mdb01_kw=mdb_int["MDB-01_kWh_14648"] * 4,
        mdb02_kw=mdb_int["MDB-02_kWh_14649"] * 4,
        mdb03_kw=mdb_int["MDB-03_kWh_14650"] * 4,
    )

    corr_tbl = rc.rank_correlations(features)
    rf_rank = rc.rank_random_forest(features)

    print("\n=== BTU ΔT Feature Ranking (Mean |r|) ===")
    print(corr_tbl["MeanAbs"].round(3).head(10))
    print ("*******************")
    print ("Random Forest Feature Ranking")
    print (rf_rank.round(3).head(10))


    fig_corr = fig.plot_feature_ranking(
        corr_tbl["MeanAbs"],
        title=" BTU ΔT – Absolute Correlation Ranking"
    )
    fig.save(fig_corr, "BTUdeltaT_corr_rank")

    fig_rf = fig.plot_feature_ranking(
        rf_rank,
        title=" BTU ΔT – Random-Forest Feature Importance Ranking"
    )
    fig.save(fig_rf, "BTUdeltaT_rf_rank")


    # -----------------------------------------------------------------
    #  Find and export simultaneous-chiller intervals
    # ------------------------------------------------------------------
    def get_multi_chiller_periods(df_chiller_pct: pd.DataFrame,
                                  threshold: float = 10.0,
                                  min_chillers: int = 2) -> pd.DatetimeIndex:
        """
        Return an index of timestamps where at least `min_chillers`
        have load % > threshold.
        """
        n_on = (df_chiller_pct > threshold).sum(axis=1)
        return n_on[n_on >= min_chillers].index

    # ── call it right after the 2.12 % printout ───────────────────────
    multi_idx = get_multi_chiller_periods(df_load_pct)

    #print(f"\nIntervals with ≥2 chillers above 10 % load: {len(multi_idx)}")
    #print("First 10 examples:")
    #print(multi_idx[:10])

    dt_multi = delta_t_series(df_btu).reindex(multi_idx)

    fig.save(fig.plot_deltaT_multichiller(dt_multi), "deltaT_multi_chiller")


    print("\nLow-ΔT diagnostic suite completed → see plots folder.")

    print("\nAll analytics complete.")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()