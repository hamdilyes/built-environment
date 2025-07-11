import streamlit as st
import pandas as pd

from aux_mdb_1 import (
    total_energy_consumption,
    load_profile_analysis,
    peak_demand_analysis,
    base_load_estimation,
    load_balancing_analysis,
    top_peak_intervals,
    daily_peak_demand,
    overloaded_mdb,
    rolling_mdb_share,
    load_factor,
    print_top_peaks,
    print_overload_stats,
    print_trend_results,

    fig_whole_elec,
    fig_contribution,
    fig_profiles,
    fig_month_heat,
    fig_load_duration,
    fig_base_hist,
    fig_daily_min,
    fig_load_balance,
    fig_top_peaks,
    fig_daily_peaks,
    fig_rolling_share,
)


mdb_cols = [
        "MDB-01_ActEnergyDlvd_14648",
        "MDB-02_ActEnergyDlvd_14649",
        "MDB-03_ActEnergyDlvd_14650"
    ]


def tab_mdb_1(df):
    # Filter to only those columns that exist in df
    available_mdb_cols = [col for col in mdb_cols if col in df.columns]
    if not available_mdb_cols:
        st.warning("No MDB data.")
        return
    
    if not st.session_state.get("run_mdb", False):
        st.session_state.run_mdb = False
    
    if st.button("RUN / HIDE", key="button_run_mdb"):
        st.session_state.run_mdb = not st.session_state.run_mdb

    if st.session_state.get("run_mdb", False):
        tab_mdb_1_run(df)


def tab_mdb_1_run(df: pd.DataFrame):
    """
    Render MDB electricity analytics plots in a Streamlit tab.
    Assumes df is preloaded and includes a datetime index.
    """
    available_mdb_cols = [col for col in mdb_cols if col in df.columns]
    df_mdb = df[available_mdb_cols].copy()

    # consumption
    df_mdb = df_mdb.diff()
    df_mdb[df_mdb < 0] = None

    df_mdb["total_kWh"] = df_mdb.sum(axis=1)

    # 1. Total Energy Consumption
    daily, weekly, monthly, annual, contrib, trend = total_energy_consumption(df_mdb.drop(columns="total_kWh"))
    st.plotly_chart(fig_whole_elec(daily), use_container_width=True)
    st.plotly_chart(fig_contribution(contrib), use_container_width=True)

    # 2. Load Profile Analysis
    weekday, weekend, month_hour = load_profile_analysis(df_mdb[['total_kWh']])
    st.plotly_chart(fig_profiles(weekday, weekend), use_container_width=True)
    st.plotly_chart(fig_month_heat(month_hour), use_container_width=True)

    # 3. Peak Demand and Load Duration Curve
    peak_15min, daily_peak, ldc = peak_demand_analysis(df_mdb[['total_kWh']])
    base_kw = base_load_estimation(df_mdb[['total_kWh']])[1] * 4
    st.plotly_chart(fig_load_duration(ldc, base_kw), use_container_width=True)

    # 4. Base Load
    night_sum_kw = df_mdb.between_time('22:00', '05:00').sum(axis=1) * 4
    daily_min_kw = df_mdb.resample('D').min()['total_kWh'] * 4
    st.plotly_chart(fig_base_hist(night_sum_kw, base_kw), use_container_width=True)
    st.plotly_chart(fig_daily_min(daily_min_kw), use_container_width=True)

    # 5. Load Balancing
    tot, ratio, imb = load_balancing_analysis(df_mdb.drop(columns="total_kWh"))
    fig1, fig2, fig3 = fig_load_balance(ratio)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    # 6. Peak Spikes
    peaks = top_peak_intervals(df_mdb.drop(columns="total_kWh"))
    print_top_peaks(peaks)
    st.plotly_chart(fig_top_peaks(peaks), use_container_width=True)

    # 7. Daily Peak Demand
    daily_kw = daily_peak_demand(df_mdb.drop(columns="total_kWh"))
    st.plotly_chart(fig_daily_peaks(daily_kw), use_container_width=True)

    # 8. Overload Stats
    over_mask = overloaded_mdb(df_mdb.drop(columns="total_kWh"))
    print_overload_stats(over_mask)

    # 9. Load Factor
    lf = load_factor(df_mdb['total_kWh'])
    st.metric("Load Factor", f"{lf:.2%}")

    # 10. Trend Analysis
    roll_share = rolling_mdb_share(df_mdb.drop(columns="total_kWh"))
    print_trend_results(roll_share)
    st.plotly_chart(fig_rolling_share(roll_share), use_container_width=True)