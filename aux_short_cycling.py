import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import random

from aux_hvac_get import get_setpoint, get_supply_temperature

def plot_short_cycling(df, customer):
    kpi_dict = get_chiller_short_cycling_kpi(df)

    # ---- KPIs Display ----
    with st.expander("KPIs", expanded=True):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Avg Cycle Duration", f"{kpi_dict['Avg Cycle Duration']:.0f} min")
        with col2:
            st.metric("Min Runtime", f"{kpi_dict['Min Runtime']:.0f} min")
        with col3:
            st.metric("Cycles per Hour", f"{kpi_dict['Cycles per Hour']:.2f}")
        with col4:
            st.metric("Load Swing Rate", f"{kpi_dict['Load Swing Rate']:.2f}")
        with col5:
            st.metric("Short Cycle Count", f"{kpi_dict['Short Cycle Count']}")
        with col6:
            st.metric("Short Cycle Ratio", f"{kpi_dict['Short Cycle Ratio']:.1%}")


    # Get the supply temperature and setpoint DataFrames
    supply_temp_df = get_supply_temperature(df, customer).min(axis=1)
    setpoint_df = get_setpoint(df, customer).mean(axis=1)

    # Filter to last 30 days
    if df.index.max() is None:
        st.warning("NO DATA")
        return

    end_time = max(setpoint_df.index.max(), supply_temp_df.index.max())
    start_time = end_time - timedelta(days=7)
    
    supply_temp_df = supply_temp_df.loc[start_time:end_time]
    setpoint_df = setpoint_df.loc[start_time:end_time]

    if supply_temp_df.empty or setpoint_df.empty:
        st.warning("Insufficient data for short cycling plot.")
        return

    col1, col2 = st.columns(2)
    with col1:
        with st.expander('SUPPLY TEMPERATURE x SET-POINT'):
            st.markdown("The set-point often isn’t met, and the chiller’s frequent cycling makes the system inefficient.")
            # Create the Plotly figure
            fig = go.Figure()

            # Plot each supply temperature column
            fig.add_trace(go.Scatter(
                x=supply_temp_df.index,
                y=supply_temp_df.values,
                mode='lines',
                name=f"Supply",
                line=dict(color='green')
            ))

            # Plot each setpoint column
            fig.add_trace(go.Scatter(
                x=setpoint_df.index,
                y=setpoint_df.values,
                mode='lines',
                name=f"Setpoint",
                line=dict(color='red')
            ))

            fig.update_layout(
                title="",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                showlegend=False,
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True, key='short_cycling_'+customer)
        
        with col2:
            with st.expander('Run Time'):
                st.write('...')


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


def get_chiller_short_cycling_kpi(df):
    tag_options = {
        "CH1": "CHILLER-01_RunHr (Hours)",
        "CH2": "CHILLER-02_RunHr (Hours)",
        "CH3": "CHILLER-03_RunHr (Hours)"
    }
    col1, col2 = st.columns([1, 5])
    with col1:
        selected_label = st.selectbox("Chiller", list(tag_options.keys()))
    tag = tag_options[selected_label]

    runhr = df[tag]
    on_min = ch_runtime_diff(runhr)  # 15-min resolution assumed
    daily = daily_runtime(on_min).dropna(how='any')

    total_cycles = daily['Cycles'].sum()
    total_runtime = on_min.sum()
    avg_cycle_duration = total_runtime / total_cycles
    min_runtime = on_min[on_min > 0].min()
    cycles_per_hour = total_cycles / total_runtime * 60
    load_swing_rate = daily['Run_min'].std() / daily['Run_min'].mean()

    # Capacity tag guess (replace with actual logic or pass in)
    cap_tag = tag.replace("RunHr", "ChlrCap")  # crude pattern match
    if cap_tag in df.columns:
        cap_pct = df[cap_tag]
    else:
        cap_pct = pd.Series(index=on_min.index, data=0)  # fallback

    # Detect cycles (min_on = 5 min, min_off = 15 min for example)
    cycles = detect_cycles_minutes(on_min, cap_pct, min_on=0, min_off=16)
    short_cycle_count = cycles['ShortCycle'].sum()
    short_cycle_ratio = short_cycle_count / len(cycles) if len(cycles) else 0

    # kpi_dict = {
    #     "Min Runtime": min_runtime,
    #     "Cycles per Hour": cycles_per_hour,
    #     "Avg Cycle Duration": avg_cycle_duration,
    #     "Load Swing Rate": load_swing_rate,
    #     "Short Cycle Count": short_cycle_count,
    #     "Short Cycle Ratio": short_cycle_ratio,
    # }

    # Only use this for KPI calculations
    cycles = detect_cycles_minutes(on_min, cap_pct, min_on=0, min_off=16)

    # Compute KPIs based on `cycles` DataFrame
    total_runtime = on_min.sum()
    total_cycles = len(cycles)
    short_cycle_count = cycles['ShortCycle'].sum()
    # --- KPIs from cycles ---
    new_min_runtime = cycles["Duration_min"].min() if not cycles.empty else 0
    new_avg_cycle_duration = cycles["Duration_min"].mean() if not cycles.empty else 0
    new_cycles_per_hour = (total_cycles / total_runtime * 60) if total_runtime > 0 else 0
    new_load_swing_rate = cycles["Duration_min"].std() / cycles["Duration_min"].mean() if total_cycles > 1 else 0
    new_short_cycle_ratio = short_cycle_count / total_cycles if total_cycles > 0 else 0

    kpi_dict = {
        "Min Runtime": new_min_runtime,
        "Cycles per Hour": new_cycles_per_hour,
        "Avg Cycle Duration": new_avg_cycle_duration,
        "Load Swing Rate": new_load_swing_rate,
        "Short Cycle Count": short_cycle_count,
        "Short Cycle Ratio": new_short_cycle_ratio,
    }

    return kpi_dict