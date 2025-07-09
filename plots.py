import pathlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTS_DIR = pathlib.Path("plots_v2")    #plots will be stored in a new directory "plots_v2".
PLOTS_DIR.mkdir(exist_ok=True)

def save(fig, name):                       # write html + jpg
    html = PLOTS_DIR / f"{name}.html"
    fig.write_html(html)
    print(f"✓ {html.name}")

# ───────────────────────────────  plotting  ────────────────────────────────────
def fig_whole_elec(daily):
    """
    Stacked bars of MDB-level daily kWh.
    NOTE: *No* total_kWh trace is drawn.
    """
    fig = go.Figure()
    stack_cols = [c for c in daily.columns if c != 'total_kWh']
    for c in stack_cols:
        fig.add_bar(x=daily.index, y=daily[c], name=c)
    fig.update_layout(
        barmode='stack',
        yaxis_title='kWh',
        title='Daily Electricity – MDB-level (no total line)',
    )
    return fig

def fig_contribution(contrib):
    fig = px.pie(values=contrib, names=contrib.index,
                 title='Annual Energy Contribution by MDB (%)',
                 hole=.4)
    return fig

def fig_profiles(weekday, weekend):
    fig = go.Figure()
    fig.add_scatter(x=weekday.index, y=weekday.sum(axis=1), mode='lines', name='Weekday')
    fig.add_scatter(x=weekend.index, y=weekend.sum(axis=1), mode='lines', name='Weekend')
    fig.update_layout(title='Average 24-h Load Profiles', xaxis_title='Hour', yaxis_title='kWh/15 min')
    return fig

def fig_month_heat(month_hour):
    month_hour.index.names = ['Month','Hour']
    pivot = month_hour.sum(axis=1).unstack(level=0)
    fig = px.imshow(pivot, aspect='auto', origin='lower',
                    labels=dict(x='Month', y='Hour', color='kWh/15 min'),
                    title='Seasonal Profile – Hour vs Month')
    return fig

def fig_load_duration(ldc, base_kw):
    fig = px.line(ldc * 4, labels={'value':'kW', 'index':'Ranked interval'},
                  title='Load Duration Curve')
    fig.add_hline(y=base_kw, line_dash='dot', annotation_text=f'Base Load ≈ {base_kw:,.0f} kW')
    return fig

def fig_base_hist(night_sum_kw, p05_kw):
    fig = px.histogram(night_sum_kw, nbins=50,
                       labels={'value':'kW'}, title='Night-time Load Histogram')
    fig.add_vline(x=p05_kw, line_dash='dot',
                  annotation_text=f'5th-pct ≈ {p05_kw:,.0f} kW')
    return fig

def fig_daily_min(daily_min_kw):
    fig = px.line(daily_min_kw, labels={'value':'kW','index':'Date'},
                  title='Daily Minimum Load (Base-load tracker)')
    return fig

def fig_daily_peaks(daily_kw: pd.Series):
    fig = px.line(
        daily_kw,
        labels={"index": "Date", "value": "kW"},
        title="Daily Peak Demand (kW)",
    )
    fig.update_traces(line_color="firebrick")
    return fig

def fig_load_balance(share):
    """
    Returns three figures:
      1. 100%-stacked area (raw shares)
      2. 24-h rolling shares line chart
      3. 24-h rolling imbalance σ
    """

    # 1) stacked area
    fig_area = px.area(
        share,
        title="Load Share per MDB (15-min raw)",
        groupnorm="fraction",
        labels={'value':'Share', 'index':'Time'}
    )

    # 2) 24-h rolling mean
    roll = share.rolling('24h', center=True).mean()
    fig_roll = px.line(
        roll,
        title="24-hour Rolling MDB Share",
        labels={'value':'Share', 'index':'Time'}
    )

    # 3) smoothed imbalance index
    imb_24 = share.std(axis=1).rolling('24h').mean()
    fig_imb = px.line(
        imb_24,
        title="24-hour Rolling Imbalance Index (σ)",
        labels={'value':'σ of share', 'index':'Time'}
    )
    fig_imb.add_hline(y=0.20, line_dash='dot',
                      annotation_text='Preferred max σ ≈ 0.20')

    return fig_area, fig_roll, fig_imb


def fig_delta_t(delta_t: pd.Series):
    """
    Hour-by-day heat-map of mean ΔT (0–8 °C clipped for clarity).
    """
    df = delta_t.to_frame()
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour
    pivot = (
        df.pivot_table(index="hour", columns="doy", values="ΔT_°C", aggfunc="mean")
          .iloc[::-1]                                # midnight at top
    )

    fig = px.imshow(
        pivot, aspect="auto", origin="lower",
        color_continuous_scale="RdBu_r",
        labels=dict(x="Day of Year", y="Hour", color="ΔT (°C)"),
        title="Supply-Return ΔT – Hourly Mean Calendar",
    )
    fig.update_coloraxes(cmin=0, cmax=8)            # clip to useful band
    return fig

def fig_top_peaks(peaks: pd.Series):
    """
    Scatter chart of the worst 15-min spikes.
    """
    fig = px.scatter(
        peaks,
        x=peaks.index,
        y=peaks.values * 4,         # convert kWh/15-min → kW
        labels={"y": "kW", "x": "Timestamp"},
        title=f"Top {len(peaks)} 15-min Demand Spikes",
    )
    fig.update_traces(marker_size=10, marker_color="crimson")
    return fig

def fig_rolling_share(roll_share: pd.DataFrame):
    import plotly.express as px
    fig = px.line(
        roll_share,
        title="30-day Rolling Load Share per MDB",
        labels={'value':'Share','index':'Date'}
    )
    return fig

#Plant-header vs BTU ΔT
def plot_deltaT_comparison(df):
    fig = px.scatter(
        df, x="BTU_ΔT", y="Header_ΔT",
        trendline="ols", title="BTU vs Plant-Header ΔT"
    )
    fig.update_layout(xaxis_title="BTU ΔT (°C)", yaxis_title="Header ΔT (°C)")
    return fig

def plot_deltaT_anomalies(mismatch: pd.DataFrame):
    fig_mis = px.scatter(
        mismatch, x="BTU_ΔT", y="Header_ΔT",
        color="anomaly",
        color_discrete_map={False: "#1f77b4", True: "#d62728"},
        title="BTU vs Header ΔT – anomalies highlighted",
        labels={"BTU_ΔT": "BTU ΔT (°C)", "Header_ΔT": "Header ΔT (°C)"}
    )
    return fig_mis


def plot_deltaT_anomaly_timeline(
        mismatch,
        hours_before=6,
        hours_after=6,
        max_anom=200,            # <-- new limit
        pick="largest",          # "largest" | "first" | "random"
):
    """
    Timeline of ΔT mismatch anomalies, subsampled for speed.
    """
    df = mismatch.copy()
    df.index = df.index.tz_localize(None)
    df["grp"] = (df["anomaly"] != df["anomaly"].shift()).cumsum()

    # pick subset ----------------------------------------------------
    anom = df[df["anomaly"]].copy()
    if pick == "largest":
        anom = anom.reindex(anom["residual"].abs().nlargest(max_anom).index)
    elif pick == "first":
        anom = anom.head(max_anom)
    elif pick == "random":
        anom = anom.sample(max_anom, random_state=42)
    anom_grps = anom["grp"].unique()

    # shading + dots only for chosen groups
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df["Header_ΔT"], name="Header ΔT",
                    line=dict(width=1, color="#1f77b4"))
    fig.add_scatter(x=df.index, y=df["BTU_ΔT"], name="BTU ΔT",
                    line=dict(width=1, color="#ff7f0e"))
    fig.add_scatter(x=df.index, y=df["residual"], name="Residual",
                    line=dict(width=1, color="#2ca02c", dash="dot"))

    for g in anom_grps:
        start, stop = df[df["grp"] == g].index[[0, -1]]
        fig.add_vrect(x0=start, x1=stop, fillcolor="rgba(255,0,0,0.1)",
                      layer="below", line_width=0)
        pts = df.loc[df["grp"] == g]
        fig.add_scatter(x=pts.index, y=pts["Header_ΔT"], mode="markers",
                        marker=dict(color="#d62728", size=6),
                        name="Anomaly pt", showlegend=False)

    fig.update_layout(title=f"ΔT mismatch timeline (showing {len(anom_grps)} of {df['anomaly'].sum()} events)",
                      yaxis_title="°C", hovermode="x unified")
    return fig


#Flow vs ΔT (over-pumping)
def plot_flow_vs_deltaT(df):
    fig = px.scatter(
        df, x="ΔT", y="Flow_Ls",
        title="Chilled-Water Flow vs ΔT",
        labels={"ΔT":"BTU ΔT (°C)", "Flow_Ls":"Flow (L/s)"},
        trendline="lowess"
    )
    return fig

#Overlay MDB-01 base-load with FAHU status

def plot_mdb1_fahu_overlay(df):
    """
    MDB-01 kW (line) + FAHU-ON shading (transparent red area)
    """
    max_kw = df["MDB1_kW"].max()

    fig = go.Figure()

    # MDB-01 electrical demand
    fig.add_scatter(
        x=df.index, y=df["MDB1_kW"],
        name="MDB-01 kW",
        line=dict(width=1, color="blue")      # default Plotly blue
    )
    fig.add_scatter(
        x=df.index, y=df["MDB2_kW"],
        name="MDB-02 kW",
        line=dict(width=1, color="orange")  # default Plotly blue
    )
    fig.add_scatter(
        x=df.index, y=df["MDB3_kW"],
        name="MDB-03 kW",
        line=dict(width=1, color="purple")  # default Plotly blue
    )

    # FAHU-on mask  →  fill to zero with very light red, no border
    fig.add_scatter(
        x=df.index,
        y=df["FAHU_1_on"] * max_kw,                 # scale to full y-range
        name="FAHU Supply-Fan-1 ON",
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),# invisible outline
        fill="tozeroy",
        fillcolor="rgba(255, 0, 0, 0.5)"         # 18 %-opaque red
    )
    fig.add_scatter(
        x=df.index,
        y=df["FAHU_2_on"] * max_kw,  # scale to full y-range
        name="FAHU Supply-Fan-2 ON",
        mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),  # invisible outline
        fill="tozeroy",
        fillcolor="rgba(0, 180, 0, 0.5)"  # 18 %-opaque red
    )

    fig.update_layout(
        title="MDB-01 Base-load vs FAHU Operation",
        yaxis_title="kW",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


#ΔT vs outside-air temperature
def plot_deltaT_vs_OAT(df):
    fig = px.scatter(
        df, x="OAT", y="ΔT",
        trendline="lowess", title="ΔT vs Outside-Air Temperature",
        labels={"OAT":"OAT (°C)", "ΔT":"BTU ΔT (°C)"})
    return fig


#ΔT vs chiller %-load (# chillers on → marker size)
def plot_deltaT_vs_chillerload(df):
    fig = px.scatter(
        df, x="Load_pct", y="ΔT",
        size="N_on", color="N_on", size_max=18,
        title="ΔT vs Chiller Plant Load",
        labels={"Load_pct":"Total Load %", "ΔT":"BTU ΔT (°C)", "N_on":"Chillers ON"})
    return fig


def plot_all_chillers_vs_deltaT(df_chiller_pct: pd.DataFrame,
                                delta_t: pd.Series) -> px.scatter:
    """
    Single-panel scatter: each chiller’s load % vs BTU ΔT.
    Colour distinguishes chillers.
    """
    df = df_chiller_pct.join(delta_t.rename("BTU_ΔT")).dropna()

    # keep only points where the chiller is really running (>10 %)
    mask = df_chiller_pct > 10
    df_running = df.where(mask).dropna(how="all")

    long = df_running.melt(id_vars="BTU_ΔT",
                           var_name="Chiller",
                           value_name="Load_pct").dropna()

    fig = px.scatter(
        long, x="Load_pct", y="BTU_ΔT", color="Chiller",
        labels={"Load_pct": "Chiller load %", "BTU_ΔT": "BTU ΔT (°C)"},
        trendline="lowess", title="Chiller load vs BTU ΔT (all chillers)",
        height=500, width=700
    )
    return fig

#Daily ΔT profile vs HRW utilisation
def plot_daily_deltaT_vs_HRW(df):
    fig = px.scatter(
        df, x="HRW_frac", y="DailyMean_ΔT",
        trendline="ols", title="Daily Mean ΔT vs HRW On-Fraction",
        labels={"HRW_frac":"HRW utilisation fraction", "DailyMean_ΔT":"Daily ΔT (°C)"})
    return fig

#Rapid ΔT recovery events
def plot_deltaT_recovery(events):
    fig = px.line(events, x="Time", y="ΔT", facet_row="Event_ID",
                  title="ΔT Recovery During Planned Shutdowns",
                  labels={"ΔT":"ΔT (°C)"})
    fig.update_yaxes(matches=None)
    return fig

def plot_cop_line(cop_proxy):
    fig_cop_line = px.line(
        cop_proxy,
        title="Chiller-Plant COP* Proxy (15-min)",
        labels={'value': 'COP*', 'index': 'Time'}
    )
    return fig_cop_line

def plot_cop_oat(cop_proxy, df_oat):
    # COP* vs outside-air temperature
    scatter_df = pd.DataFrame({
        "COP*": cop_proxy,
        "OAT": df_oat["OAT-SENSOR_OutAirTemp (°C)"]  # df_oat already exists in script
    }).dropna()

    fig_cop_oat = px.scatter(
        scatter_df, x="OAT", y="COP*",
        trendline="lowess",
        title="COP* vs Outside-Air Temperature"
    )

    return fig_cop_oat

def plot_deltaT_multichiller(dt_multi):
    fig_multi = px.scatter(
        dt_multi, title="ΔT during intervals with ≥2 chillers ON",
        labels={'value': "BTU ΔT (°C)", 'index': "Timestamp"},
    )
    return fig_multi


# Plot feature ranking for rootcause analysis
def plot_feature_ranking(rank: pd.Series, title: str) -> px.bar:
    """
    Horizontal bar chart of feature ranking.
    """
    fig = px.bar(
        rank.iloc[::-1],  # smallest to largest top→bottom
        orientation="h",
        labels={"value": "Score", "index": "Feature"},
        title=title,
    )
    return fig


def plot_cop_timeseries(cop, roll='D'):
    """
    Smoothed COP* curve (daily or weekly mean).
    roll : 'D' or 'W'
    """
    title = f"COP* ({'Daily' if roll=='D' else 'Weekly'} mean)"
    sm = cop.resample(roll).mean()
    fig = px.line(sm, title=title, labels={'value':'COP*', 'index':'Date'})
    return fig

def plot_cop_anomaly_scatter(df):
    """
    Scatter showing 0-COP and >10-COP anomalies vs time.
    """
    fig = px.scatter(
        df, x=df.index, y='COP*',
        color='anomaly',
        color_discrete_map={True:"#d62728", False:"#1f77b4"},
        title="COP* with anomalies highlighted",
        labels={'value':'COP*'}
    )
    return fig