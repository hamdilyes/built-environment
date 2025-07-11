import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import plotly.express as px
import plotly.graph_objects as go


##### AUX #####

def total_energy_consumption(df):
    daily = df.resample('D').sum()
    weekly = df.resample('W').sum()
    monthly = df.resample('ME').sum()
    annual = df.resample('YE').sum()
    contribution = df.sum() / df.sum().sum() * 100
    trend = daily.sum(axis=1)
    return daily, weekly, monthly, annual, contribution, trend


def load_profile_analysis(df):
    tmp = df.copy()
    tmp['Hour'] = tmp.index.hour
    tmp['Weekday'] = tmp.index.weekday
    weekday = tmp[tmp['Weekday'] < 5].groupby('Hour').mean()
    weekend = tmp[tmp['Weekday'] >= 5].groupby('Hour').mean()
    monthly = tmp.groupby([tmp.index.month, tmp.index.hour]).mean()
    return weekday, weekend, monthly


def peak_demand_analysis(df):
    peak15 = df.max()
    daily_pk = df.resample('D').max()
    ldc = df.sum(axis=1).sort_values(ascending=False).reset_index(drop=True)
    return peak15, daily_pk, ldc


def base_load_estimation(df, night_start='22:00', night_end='05:00', pctl=0.05):
    night = df.between_time(night_start, night_end)
    min_n = night.min()
    p05 = df.sum(axis=1).quantile(pctl)
    return min_n, p05


def load_balancing_analysis(df):
    tot = df.sum()
    ratio = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
    imb = ratio.std(axis=1).fillna(0)
    return tot, ratio, imb


def top_peak_intervals(df, top_n=20):
    total = df.sum(axis=1)
    return total.nlargest(top_n).sort_values(ascending=False)


def daily_peak_demand(df):
    return (df.sum(axis=1) * 4).resample("D").max()


def overloaded_mdb(df, threshold_share=0.65):
    share = df.div(df.sum(axis=1), axis=0)
    return share.gt(threshold_share)


def rolling_mdb_share(df, window="30D"):
    share = df.div(df.sum(axis=1), axis=0)
    return share.rolling(window, center=True, min_periods=10).mean()


def load_factor(kwh_series, interval_minutes=15, by=None):
    if isinstance(kwh_series, pd.DataFrame):
        kwh_series = kwh_series.iloc[:, 0]

    kwh_series = kwh_series.dropna()
    kw_series = kwh_series * (60 / interval_minutes)

    if by is None:
        total_kwh = kwh_series.sum()
        hours = len(kwh_series) * interval_minutes / 60
        avg_kw = total_kwh / hours
        peak_kw = kw_series.max()
        return avg_kw / peak_kw

    grouped = (
        pd.DataFrame({"kWh": kwh_series, "kW": kw_series})
        .groupby(pd.Grouper(freq=by))
        .agg(total_kWh=("kWh", "sum"),
             peak_kW=("kW", "max"),
             intervals=("kWh", "count"))
    )
    grouped["avg_kW"] = grouped["total_kWh"] * (60 / interval_minutes) / grouped["intervals"]
    grouped["load_factor"] = grouped["avg_kW"] / grouped["peak_kW"]
    return grouped["load_factor"]


def print_top_peaks(peaks: pd.Series):
    print("\nTop-{} 15-min demand spikes (kW):".format(len(peaks)))
    for ts, kwh in peaks.items():
        print(f"  {ts:%Y-%m-%d %H:%M}  →  {kwh*4:6.0f} kW")


def print_overload_stats(mask: pd.DataFrame):
    pct = mask.mean() * 100
    print("\nOverload incidence (share >65 % of total load):")
    for col, p in pct.items():
        print(f"  {col:20s}: {p:5.1f} % of intervals")


def detect_trend(series: pd.Series):
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


##### PLOTS #####


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
    fig = px.line(
        roll_share,
        title="30-day Rolling Load Share per MDB",
        labels={'value':'Share','index':'Date'}
    )
    return fig