import pandas as pd
import streamlit as st

from chiller_kpi_utils import (
    diff_cum,
    compute_kpi_suite,
)

def get_chiller_kpi(df, chiller_name):

    # --- Required Columns ---
    i = chiller_name[2]
    required_columns = [
        "BTU_load",
        f'Value_Chiller_{i}_kWh_2024',
        f'CHILLER-0{i}_ChlrCap (%)',
        f'CHILLER-0{i}_RunHr (Hours)',
        "MDB-01_ActEnergyDlvd_14648",
        "MDB-02_ActEnergyDlvd_14649",
        "MDB-03_ActEnergyDlvd_14650",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.warning("NO DATA")
        return {}

    dff_btu = df[["BTU_load"]]
    df_btu_kwh = dff_btu.rename(columns={"BTU_load": "kWh"})
    KW_TO_TON = 3.517
    df_btu_kwh["Cooling_kW"] = df_btu_kwh["kWh"] * 4
    df_btu_kwh["Cooling_Tons"] = df_btu_kwh["Cooling_kW"] / KW_TO_TON

    cols = [
        "MDB-01_ActEnergyDlvd_14648",
        "MDB-02_ActEnergyDlvd_14649",
        "MDB-03_ActEnergyDlvd_14650"
    ]
    mdb_cum = df[cols]
    mdb_int = pd.concat({c.replace('ActEnergyDlvd_', 'kWh_'): diff_cum(mdb_cum[c]) for c in cols}, axis=1)
    df_mdb_kw = mdb_int.sum(axis=1)

    cap_cols = {
        'CH1': 'CHILLER-01_ChlrCap (%)',
        'CH2': 'CHILLER-02_ChlrCap (%)',
        'CH3': 'CHILLER-03_ChlrCap (%)',
    }
    runhr_cols = {
        'CH1': 'CHILLER-01_RunHr (Hours)',
        'CH2': 'CHILLER-02_RunHr (Hours)',
        'CH3': 'CHILLER-03_RunHr (Hours)',
    }

    df_load_pct = (
        df[[cap_cols[chiller_name]]]
        .apply(pd.to_numeric, errors="coerce")
        .clip(lower=0, upper=100)
        .rename(columns={cap_cols[chiller_name]: f"{chiller_name}_LoadPct"})
    )

    i = chiller_name[2]
    df_kW = df[[f'Value_Chiller_{i}_kWh_2024']]
    df_kW *= 4
    df_ch = df_kW.rename(columns={f'Value_Chiller_{i}_kWh_2024': f"{chiller_name}_kW"})
    df_ch[f"{chiller_name}_ton"] = df_ch[f"{chiller_name}_kW"] / KW_TO_TON

    df_runhr = df[[runhr_cols[chiller_name]]].rename(
        columns={runhr_cols[chiller_name]: f"{chiller_name}_RunHr"}
    )

    result = {}

    # --- Full Timeline KPI ---
    kpi = compute_kpi_suite(
        cooling_kw=df_btu_kwh["Cooling_kW"],
        cooling_tons=df_btu_kwh["Cooling_Tons"],
        input_kw=df_ch[f"{chiller_name}_kW"],
        input_ton=df_ch[f"{chiller_name}_ton"],
        runhr_series=df_runhr[f"{chiller_name}_RunHr"],
        cap_pct_series=df_load_pct[f"{chiller_name}_LoadPct"],
        mdb_kw=df_mdb_kw,
    )

    kpi["kW_per_ton"] = kpi["kW_per_ton"][kpi["kW_per_ton"] > 0]
    kpi["CapacityUtil_%"] = kpi["CapacityUtil_%"][kpi["CapacityUtil_%"] > 0]

    result.update({
        "avg_COP_full": kpi["COP"].mean(),
        "avg_EER_full": kpi["EER"].mean(),
        "avg_kW_per_ton_full": kpi["kW_per_ton"].mean(),
        "avg_CapacityUtil_%_full": kpi["CapacityUtil_%"].mean(),
        "avg_daily_hours_full": kpi["OperatingHours"].mean(),
        "avg_hours_by_band_full": kpi["OperatingHoursByBand"].filter(like="Hrs_").mean().to_dict(),
        "CyclesTable": kpi["CyclesTable"],
    })

    # # --- KPI when chiller is ON ---
    # last_ts = df_runhr.index.max()
    # runtime_diff = ch_runtime_diff(df_runhr[f"{chiller_name}_RunHr"])
    # is_on_now = (
    #     runtime_diff.loc[last_ts] > 0
    #     or df_load_pct.loc[last_ts, f"{chiller_name}_LoadPct"] > 0
    # )

    # if is_on_now:
    #     on_mask = runtime_diff > 0

    #     mask_btu = on_mask.reindex(df_btu_kwh.index, fill_value=False)
    #     mask_ch = on_mask.reindex(df_ch.index, fill_value=False)
    #     mask_pct = on_mask.reindex(df_load_pct.index, fill_value=False)

    #     kpi_on = compute_kpi_suite(
    #         cooling_kw=df_btu_kwh.loc[mask_btu, "Cooling_kW"],
    #         cooling_tons=df_btu_kwh.loc[mask_btu, "Cooling_Tons"],
    #         input_kw=df_ch.loc[mask_ch, f"{chiller_name}_kW"],
    #         runhr_series=df_runhr.loc[on_mask, f"{chiller_name}_RunHr"],
    #         cap_pct_series=df_load_pct.loc[mask_pct, f"{chiller_name}_LoadPct"],
    #         mdb_kw=df_mdb_kw,
    #     )

    #     result.update({
    #         "avg_COP_on": kpi_on["COP"].mean(),
    #         "avg_EER_on": kpi_on["EER"].mean(),
    #         "avg_kW_per_ton_on": kpi_on["kW_per_ton"].mean(),
    #         "avg_CapacityUtil_%_on": kpi_on["CapacityUtil_%"].mean(),
    #         "avg_daily_hours_on": kpi_on["OperatingHours"].mean(),
    #         "avg_hours_by_band_on": kpi_on["OperatingHoursByBand"].filter(like="Hrs_").mean().to_dict()
    #     })

    # else:
    #     result["chiller_status"] = "OFF at last timestamp"
    return result