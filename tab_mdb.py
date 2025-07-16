import streamlit as st

previous_month_consumption = 15003
month_to_date_consumption = 6489
end_of_month_forecast = 14286

def tab_mdb(df):
    if not st.toggle("COST", key=1):
        with st.expander("CONSUMPTION", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Previous Month", f"{previous_month_consumption:,} kWh")

            col2.metric(
                "Month-to-Date",
                f"{month_to_date_consumption:,} kWh",
            )

            delta_forecast = 100 * (end_of_month_forecast - previous_month_consumption) / previous_month_consumption
            col3.metric(
                "End of Month Forecast",
                f"{end_of_month_forecast:,} kWh",
                delta=f"{delta_forecast:.1f}%",
                delta_color="normal",
            )
    
    else:
        with st.expander("COST", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Previous Month", f"{previous_month_consumption:,} AED")

            col2.metric(
                "Month-to-Date",
                f"{month_to_date_consumption:,} AED",
            )

            delta_forecast = 100 * (end_of_month_forecast - previous_month_consumption) / previous_month_consumption
            col3.metric(
                "End of Month Forecast",
                f"{end_of_month_forecast:,} AED",
                delta=f"{delta_forecast:.1f}%",
                delta_color="normal",
            )

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("INSIGHTS", expanded=False):
            st.write("NO DATA")
    with col2:
        with st.expander("ANOMALIES", expanded=False):
            st.write("NO DATA")