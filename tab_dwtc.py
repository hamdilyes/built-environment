import streamlit as st

from aux_delta_t import plot_delta_t
from aux_cop import plot_cop

from aux_dwtc import plot_live, plot_avg


def tab_dwtc(df):
    with st.expander("LIVE ∆T", expanded=True):
        plot_live(df)
    
    with st.expander("MONTH-TO-DATE AVG ∆T", expanded=True):
        plot_avg(df)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("∆T", expanded=False):
            plot_delta_t(df)
    with col2:
        with st.expander("COP", expanded=False):
            plot_cop(df)