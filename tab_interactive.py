import streamlit as st

from aux_distributions import plot_distributions
from aux_relationships import plot_relationships
from aux_parallel import plot_parallel
from tab_hvac_1 import tab_hvac_1
from tab_mdb_1 import tab_mdb_1


def tab_interactive(df):
    tab_functions = {
        "Line / Scatter Plot": lambda: plot_relationships(df),
        "Distribution Plot": lambda: plot_distributions(df),
        "Parallel Plot": lambda: plot_parallel(df),
        "MDB Dump": lambda: tab_mdb_1(df),
        "HVAC Dump": lambda: tab_hvac_1(df),
    }

    tabs = st.tabs(tab_functions.keys())

    for tab, (label, func) in zip(tabs, tab_functions.items()):
        with tab:
            func()