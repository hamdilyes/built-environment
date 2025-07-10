import streamlit as st

from aux_distributions import plot_distributions
from aux_relationships import plot_relationships
from aux_parallel import plot_parallel


def tab_interactive(filtered_df):
    tab_functions = {
        "Line / Scatter Plot": lambda: plot_relationships(filtered_df),
        "Distribution Plot": lambda: plot_distributions(filtered_df),
        "Parallel Plot": lambda: plot_parallel(filtered_df),
    }

    tabs = st.tabs(tab_functions.keys())

    for tab, (label, func) in zip(tabs, tab_functions.items()):
        with tab:
            func()