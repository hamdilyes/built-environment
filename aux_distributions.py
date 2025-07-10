import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def plot_distributions(df, title_suffix=""):
    """
    Create distribution plots for selected numerical columns in the dataframe.
    Includes histograms, box plots, and summary statistics.
    By default, nothing is shown until the user selects features.
    """
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_cols:
        st.warning("No numerical columns found for plotting.")
        return

    # User selection
    selected_cols = st.multiselect("Select Features", options=numerical_cols)

    if not selected_cols:
        return

    colors = px.colors.qualitative.Set1

    for i, col in enumerate(selected_cols):
        st.subheader(f"{col} - Distribution{title_suffix}")

        # Two-column layout
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                df,
                x=col,
                nbins=30,
                title=f"{col} - Histogram",
                color_discrete_sequence=[colors[i % len(colors)]]
            )
            fig_hist.update_layout(height=400, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = px.box(
                df,
                y=col,
                title=f"{col} - Box Plot",
                color_discrete_sequence=[colors[i % len(colors)]]
            )
            fig_box.update_layout(height=400, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
            st.plotly_chart(fig_box, use_container_width=True)

        # Summary statistics
        st.subheader(f"{col} - Summary Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Min', 'Max', '% Missing Values', '% Duplicate Values'],
            'Value': [
                df[col].min(),
                df[col].max(),
                round((df[col].isnull().sum() / len(df)) * 100, 2),
                round((df[col].duplicated().sum() / len(df)) * 100, 2)
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("---")  # Visual separator