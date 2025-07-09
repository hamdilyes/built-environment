import streamlit as st
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_time_series(df, title_suffix=""):
    """
    Create individual time series plots for all numerical columns in the dataframe
    One plot per column, displayed vertically
    """
    # Get numerical columns only, excluding timestamp-like columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
   
    if not numerical_cols:
        st.warning("No numerical columns found for plotting.")
        return
   
    # Color palette
    colors = px.colors.qualitative.Set1
   
    # Create individual plots for each column
    for i, col in enumerate(numerical_cols):
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2)
            )
        )
        
        # Update layout for bigger plots
        fig.update_layout(
            height=500,  # Bigger height for each plot
            title=f"{col}{title_suffix}",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title=col,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)