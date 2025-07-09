import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def plot_relationships(df):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns found for plotting.")
        return

    with st.expander("Settings", expanded=True):
        x_axis = st.selectbox("X", options=["Timestamp"] + numeric_columns, index=0)
        y_axis_1 = st.selectbox("Y1", options=numeric_columns, index=0)
        y_axis_2 = st.selectbox("Y2 (optional)", options=["None"] + numeric_columns, index=0)

    fig = go.Figure()

    x_data = df.index if x_axis == "Timestamp" else df[x_axis]
    plot_mode = "lines+markers" if x_axis == "Timestamp" else "markers"

    fig.add_trace(go.Scatter(
        x=x_data,
        y=df[y_axis_1],
        mode=plot_mode,
        name=y_axis_1,
        yaxis="y1"
    ))

    if y_axis_2 != "None" and y_axis_2 != y_axis_1:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df[y_axis_2],
            mode=plot_mode,
            name=y_axis_2,
            yaxis="y2"
        ))

        fig.update_layout(
            yaxis=dict(title=y_axis_1),
            yaxis2=dict(
                title=y_axis_2,
                overlaying="y",
                side="right"
            ),
        )
    else:
        fig.update_layout(yaxis=dict(title=y_axis_1))

    fig.update_layout(
        xaxis_title=x_axis,
        title="",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)