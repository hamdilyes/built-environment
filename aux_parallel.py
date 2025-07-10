import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def plot_parallel(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns found.")
        return

    with st.expander("Select Features", expanded=True):
        selected_features = []

        # Create 2 rows of 3 columns each (total 6 selectors)
        for row in range(2):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                feature = cols[col].selectbox(
                    f"Feature {idx + 1}", 
                    options=["None"] + numeric_columns, 
                    index=0, 
                    key=f"parallel_feature_{idx}"
                )
                if feature != "None":
                    selected_features.append(feature)

    if len(selected_features) < 2:
        return

    fig = go.Figure(data=go.Parcoords(
        line=dict(colorscale='Viridis', showscale=False),
        dimensions=[dict(label=col, values=df[col]) for col in selected_features]
    ))

    fig.update_layout(
        title="",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)