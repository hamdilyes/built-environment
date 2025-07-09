import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def plot_parallel(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns found for parallel coordinates plot.")
        return

    with st.expander("Parallel Coordinates Plot Settings", expanded=True):
        st.markdown("Select up to 6 features in the order you'd like them to appear.")
        selected_features = []
        for i in range(6):
            feature = st.selectbox(f"Feature {i + 1}", options=["None"] + numeric_columns, index=0, key=f"parallel_feature_{i}")
            if feature != "None":
                selected_features.append(feature)

    if len(selected_features) < 2:
        st.info("Please select at least two features for plotting.")
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