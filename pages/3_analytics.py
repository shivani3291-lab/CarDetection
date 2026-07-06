"""3D car visualization and session analytics."""

import streamlit as st
import streamlit.components.v1 as components

from src.serve.session import render_model_status_sidebar
from src.serve.viz import render_3d_car

with st.sidebar:
    render_model_status_sidebar()

st.title("3D car visualization")

components.html(render_3d_car(), height=320)

st.divider()
st.subheader("Session stats")

history = st.session_state.get("history", [])
col1, col2, col3 = st.columns(3)
col1.metric("Total detections", len(history))
col2.metric("Unique classes", len({h["class"] for h in history}) if history else 0)
col3.metric(
    "Avg confidence",
    f"{sum(h['score'] for h in history) / len(history):.1%}" if history else "0%",
)

if history:
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(history)
    fig = px.line(df.reset_index(), x="index", y="score", title="Confidence over time")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
