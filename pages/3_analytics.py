"""3D car visualization and session analytics."""

import streamlit as st
import streamlit.components.v1 as components

from src.serve.feedback import feedback_summary, out_of_taxonomy_requests
from src.serve.session import get_class_names, render_model_status_sidebar
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

st.divider()
st.subheader("User feedback collected")
st.caption(
    "Thumbs up/down from the detect page, saved as labeled samples for future "
    "retraining or calibration - not used to alter any displayed confidence score."
)
fb = feedback_summary()
fcol1, fcol2, fcol3 = st.columns(3)
fcol1.metric("Total feedback", fb["total"])
fcol2.metric("Confirmed correct", fb["correct"])
fcol3.metric("Reported incorrect", fb["incorrect"])

gaps = out_of_taxonomy_requests(get_class_names())
if gaps:
    st.caption("Cars users have flagged that aren't one of the 196 known classes yet:")
    st.dataframe(
        {"Requested car": [g[0] for g in gaps], "Times reported": [g[1] for g in gaps]},
        hide_index=True,
        use_container_width=True,
    )
