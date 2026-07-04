"""CarVision AI — multi-page Streamlit dashboard entrypoint."""

import streamlit as st
import streamlit.components.v1 as components

from src.serve.session import render_model_status_sidebar
from src.serve.viz import render_3d_car

st.set_page_config(
    page_title="CarVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    render_model_status_sidebar()

st.title("CarVision AI")
st.caption("AI-powered car detection and recognition")

col1, col2 = st.columns([1.4, 1])
with col1:
    st.subheader("Quick start")
    st.markdown(
        """
        Use the sidebar to navigate:
        - **Live detection** — capture from webcam
        - **Upload & detect** — analyse uploaded images
        - **Analytics** — session detection history
        - **About** — project overview
        """
    )
with col2:
    st.subheader("3D preview")
    components.html(render_3d_car(25), height=180)
