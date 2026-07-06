"""CarVision AI — single-page Streamlit dashboard entrypoint."""

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.serve.session import get_class_names, get_predictor, render_model_status_sidebar
from src.serve.viz import draw_boxes, render_3d_car

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

# ---------- Row 1: sample 3D car ----------
st.subheader("3D preview")
st.caption("A live 3D render — rotates continuously. Upload a photo below to see your own result here instead.")
components.html(render_3d_car(), height=320)

st.divider()

# ---------- Row 2: upload & detect ----------
st.subheader("Upload & detect")

predictor = get_predictor()
class_names = get_class_names()

if "history" not in st.session_state:
    st.session_state.history = []

if not predictor.loaded:
    st.warning(
        "No model found. On a new machine run:\n"
        "`python -m src.train.classifier_train` then `python -m src.mlops.export_onnx`"
    )
else:
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    uploaded = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running inference..."):
            results = predictor.predict(image, conf_threshold=conf)

        if results:
            annotated = draw_boxes(image, results)
            ranked = sorted(results, key=lambda x: -x["score"])
            top = ranked[0]

            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.image(annotated, caption="Detection result", use_container_width=True)
            with col2:
                for r in ranked:
                    st.metric(r["class"], f"{r['score']:.1%}")
                    st.progress(r["score"])

            # ---------- Row 3: uploaded image's 3D view ----------
            st.subheader("Detected car — 3D view")
            components.html(render_3d_car(label=top["class"], score=top["score"]), height=320)

            st.session_state.history.insert(
                0, {"class": top["class"], "score": top["score"], "thumbnail": annotated}
            )
        else:
            st.info("No detection above confidence threshold.")

st.divider()

# ---------- Row 4: rest ----------
st.subheader("Detection history")
if not st.session_state.history:
    st.caption("No detections yet this session.")
for entry in st.session_state.history[:10]:
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        st.image(entry["thumbnail"], width=60)
    with c2:
        st.write(entry["class"])
    with c3:
        st.write(f"{entry['score']:.1%}")

st.divider()
st.markdown(
    """
    Use the sidebar to navigate to **Live detection** (webcam capture),
    **Analytics** (session history), or **About** (project overview).
    """
)
