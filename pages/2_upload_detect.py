"""Upload and detect cars from static images."""

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.serve.session import get_class_names, get_predictor, render_model_status_sidebar
from src.serve.viz import draw_boxes, render_3d_car

st.title("Upload and detect")

with st.sidebar:
    render_model_status_sidebar()

predictor = get_predictor()
class_names = get_class_names()

if not predictor.loaded:
    st.warning(
        "No model found. On a new machine run:\n"
        "`python -m src.train.classifier_train` then `python -m src.mlops.export_onnx`"
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

uploaded = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Running inference..."):
        results = predictor.predict(image, conf_threshold=conf)

    if results:
        annotated = draw_boxes(image, results)
        ranked = sorted(results, key=lambda x: -x["score"])
        col1, col2, col3 = st.columns([1.3, 1, 1])
        with col1:
            st.image(annotated, caption="Detection result", use_container_width=True)
        with col2:
            for r in ranked:
                st.metric(r["class"], f"{r['score']:.1%}")
                st.progress(r["score"])
        with col3:
            top = ranked[0]
            components.html(render_3d_car(label=top["class"], score=top["score"]), height=220)

        st.session_state.history.insert(
            0,
            {"class": results[0]["class"], "score": results[0]["score"], "thumbnail": annotated},
        )
    else:
        st.info("No detection above confidence threshold.")

st.subheader("Detection results")
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
