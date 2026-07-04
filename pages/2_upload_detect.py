"""Upload and detect cars from static images."""

import streamlit as st
from PIL import Image

from src.serve.session import get_class_names, get_predictor, render_model_status_sidebar
from src.serve.viz import draw_boxes

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
        col1, col2 = st.columns(2)
        with col1:
            st.image(annotated, caption="Detection result", use_container_width=True)
        with col2:
            for r in sorted(results, key=lambda x: -x["score"]):
                st.metric(r["class"], f"{r['score']:.1%}")
                st.progress(r["score"])

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
