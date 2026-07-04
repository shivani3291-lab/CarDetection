"""Live single-frame detection via webcam."""

import streamlit as st
from PIL import Image

from src.serve.session import get_class_names, get_predictor, render_model_status_sidebar
from src.serve.viz import draw_boxes

st.title("Live detection")

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

conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
camera_image = st.camera_input("Point your camera at a car")

if camera_image:
    image = Image.open(camera_image).convert("RGB")
    with st.spinner("Running inference..."):
        results = predictor.predict(image, conf_threshold=conf)
    if results:
        annotated = draw_boxes(image, results)
        st.image(annotated, caption="Detection result", use_container_width=True)
        for r in results:
            st.metric(r["class"], f"{r['score']:.1%}")
    else:
        st.info("No detection above confidence threshold.")
