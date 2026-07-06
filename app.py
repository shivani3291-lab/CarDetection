"""CarVision AI — single-page Streamlit dashboard entrypoint."""

import plotly.express as px
import streamlit as st
from PIL import Image

from src.serve.session import get_class_names, get_predictor, render_model_status_sidebar
from src.serve.viz import draw_boxes

st.set_page_config(
    page_title="CarVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    render_model_status_sidebar()

predictor = get_predictor()
class_names = get_class_names()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Hero ----------
st.title("CarVision AI")
st.caption("Upload a photo of a car and get its make, model, and year in seconds.")

stat1, stat2, stat3 = st.columns(3)
stat1.metric("Test accuracy", "87%")
stat2.metric("Classes recognized", f"{len(class_names)}")
stat3.metric("Training images", "8,103")

st.divider()

# ---------- How it works ----------
st.subheader("How it works")
step1, step2, step3 = st.columns(3)
with step1:
    with st.container(border=True):
        st.markdown("**1. Upload a photo**")
        st.caption("Any clear photo of a car — JPG or PNG, up to 200MB.")
with step2:
    with st.container(border=True):
        st.markdown("**2. The model looks it over**")
        st.caption("A ResNet-50 classifier trained on 196 makes/models scores your photo in under a second.")
with step3:
    with st.container(border=True):
        st.markdown("**3. Get ranked results**")
        st.caption("See the top match plus the next-best guesses, ranked by confidence.")

st.divider()

# ---------- Upload & detect ----------
st.subheader("Upload & detect")

if not predictor.loaded:
    st.warning(
        "No model found. On a new machine run:\n"
        "`python -m src.train.classifier_train` then `python -m src.mlops.export_onnx`"
    )
else:
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption(
        "Tip: a clear 3/4 front or side angle, with the whole car in frame and good "
        "lighting, gives the most reliable result."
    )
    uploaded = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running inference..."):
            results = predictor.predict(image, conf_threshold=conf)
            topk = predictor.predict_topk(image, k=5)

        if results:
            annotated = draw_boxes(image, results)
            ranked = sorted(results, key=lambda x: -x["score"])
            top = ranked[0]

            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.image(annotated, caption="Detection result", use_container_width=True)
            with col2:
                st.markdown(f"### {top['class']}")
                st.markdown(f"#### {top['score']:.1%} confidence")
                st.progress(top["score"])

                if len(topk) > 1:
                    st.caption("Other close matches")
                    fig = px.bar(
                        x=[r["score"] for r in reversed(topk)],
                        y=[r["class"] for r in reversed(topk)],
                        orientation="h",
                        labels={"x": "Confidence", "y": ""},
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        height=220,
                        margin=dict(l=0, r=10, t=10, b=10),
                        xaxis_tickformat=".0%",
                    )
                    fig.update_traces(marker_color="#7F77DD")
                    st.plotly_chart(fig, use_container_width=True)

            st.session_state.history.insert(
                0, {"class": top["class"], "score": top["score"], "thumbnail": annotated}
            )
        else:
            st.info("No detection above confidence threshold. Try lowering the threshold in the sidebar.")

st.divider()

# ---------- Detection history ----------
st.subheader("Detection history")
if not st.session_state.history:
    st.caption("No detections yet this session — upload a photo above to get started.")
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
