"""CarVision AI — single-page Streamlit dashboard entrypoint."""

import plotly.express as px
import streamlit as st
from PIL import Image

from src.serve.feedback import log_feedback
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
        st.caption(
            "A ResNet-50 classifier trained on 196 makes/models scores your photo "
            "in under a second."
        )
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
    conf = st.sidebar.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.5,
        0.05,
        help=(
            "With 196 similar classes, even a correct best guess can score low on "
            "raw confidence. This only flags results below the threshold for a "
            "second look — it no longer hides them."
        ),
    )
    st.caption(
        "Tip: a clear 3/4 front or side angle, with the whole car in frame and good "
        "lighting, gives the most reliable result."
    )
    uploaded = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running inference..."):
            if predictor.mode == "onnx":
                # Classifier always has a best guess among 196 classes - show it
                # regardless of the threshold, and just flag when it's a weak one.
                topk = predictor.predict_topk(image, k=5)
                top = topk[0] if topk else None
                low_confidence = bool(top) and top["score"] < conf
                bbox_result = (
                    [
                        {
                            "class": top["class"],
                            "score": top["score"],
                            "bbox": [0, 0, image.width, image.height],
                        }
                    ]
                    if top
                    else []
                )
            else:
                # A real object detector: "nothing above threshold" is a valid outcome.
                bbox_result = predictor.predict(image, conf_threshold=conf)
                topk = []
                top = bbox_result[0] if bbox_result else None
                low_confidence = False

        if top:
            annotated = draw_boxes(image, bbox_result)

            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.image(annotated, caption="Detection result", use_container_width=True)
            with col2:
                st.markdown(f"### {top['class']}")
                st.markdown(f"#### {top['score']:.1%} confidence")
                st.progress(top["score"])
                if low_confidence:
                    st.warning(
                        f"Below your {conf:.0%} threshold — treat this as a best guess "
                        "rather than a certain match."
                    )

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

            fb_id = getattr(uploaded, "file_id", f"{uploaded.name}-{uploaded.size}")
            history_logged_key = f"history_logged_{fb_id}"
            if not st.session_state.get(history_logged_key):
                st.session_state.history.insert(
                    0, {"class": top["class"], "score": top["score"], "thumbnail": annotated}
                )
                st.session_state[history_logged_key] = True

            # ---------- Feedback ----------
            verdict_key = f"feedback_verdict_{fb_id}"
            logged_key = f"feedback_logged_{fb_id}"

            st.divider()
            verdict = st.session_state.get(verdict_key)
            if verdict is None:
                st.markdown("**Was this detection correct?**")
                yes_col, no_col, _ = st.columns([1, 1, 4])
                if yes_col.button("👍 Correct", key=f"{fb_id}_yes"):
                    log_feedback(image, top["class"], top["score"], True, conf, topk=topk)
                    st.session_state[verdict_key] = "correct"
                    st.session_state[logged_key] = True
                    st.rerun()
                if no_col.button("👎 Incorrect", key=f"{fb_id}_no"):
                    st.session_state[verdict_key] = "incorrect"
                    st.rerun()
            elif verdict == "correct":
                st.success("Thanks - logged as confirmed feedback for future model calibration.")
            elif verdict == "incorrect":
                if st.session_state.get(logged_key):
                    st.info("Thanks - logged as incorrect feedback for future model calibration.")
                else:
                    corrected = st.selectbox(
                        "What's the correct make/model? (optional)",
                        ["Not sure", "Other (type below)", *class_names],
                        key=f"{fb_id}_correction",
                    )
                    other_text = ""
                    if corrected == "Other (type below)":
                        other_text = st.text_input(
                            "Type the correct make/model/year",
                            key=f"{fb_id}_correction_other",
                            help="Not in the list above? This car may not be one of the "
                            "196 classes the model knows - typing it here still helps us "
                            "track how often that happens.",
                        )
                    if st.button("Submit correction", key=f"{fb_id}_submit"):
                        if corrected == "Not sure":
                            final_corrected = None
                        elif corrected == "Other (type below)":
                            final_corrected = other_text.strip() or None
                        else:
                            final_corrected = corrected
                        log_feedback(
                            image,
                            top["class"],
                            top["score"],
                            False,
                            conf,
                            corrected_class=final_corrected,
                            topk=topk,
                        )
                        st.session_state[logged_key] = True
                        st.rerun()
        else:
            st.info("No car detected in this image. Try a clearer photo with the car in frame.")

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
