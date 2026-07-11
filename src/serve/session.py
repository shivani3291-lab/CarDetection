"""Shared Streamlit session helpers (import from pages, not app.py)."""

from __future__ import annotations

import streamlit as st

from src.serve.inference import load_class_names, load_model_metadata, load_predictor

_PREDICTOR_CACHE_VERSION = 2  # bump this whenever Predictor's interface changes -
# st.cache_resource keys on get_predictor()'s own source, not on src.serve.inference,
# so editing the Predictor class alone won't invalidate an already-cached instance
# on a redeploy that doesn't fully restart the process (missing methods -> AttributeError)


@st.cache_resource
def get_predictor(cache_version: int = _PREDICTOR_CACHE_VERSION):
    return load_predictor()


@st.cache_data
def get_class_names():
    return load_class_names()


def get_model_info() -> dict:
    predictor = get_predictor()
    metadata = load_model_metadata()
    if not predictor.loaded:
        return {
            "loaded": False,
            "message": (
                "Run: python -m src.train.classifier_train && python -m src.mlops.export_onnx"
            ),
        }
    info = {
        "loaded": True,
        "model_type": predictor.mode,
        "description": metadata.get("description", f"{predictor.mode} model"),
    }
    if "mAP@0.5" in metadata:
        info["mAP@0.5"] = metadata["mAP@0.5"]
    return info


def render_model_status_sidebar():
    info = get_model_info()
    st.markdown("### Model status")
    if info["loaded"]:
        st.markdown(":green[●] Active")
        st.caption(info["description"])
        if info.get("mAP@0.5") is not None:
            st.metric("mAP@0.5", f"{info['mAP@0.5']:.1%}")
        else:
            st.metric("Model", info["model_type"])
    else:
        st.markdown(":orange[●] No model")
        st.caption(info.get("message", "Train and export a model"))
    st.divider()
    st.caption("Stanford Cars - 196 classes")
