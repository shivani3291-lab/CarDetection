"""Project overview and documentation links."""

import streamlit as st

from src.serve.session import render_model_status_sidebar

with st.sidebar:
    render_model_status_sidebar()

st.title("About CarVision AI")

st.markdown(
    """
    ## Car Detection Capstone

    End-to-end deep learning system for **Stanford Cars** (196 classes, ~16K images).

    ### Stack
    - **Data:** DVC, Albumentations, Great Expectations
    - **Models:** Custom CNN, ResNet-50, EfficientNet-B3, Faster R-CNN, YOLOv8
    - **MLOps:** MLflow, GitHub Actions, Evidently AI
    - **Serving:** ONNX Runtime in Streamlit

    ### Fresh machine setup
    1. `python scripts/setup_data.py --zip "Car Images.zip"`
    2. `python -m src.data.preprocess`
    3. `python -m src.data.validate`
    4. `python -m src.train.classifier_train --config configs/classifier_resnet50.yaml`
    5. `python -m src.mlops.export_onnx`
    6. `streamlit run app.py`

    Or run the full pipeline: `powershell scripts/run_pipeline.ps1`

  See `README.md` and `MODEL_CARD.md` in the repository for full documentation.
    """
)

st.divider()
st.caption("Built as an AI/ML Engineer + MLOps capstone project.")
