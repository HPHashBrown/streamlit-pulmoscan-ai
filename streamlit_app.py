"""
streamlit_app.py

Main page of the PulmoScan AI Streamlit app: upload/example selection,
analysis (prediction + Grad-CAM), results display, and PDF download.

All model/inference logic lives in model.py and predict.py (shared,
unchanged from the original Flask version). PDF generation lives in
report.py. This file only handles the UI and page flow.
"""

import base64
import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from model import load_model
from predict import (
    allowed_file,
    load_image,
    predict_with_gradcam,
    InvalidImageError,
)
from report import build_pdf_report
from dicom_utils import is_dicom_file, load_dicom, InvalidDicomError
from components_ui import render_zoomable_image, render_read_aloud_button

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

EXPLANATIONS = {
    "Normal": (
        "The model did not detect image features resembling a suspicious "
        "mass or nodule in this X-ray. This does NOT guarantee the "
        "absence of disease \u2014 the AI only recognizes patterns similar "
        "to what it saw during training, and it can miss findings a "
        "radiologist would catch."
    ),
    "Suspicious": (
        "The model detected features similar to masses or nodules seen in "
        "its training data. This is not a diagnosis. Please consult a "
        "qualified healthcare professional for a proper evaluation of "
        "this image."
    ),
}

EXAMPLE_FILES = [
    "example-normal-1.png",
    "example-normal-2.png",
    "example-normal-3.png",
    "example-suspicious-1.png",
    "example-suspicious-2.png",
    "example-suspicious-3.png",
]

st.set_page_config(
    page_title="PulmoScan AI",
    page_icon="\U0001FAC1",
    layout="centered",
)


# -----------------------
# Model loading (cached so it only happens once per server process,
# not on every rerun/interaction, which is critical in Streamlit)
# -----------------------
@st.cache_resource(show_spinner="Loading the model\u2026")
def get_model():
    return load_model()


model, device = None, None
model_load_error = None
try:
    model, device = get_model()
except Exception as exc:  # noqa: BLE001
    model_load_error = str(exc)


# -----------------------
# Shared styling
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .disclaimer-banner {
        background: color-mix(in srgb, var(--primary-color) 8%, var(--secondary-background-color));
        border: 1px solid color-mix(in srgb, var(--primary-color) 25%, transparent);
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 13.5px;
        line-height: 1.55;
        margin-bottom: 18px;
        color: var(--text-color);
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--primary-color);
        font-weight: 500;
        margin-bottom: 6px;
    }

    .example-caption {
        text-align: center;
        font-size: 11.5px;
        color: var(--text-color);
        opacity: 0.6;
        margin-top: 2px;
    }

    .result-card {
        border-radius: 16px;
        padding: 24px 26px;
        border: 1px solid transparent;
        margin-bottom: 10px;
    }
    .result-card.normal {
        background: rgba(34, 168, 120, 0.12);
        border-color: rgba(34, 168, 120, 0.35);
    }
    .result-card.suspicious {
        background: rgba(217, 122, 44, 0.12);
        border-color: rgba(217, 122, 44, 0.35);
    }
    .result-badge {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .result-card.normal .result-badge { color: #0D6B4C; }
    .result-card.suspicious .result-badge { color: #8A3F08; }

    .explanation-box {
        background: var(--secondary-background-color);
        border-radius: 14px;
        padding: 20px 24px;
        line-height: 1.65;
        font-size: 15px;
        margin-top: 6px;
    }

    .footer-note {
        text-align: center;
        font-size: 12.5px;
        opacity: 0.55;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------
# Disclaimer (top of page, per requirements)
# -----------------------
st.markdown(
    """
    <div class="disclaimer-banner">
        &#9888; <strong>Disclaimer:</strong> This application is an AI demonstration
        created for educational and research purposes only. It is <strong>NOT FDA
        approved</strong>. It is <strong>NOT a medical device</strong>. It must
        <strong>NOT</strong> be used to diagnose disease. Always consult a licensed
        healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)

if model_load_error:
    st.error(f"Model unavailable: {model_load_error}")
    st.stop()


# -----------------------
# Header
# -----------------------
st.markdown('<div class="eyebrow">DenseNet121 &middot; Binary classifier &middot; Demo</div>', unsafe_allow_html=True)
st.title("See what the model sees in a chest X-ray.")
st.write(
    "Upload a chest X-ray image and PulmoScan AI will classify it as "
    "**Normal** or **Suspicious** (mass or nodule), with a confidence "
    "score. Built for education and research \u2014 not for clinical decisions."
)

st.divider()

# -----------------------
# Session state defaults
# -----------------------
if "active_image_bytes" not in st.session_state:
    st.session_state.active_image_bytes = None
if "active_image_name" not in st.session_state:
    st.session_state.active_image_name = None
if "result" not in st.session_state:
    st.session_state.result = None
if "gradcam_uri" not in st.session_state:
    st.session_state.gradcam_uri = None
if "original_uri" not in st.session_state:
    st.session_state.original_uri = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Original"
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "is_dicom" not in st.session_state:
    st.session_state.is_dicom = False
if "dicom_metadata" not in st.session_state:
    st.session_state.dicom_metadata = None
if "dicom_original_name" not in st.session_state:
    st.session_state.dicom_original_name = None
if "history" not in st.session_state:
    st.session_state.history = []


def _reset_result_state():
    st.session_state.result = None
    st.session_state.gradcam_uri = None
    st.session_state.pdf_bytes = None


def _select_example(filename: str):
    path = os.path.join(EXAMPLES_DIR, filename)
    with open(path, "rb") as f:
        st.session_state.active_image_bytes = f.read()
    st.session_state.active_image_name = filename
    st.session_state.is_dicom = False
    st.session_state.dicom_metadata = None
    st.session_state.dicom_original_name = None
    _reset_result_state()


# -----------------------
# Upload widget
# -----------------------
uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg", "dcm"],
    help="PNG, JPG, JPEG, or DICOM (.dcm) \u00b7 up to 10 MB",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    already_loaded = file_bytes == st.session_state.active_image_bytes or (
        st.session_state.is_dicom and uploaded_file.name == st.session_state.dicom_original_name
    )

    if not already_loaded:
        if is_dicom_file(uploaded_file.name):
            try:
                dicom_image, dicom_metadata = load_dicom(file_bytes)
            except InvalidDicomError as exc:
                st.error(str(exc))
            else:
                buffer = io.BytesIO()
                dicom_image.save(buffer, format="PNG")
                st.session_state.active_image_bytes = buffer.getvalue()
                stem = os.path.splitext(uploaded_file.name)[0]
                st.session_state.active_image_name = f"{stem}_converted.png"
                st.session_state.is_dicom = True
                st.session_state.dicom_metadata = dicom_metadata
                st.session_state.dicom_original_name = uploaded_file.name
                _reset_result_state()
        else:
            st.session_state.active_image_bytes = file_bytes
            st.session_state.active_image_name = uploaded_file.name
            st.session_state.is_dicom = False
            st.session_state.dicom_metadata = None
            st.session_state.dicom_original_name = None
            _reset_result_state()


# -----------------------
# Example gallery
# -----------------------
st.markdown("**Don't have an X-ray handy? Try a sample.**")
st.caption("Six real chest X-rays from the ChestX-ray8 dataset \u2014 pick one to see how the model classifies it.")

cols = st.columns(6)
for i, filename in enumerate(EXAMPLE_FILES):
    with cols[i]:
        img_path = os.path.join(EXAMPLES_DIR, filename)
        st.image(img_path, width='stretch')
        if st.button(f"Sample {i + 1}", key=f"example_btn_{i}", width='stretch'):
            _select_example(filename)
            st.rerun()

st.divider()

# -----------------------
# Preview + Analyze
# -----------------------
if st.session_state.active_image_bytes:
    preview_col, _ = st.columns([1, 1])
    with preview_col:
        st.image(st.session_state.active_image_bytes, caption=st.session_state.active_image_name, width='stretch')

    if st.session_state.is_dicom:
        with st.expander("\U0001FA7A Patient details (from DICOM header)"):
            st.caption(
                "\u26a0 This file's header may contain real patient information. "
                "It is shown only in your current browser session and is never "
                "stored, logged, or sent anywhere beyond this analysis."
            )
            if st.session_state.dicom_metadata:
                st.table(
                    pd.DataFrame(
                        st.session_state.dicom_metadata.items(),
                        columns=["Field", "Value"],
                    ).set_index("Field")
                )
            else:
                st.write("No patient metadata fields were found in this file's header.")

    analyze_clicked = st.button("Analyze X-ray", type="primary", width='stretch')
    st.caption("\u26a0 For educational and research purposes only. Not a substitute for professional medical evaluation.")

    if analyze_clicked:
        filename = st.session_state.active_image_name or "upload.png"

        if not allowed_file(filename):
            st.error("Unsupported file type. Please upload a PNG, JPG, JPEG, or DICOM image.")
        else:
            try:
                image = load_image(st.session_state.active_image_bytes)
            except InvalidImageError as exc:
                st.error(str(exc))
            else:
                with st.spinner("Analyzing X-ray\u2026"):
                    try:
                        result, gradcam_uri = predict_with_gradcam(model, device, image)
                    except Exception:  # noqa: BLE001
                        st.error("Something went wrong while analyzing this image. Please try again.")
                    else:
                        mime = "image/png" if filename.lower().endswith("png") else "image/jpeg"
                        encoded = base64.b64encode(st.session_state.active_image_bytes).decode("utf-8")
                        original_uri = f"data:{mime};base64,{encoded}"

                        st.session_state.result = result
                        st.session_state.gradcam_uri = gradcam_uri
                        st.session_state.original_uri = original_uri
                        st.session_state.view_mode = "Original"
                        st.session_state.pdf_bytes = None

                        # Suspicious-probability on a single consistent 0-100
                        # scale, regardless of which class "won" -- makes a
                        # sensible y-axis for the trend chart across scans.
                        susp_prob = (
                            result["confidence"]
                            if result["prediction"] == "Suspicious"
                            else round(100 - result["confidence"], 1)
                        )
                        display_name = (
                            st.session_state.dicom_original_name
                            if st.session_state.is_dicom
                            else st.session_state.active_image_name
                        )
                        st.session_state.history.append({
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "File": display_name,
                            "Prediction": result["prediction"],
                            "Suspicious probability (%)": susp_prob,
                        })

                        st.rerun()
else:
    st.info("Upload an X-ray above, or pick a sample to try the classifier.")


# -----------------------
# Results
# -----------------------
if st.session_state.result:
    result = st.session_state.result
    prediction = result["prediction"]
    confidence = result["confidence"]
    css_class = "normal" if prediction == "Normal" else "suspicious"

    st.divider()
    st.markdown('<div class="eyebrow">Analysis complete</div>', unsafe_allow_html=True)
    st.header("Here's what the model found.")

    image_col, summary_col = st.columns([1, 1])

    with image_col:
        st.session_state.view_mode = st.radio(
            "View",
            options=["Original", "Grad-CAM"],
            horizontal=True,
            label_visibility="collapsed",
            key="view_mode_radio",
            index=0 if st.session_state.view_mode == "Original" else 1,
        )
        shown_uri = (
            st.session_state.original_uri
            if st.session_state.view_mode == "Original"
            else st.session_state.gradcam_uri
        )
        render_zoomable_image(shown_uri, height=360)

    with summary_col:
        badge_icon = "\u2705" if prediction == "Normal" else "\u26a0\ufe0f"
        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <div class="result-badge">{badge_icon} {prediction}</div>
                <div style="font-size: 13px; opacity: 0.75; margin-bottom: 6px;">Model confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(int(confidence), 100), text=f"{confidence}%")
        st.caption(
            "\u26a0 This AI is intended only for educational and research "
            "purposes and must not be used as a substitute for professional "
            "medical evaluation."
        )

    st.markdown("#### What does this mean?")
    if prediction == "Normal":
        st.write(
            "The model did not detect image features resembling a "
            "suspicious mass or nodule in this X-ray."
        )
        st.write(
            "**This does not guarantee the absence of disease.** The AI "
            "only recognizes patterns similar to what it saw during "
            "training, and subtle or atypical findings can be missed. A "
            "normal result from this tool is not a clean bill of health."
        )
    else:
        st.write(
            "The AI detected features similar to masses or nodules found "
            "in its training data."
        )
        st.write(
            "**We recommend consulting a qualified healthcare "
            "professional** to properly evaluate this X-ray."
        )
        st.write(
            "This result is **not a diagnosis**. It reflects a statistical "
            "pattern match from a machine learning model, not a clinical "
            "assessment."
        )
    st.caption(
        "\U0001F4A1 Tip: switch the view above to **Grad-CAM** to see which "
        "regions of the X-ray most influenced this result."
    )

    # Build the PDF once per result, cache it in session state so it's
    # not regenerated on every rerun (e.g. when toggling the view above).
    if st.session_state.pdf_bytes is None:
        pdf_buffer = build_pdf_report(
            prediction=prediction,
            confidence=confidence,
            image_data_uri=st.session_state.original_uri,
            gradcam_data_uri=st.session_state.gradcam_uri,
            explanation=EXPLANATIONS[prediction],
        )
        st.session_state.pdf_bytes = pdf_buffer.read()

    st.download_button(
        "Download PDF report",
        data=st.session_state.pdf_bytes,
        file_name="pulmoscan-ai-report.pdf",
        mime="application/pdf",
        width='stretch',
    )

    st.write("")
    report_text = (
        f"Analysis result: {prediction}, with {confidence} percent model confidence. "
        f"{EXPLANATIONS[prediction]}"
    )
    render_read_aloud_button(report_text)


# -----------------------
# Session history / trend chart
# -----------------------
if st.session_state.history:
    st.divider()
    st.markdown("#### Session history")
    st.caption(
        "Suspicious-probability across the scans you've run this session "
        "(resets when you close or reload the app)."
    )

    history_df = pd.DataFrame(st.session_state.history)
    chart_df = history_df[["Suspicious probability (%)"]].copy()
    chart_df.index = [f"Scan {i + 1}" for i in range(len(history_df))]
    st.line_chart(chart_df, height=220)
    st.dataframe(history_df, width='stretch', hide_index=True)

st.markdown(
    '<div class="footer-note">PulmoScan AI &mdash; an educational demonstration project. Not for clinical use.</div>',
    unsafe_allow_html=True,
)
