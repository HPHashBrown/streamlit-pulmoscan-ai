"""
streamlit_app.py

Main page of the PulmoScan AI Streamlit app: upload/example selection,
analysis (prediction + Grad-CAM), results display, and PDF download.

Model/inference logic lives in model.py and predict.py. PDF generation
lives in report.py. DICOM handling lives in dicom_utils.py. Custom
HTML/JS widgets (viewer, voice) live in components_ui.py. This file
handles the UI/page flow only.
"""

import base64
import io
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from model import load_model
from predict import (
    allowed_file,
    load_image,
    predict_with_gradcam,
    render_gradcam_overlay,
    compute_gradcam_stats,
    check_image_quality,
    InvalidImageError,
)
from report import build_pdf_report
from dicom_utils import is_dicom_file, load_dicom, get_pixel_spacing_mm, InvalidDicomError
from components_ui import render_advanced_viewer, render_read_aloud_button

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

VIEWER_MODES = ["Original", "Grad-CAM", "Side-by-side", "Split slider", "Flicker"]

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
        margin-bottom: 10px;
        color: var(--text-color);
    }

    .privacy-note {
        font-size: 12.5px;
        opacity: 0.7;
        margin-bottom: 18px;
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

    .conf-bar-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
    .conf-bar-label { width: 78px; font-size: 12.5px; font-weight: 600; flex-shrink:0; }
    .conf-bar-track { flex:1; height:14px; border-radius:999px; background: color-mix(in srgb, var(--text-color) 10%, transparent); overflow:hidden; }
    .conf-bar-fill { height:100%; border-radius:999px; }
    .conf-bar-pct { width: 52px; text-align:right; font-family:'IBM Plex Mono', monospace; font-size:12.5px; flex-shrink:0; }

    .explain-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
    .explain-card {
        background: var(--secondary-background-color);
        border-radius: 12px; padding: 12px 14px;
    }
    .explain-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.6; font-weight: 600; }
    .explain-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 16px; font-weight: 500; color: var(--primary-color); margin-top: 2px; }

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
# Disclaimer (top of page, per requirements) + accurate privacy note
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
    <div class="privacy-note">
        &#128274; Images are processed in memory on the server for analysis and are
        never written to disk, stored, or logged. (Note: this is a hosted app, so
        images are sent to the server for processing \u2014 if you'd rather nothing
        leave your own computer, run this app locally; see the README.)
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
_DEFAULTS = {
    "active_image_bytes": None,
    "active_image_name": None,
    "result": None,
    "raw_cam": None,
    "original_uri": None,
    "pdf_bytes": None,
    "is_dicom": False,
    "dicom_metadata": None,
    "dicom_original_name": None,
    "dicom_pixel_spacing": None,
    "history": [],
    "viewer_mode": "Original",
    "heatmap_threshold_pct": 0,
    "show_bbox": False,
    # Settings (also editable on the Settings page)
    "settings_alpha": 0.45,
    "settings_default_threshold": 0,
    "settings_default_bbox": False,
    "settings_default_mode": "Original",
    "settings_default_zoom_pct": 100,
    "settings_flicker_ms": 600,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


def _reset_result_state():
    st.session_state.result = None
    st.session_state.raw_cam = None
    st.session_state.pdf_bytes = None
    st.session_state.viewer_mode = st.session_state.settings_default_mode
    st.session_state.heatmap_threshold_pct = st.session_state.settings_default_threshold
    st.session_state.show_bbox = st.session_state.settings_default_bbox


def _select_example(filename: str):
    path = os.path.join(EXAMPLES_DIR, filename)
    with open(path, "rb") as f:
        st.session_state.active_image_bytes = f.read()
    st.session_state.active_image_name = filename
    st.session_state.is_dicom = False
    st.session_state.dicom_metadata = None
    st.session_state.dicom_original_name = None
    st.session_state.dicom_pixel_spacing = None
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
                st.session_state.dicom_pixel_spacing = get_pixel_spacing_mm(file_bytes)
                _reset_result_state()
        else:
            st.session_state.active_image_bytes = file_bytes
            st.session_state.active_image_name = uploaded_file.name
            st.session_state.is_dicom = False
            st.session_state.dicom_metadata = None
            st.session_state.dicom_original_name = None
            st.session_state.dicom_pixel_spacing = None
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
        st.image(img_path, width="stretch")
        if st.button(f"Sample {i + 1}", key=f"example_btn_{i}", width="stretch"):
            _select_example(filename)
            st.rerun()

st.divider()

# -----------------------
# Preview + metadata inspector + Analyze
# -----------------------
if st.session_state.active_image_bytes:
    preview_col, _ = st.columns([1, 1])
    with preview_col:
        st.image(st.session_state.active_image_bytes, caption=st.session_state.active_image_name, width="stretch")

    preview_image = Image.open(io.BytesIO(st.session_state.active_image_bytes)).convert("RGB")

    with st.expander("\U0001F50E Metadata inspector"):
        basic_rows = [
            ("Dimensions", f"{preview_image.size[0]} \u00d7 {preview_image.size[1]} px"),
            ("Mode / bit depth", f"{preview_image.mode} (8 bits/channel)"),
            ("File size", f"{len(st.session_state.active_image_bytes) / 1024:.1f} KB"),
        ]
        if st.session_state.is_dicom and st.session_state.dicom_pixel_spacing:
            spacing_at_224 = st.session_state.dicom_pixel_spacing
            basic_rows.append(("Pixel spacing (at 224\u00d7224 display)", f"{spacing_at_224:.3f} mm/px"))
        st.table(pd.DataFrame(basic_rows, columns=["Field", "Value"]).set_index("Field"))

        if st.session_state.is_dicom:
            st.caption(
                "\u26a0 The fields below come from this file's DICOM header and may "
                "contain real patient information. Shown only in your current "
                "session \u2014 never stored, logged, or sent anywhere beyond this analysis."
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

    analyze_clicked = st.button("Analyze X-ray", type="primary", width="stretch")
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
                with st.status("Analyzing X-ray\u2026", expanded=True) as status:
                    st.write("\u2705 Image loaded")
                    time.sleep(0.15)

                    quality = check_image_quality(image)
                    if quality["ok"]:
                        st.write("\u2705 Quality verified")
                    else:
                        st.write("\u26a0\ufe0f Quality check: " + "; ".join(quality["warnings"]))
                    time.sleep(0.15)

                    try:
                        result, raw_cam = predict_with_gradcam(model, device, image)
                        st.write("\u2705 Lung regions analyzed")

                        default_uri = render_gradcam_overlay(
                            raw_cam, image,
                            threshold=st.session_state.settings_default_threshold / 100,
                            alpha=st.session_state.settings_alpha,
                        )
                        st.write("\u2705 Heatmap generated")

                        mime = "image/png" if filename.lower().endswith("png") else "image/jpeg"
                        encoded = base64.b64encode(st.session_state.active_image_bytes).decode("utf-8")
                        original_uri = f"data:{mime};base64,{encoded}"

                        pdf_buffer = build_pdf_report(
                            prediction=result["prediction"],
                            confidence=result["confidence"],
                            image_data_uri=original_uri,
                            gradcam_data_uri=default_uri,
                            explanation=EXPLANATIONS[result["prediction"]],
                        )
                        pdf_bytes = pdf_buffer.read()
                        st.write("\u2705 Report generated")

                    except Exception:  # noqa: BLE001
                        status.update(label="Analysis failed", state="error")
                        st.error("Something went wrong while analyzing this image. Please try again.")
                    else:
                        status.update(label="Analysis complete", state="complete")

                        st.session_state.result = result
                        st.session_state.raw_cam = raw_cam
                        st.session_state.original_uri = original_uri
                        st.session_state.pdf_bytes = pdf_bytes
                        st.session_state.viewer_mode = st.session_state.settings_default_mode
                        st.session_state.heatmap_threshold_pct = st.session_state.settings_default_threshold
                        st.session_state.show_bbox = st.session_state.settings_default_bbox

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
                            "Favorite": False,
                            "Notes": "",
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
    raw_cam = st.session_state.raw_cam
    current_image = load_image(st.session_state.active_image_bytes)

    st.divider()
    st.markdown('<div class="eyebrow">Analysis complete</div>', unsafe_allow_html=True)
    st.header("Here's what the model found.")

    # --- Viewer controls ---
    control_col1, control_col2, control_col3 = st.columns([1.3, 1, 1])
    with control_col1:
        st.session_state.viewer_mode = st.selectbox(
            "Viewer mode", VIEWER_MODES,
            index=VIEWER_MODES.index(st.session_state.viewer_mode)
            if st.session_state.viewer_mode in VIEWER_MODES else 0,
        )
    with control_col2:
        st.session_state.heatmap_threshold_pct = st.slider(
            "Heatmap threshold", 0, 95, st.session_state.heatmap_threshold_pct,
            help="Higher = only the strongest activated regions are highlighted.",
        )
    with control_col3:
        st.session_state.show_bbox = st.checkbox(
            "Show bounding box", value=st.session_state.show_bbox,
            help="Outline the region above the current threshold.",
        )

    threshold_frac = st.session_state.heatmap_threshold_pct / 100
    heatmap_uri = render_gradcam_overlay(
        raw_cam, current_image,
        threshold=threshold_frac,
        alpha=st.session_state.settings_alpha,
        draw_bbox=st.session_state.show_bbox,
    )

    pixel_spacing = st.session_state.dicom_pixel_spacing if st.session_state.is_dicom else None

    render_advanced_viewer(
        st.session_state.original_uri,
        heatmap_uri,
        mode=st.session_state.viewer_mode,
        height=380,
        pixel_spacing_mm=pixel_spacing,
        initial_zoom=st.session_state.settings_default_zoom_pct / 100,
        flicker_ms=st.session_state.settings_flicker_ms,
    )

    # --- Result badge + confidence breakdown ---
    badge_icon = "\u2705" if prediction == "Normal" else "\u26a0\ufe0f"
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-badge">{badge_icon} {prediction}</div>
        """,
        unsafe_allow_html=True,
    )

    probs = result.get("probabilities", {prediction: confidence})
    bar_colors = {"Normal": "#22A878", "Suspicious": "#D97A2C"}
    bars_html = ""
    for label, pct in probs.items():
        color = bar_colors.get(label, "#1957D6")
        bars_html += f"""
        <div class="conf-bar-row">
            <div class="conf-bar-label">{label}</div>
            <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{pct}%; background:{color};"></div></div>
            <div class="conf-bar-pct">{pct}%</div>
        </div>
        """
    st.markdown(bars_html + "</div>", unsafe_allow_html=True)

    st.caption(
        "\u26a0 This AI is intended only for educational and research "
        "purposes and must not be used as a substitute for professional "
        "medical evaluation."
    )

    # --- Explainability panel ---
    with st.expander("\U0001F9E0 Explainability panel", expanded=True):
        stats = compute_gradcam_stats(raw_cam, threshold=max(threshold_frac, 0.5))
        bbox = stats["bounding_box"]
        bbox_text = (
            f"({bbox['x_min']}, {bbox['y_min']}) \u2013 ({bbox['x_max']}, {bbox['y_max']})"
            if bbox else "N/A"
        )
        st.markdown(
            f"""
            <div class="explain-grid">
                <div class="explain-card"><div class="label">Highest attention region</div><div class="value">{stats['region_name'].title()}</div></div>
                <div class="explain-card"><div class="label">% of image highlighted</div><div class="value">{stats['percent_highlighted']}%</div></div>
                <div class="explain-card"><div class="label">Bounding box (px)</div><div class="value" style="font-size:13px;">{bbox_text}</div></div>
                <div class="explain-card"><div class="label">Peak activation coords</div><div class="value" style="font-size:13px;">x={stats['peak_x']}, y={stats['peak_y']} ({stats['peak_x_pct']}%, {stats['peak_y_pct']}%)</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "These are derived directly from the Grad-CAM heatmap above \u2014 "
            "no separate model is used to compute them."
        )

    st.markdown("#### What does this mean?")
    if prediction == "Normal":
        st.write("The model did not detect image features resembling a suspicious mass or nodule in this X-ray.")
        st.write(
            "**This does not guarantee the absence of disease.** The AI "
            "only recognizes patterns similar to what it saw during "
            "training, and subtle or atypical findings can be missed. A "
            "normal result from this tool is not a clean bill of health."
        )
    else:
        st.write("The AI detected features similar to masses or nodules found in its training data.")
        st.write("**We recommend consulting a qualified healthcare professional** to properly evaluate this X-ray.")
        st.write(
            "This result is **not a diagnosis**. It reflects a statistical "
            "pattern match from a machine learning model, not a clinical assessment."
        )
    st.caption(
        "\U0001F4A1 Tip: use the viewer mode dropdown above to compare views, "
        "or drag the threshold slider to isolate the strongest activations."
    )

    dl_col, voice_col = st.columns([1, 1])
    with dl_col:
        st.download_button(
            "Download PDF report",
            data=st.session_state.pdf_bytes,
            file_name="pulmoscan-ai-report.pdf",
            mime="application/pdf",
            width="stretch",
        )
        st.caption("PDF reflects the default heatmap view (not the adjusted threshold above).")
    with voice_col:
        report_text = (
            f"Analysis result: {prediction}, with {confidence} percent model confidence. "
            f"{EXPLANATIONS[prediction]}"
        )
        render_read_aloud_button(report_text)


# -----------------------
# Session history: trend chart + editable favorites/notes
# -----------------------
if st.session_state.history:
    st.divider()
    st.markdown("#### Session history")
    st.caption(
        "Suspicious-probability across the scans you've run this session "
        "(resets when you close or reload the app). Check Favorite or add "
        "Notes directly in the table below."
    )

    history_df = pd.DataFrame(st.session_state.history)
    chart_df = history_df[["Suspicious probability (%)"]].copy()
    chart_df.index = [f"Scan {i + 1}" for i in range(len(history_df))]
    st.line_chart(chart_df, height=220)

    edited_df = st.data_editor(
        history_df,
        width="stretch",
        hide_index=True,
        disabled=["Time", "File", "Prediction", "Suspicious probability (%)"],
        column_config={
            "Favorite": st.column_config.CheckboxColumn("\u2b50 Favorite"),
            "Notes": st.column_config.TextColumn("Notes", width="large"),
        },
        key="history_editor",
    )
    st.session_state.history = edited_df.to_dict("records")

    favorites = edited_df[edited_df["Favorite"] == True]  # noqa: E712
    if len(favorites) > 0:
        st.caption(f"\u2b50 {len(favorites)} favorited scan(s) this session.")

st.markdown(
    '<div class="footer-note">PulmoScan AI &mdash; an educational demonstration project. Not for clinical use.</div>',
    unsafe_allow_html=True,
)
