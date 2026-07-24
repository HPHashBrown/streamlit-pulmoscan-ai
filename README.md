---
title: PulmoScan AI
emoji: 🫁
---

# PulmoScan AI — Streamlit edition

A DenseNet121 chest X-ray classifier (Normal vs. Suspicious) with
Grad-CAM heatmaps and PDF report downloads, built as a Streamlit app.
**Not FDA approved. Not a medical device. Not for clinical use.**

This is a port of the original Flask version to Streamlit, done to get
more free RAM headroom (Streamlit Community Cloud gives 1GB vs. Render's
free-tier 512MB) since Grad-CAM's backward pass is memory-hungry. All the
model/inference logic (`model.py`, `predict.py`, `report.py`) is reused
unchanged from the Flask version — only the UI layer is new.

## Features

- Upload PNG/JPG/JPEG **or DICOM (.dcm)** chest X-rays
- Drag-and-drop upload (built into Streamlit's file uploader)
- DICOM header metadata display (Patient Name/ID, Study Date, etc.) —
  read directly from the file's structured header fields, not OCR
- Prediction with confidence score
- Grad-CAM heatmap toggle
- Zoomable/pannable image viewer (scroll to zoom, drag to pan)
- "Read my report aloud" — browser-native text-to-speech, only starts
  on a button click, never automatically
- Session history: a chart tracking suspicious-probability across every
  scan you run in the current session
- PDF report download
- Light/dark mode (Streamlit's built-in toggle, custom brand colors)
- Example gallery of real chest X-rays to try without your own file

## Project structure

```
streamlit-lung-ai/
├── streamlit_app.py        # Main page: upload, examples, results, PDF download
├── pages/
│   └── 1_About.py           # Model card, training data, citations, metrics
├── model.py                  # Loads DenseNet121 + trained weights (unchanged)
├── predict.py                 # Preprocessing, prediction, Grad-CAM (unchanged)
├── report.py                   # PDF report generation (unchanged)
├── dicom_utils.py                # DICOM file reading + header metadata (new)
├── components_ui.py               # Zoom/pan viewer + read-aloud button (new)
├── lung_model.pth                  # Trained model weights
├── examples/                        # Sample X-rays for the "try it out" gallery
├── requirements.txt
└── .streamlit/
    └── config.toml                    # Custom light/dark theme colors
```

## Run locally

```bash
cd streamlit-lung-ai
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the local URL it prints (usually http://localhost:8501).

Dark/light mode: click the **⋮** menu in the top-right corner of the app
→ **Settings** → **Theme**. This is Streamlit's built-in toggle; the
custom colors in `.streamlit/config.toml` apply to both automatically.

## Deploy to Streamlit Community Cloud (free)

1. Push this project to a GitHub repo (same process as before: GitHub
   Desktop → Add Local Repository → Commit → Publish repository).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **New app**, select your repo and branch.
4. Set **Main file path** to `streamlit_app.py`.
5. Click **Deploy**.

That's it — no build command, no Start Command, no Docker file needed.
Streamlit Cloud reads `requirements.txt` and `.streamlit/config.toml`
automatically. The first deploy will take a few minutes (installing
PyTorch); after that, redeploys on new commits are much faster.

## Notes

- Uploaded images are processed in memory only, never written to disk.
- The model loads once per server process via `st.cache_resource`, not
  on every interaction — this is important in Streamlit, since the
  whole script re-runs on every button click/widget change.
- If you still hit memory limits on Streamlit Cloud's free tier, the
  same CPU-only PyTorch trick from the Flask version can be applied by
  adding `--extra-index-url https://download.pytorch.org/whl/cpu` as
  the first line of `requirements.txt`.
