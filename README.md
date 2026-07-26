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
- DICOM header metadata display (Patient Name/ID, Study Date, pixel
  spacing, etc.) — read directly from the file's structured header
  fields, not OCR
- **Analysis checklist** — real processing steps shown live (image
  loaded, quality verified, lung regions analyzed, heatmap generated,
  report generated), not a fake animation
- **AI confidence breakdown** — both classes' probabilities as bars,
  not just the winning one
- Grad-CAM heatmap with an **adjustable threshold slider** (isolate only
  the strongest activations) and optional bounding box overlay
- **Explainability panel** — highest-attention region, % of image
  highlighted, bounding box, and peak-activation coordinates, all
  derived directly from the existing heatmap (no extra model needed)
- **5 viewer modes**: Original, Grad-CAM, Side-by-side, Split slider,
  Flicker
- **Measurement & annotation tools**: ruler (with mm conversion when
  DICOM pixel spacing is available), angle, freehand markup
- Zoomable/pannable viewer (scroll to zoom, drag to pan) with keyboard
  shortcuts (Z zoom, F fullscreen, R reset)
- **Snapshot export** — download exactly what's on screen (including
  annotations) as a PNG
- "Read my report aloud" — browser-native text-to-speech, only starts
  on a button click, never automatically
- **Session history** with a trend chart, plus an editable table for
  marking favorites (\u2b50) and adding notes per scan
- **Metadata inspector** — resolution, dimensions, bit depth, file size,
  plus DICOM-specific fields when applicable
- PDF report download
- **Settings page** for session-scoped preferences (heatmap opacity/
  threshold defaults, default viewer mode/zoom, flicker speed)
- Light/dark mode (Streamlit's built-in toggle, custom brand colors)
- Example gallery of real chest X-rays to try without your own file

**Not included** (would need real backend infrastructure this app
doesn't have, e.g. a database and/or user accounts): persistent
collections/cases across visits, usage analytics for logged-in users,
and public share links. Also skipped: a PWA manifest (not cleanly
supported on Streamlit Cloud) and an API playground (architectural
mismatch — the old Flask version's `/predict` endpoint is the better
fit for that).

## Project structure

```
streamlit-lung-ai/
├── streamlit_app.py        # Main page: upload, examples, results, PDF download
├── pages/
│   ├── 1_About.py            # Model card, training data, citations, metrics
│   └── 2_Settings.py           # Session-scoped preferences
├── model.py                      # Loads DenseNet121 + trained weights (unchanged)
├── predict.py                     # Preprocessing, prediction, Grad-CAM
├── report.py                       # PDF report generation (unchanged)
├── dicom_utils.py                    # DICOM file reading + header metadata
├── components_ui.py                    # Viewer (zoom/pan/annotate/modes) + voice
├── lung_model.pth                        # Trained model weights
├── examples/                              # Sample X-rays for the "try it out" gallery
├── requirements.txt
└── .streamlit/
    └── config.toml                        # Custom light/dark theme colors
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
