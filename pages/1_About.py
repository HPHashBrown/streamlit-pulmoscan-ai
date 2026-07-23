"""
pages/1_About.py

Model card / About page: architecture, training data, citations, and
evaluation metrics. Static content, no model inference happens here.
"""

import streamlit as st

st.set_page_config(page_title="About \u2014 PulmoScan AI", page_icon="\U0001FAC1", layout="centered")

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
    .info-card {
        background: var(--secondary-background-color);
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }
    .info-card h4 {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.6;
        margin: 0 0 4px;
    }
    .info-card .value {
        font-size: 16px;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0 0 8px;
    }
    .metric-card {
        background: color-mix(in srgb, var(--primary-color) 8%, var(--secondary-background-color));
        border-radius: 12px;
        padding: 14px 16px;
        text-align: left;
    }
    .metric-card .label {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.65;
        font-weight: 600;
    }
    .metric-card .number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 500;
        color: var(--primary-color);
    }
    .reference-list li { margin-bottom: 10px; font-size: 13.5px; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.markdown('<div class="eyebrow">Model card</div>', unsafe_allow_html=True)
st.title("About PulmoScan AI")
st.write(
    "A transparent look at how this demo model was built, trained, and "
    "evaluated \u2014 and where its limits are."
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <div class="info-card">
            <h4>Architecture</h4>
            <p class="value">DenseNet121</p>
            <p>A densely-connected convolutional network where each layer receives
            the feature maps of all preceding layers, chosen for strong performance
            on medical imaging tasks with efficient parameter use.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="info-card">
            <h4>Transfer learning</h4>
            <p class="value">ImageNet-pretrained backbone</p>
            <p>The convolutional backbone started from ImageNet-pretrained weights.
            The final classifier layer was replaced with a 2-class linear head and
            fine-tuned on chest X-ray data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="info-card">
            <h4>Framework</h4>
            <p class="value">PyTorch &amp; torchvision</p>
            <p>Model definition, training loop, and inference are all implemented in
            PyTorch, with torchvision supplying the DenseNet121 backbone and image
            transforms.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="info-card">
            <h4>Task</h4>
            <p class="value">Binary classification</p>
            <p>Each chest X-ray is classified as <strong>Normal</strong> or
            <strong>Suspicious</strong> (mass or nodule), using images resized to
            224&times;224 and normalized with standard ImageNet statistics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("### Training data")
st.write(
    "The model was trained on publicly available chest X-ray datasets, "
    "split into training, validation, and test sets. Class imbalance "
    "between normal and suspicious cases was addressed with weighted "
    "sampling during training. Data augmentation (flips, small rotations, "
    "affine jitter, and brightness/contrast jitter) was used to improve "
    "robustness."
)

st.markdown("### Data sources &amp; references")
st.markdown(
    """
    <ol class="reference-list">
      <li>
        Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri,
        Ronald Summers. <em>ChestX-ray8: Hospital-scale Chest X-ray Database
        and Benchmarks on Weakly-Supervised Classification and Localization
        of Common Thorax Diseases.</em> IEEE CVPR, pp. 3462&ndash;3471, 2017.
      </li>
      <li>
        Hoo-chang Shin, Kirk Roberts, Le Lu, Dina Demner-Fushman, Jianhua
        Yao, Ronald M. Summers. <em>Learning to Read Chest X-Rays: Recurrent
        Neural Cascade Model for Automated Image Annotation.</em> IEEE CVPR,
        pp. 2497&ndash;2506, 2016.
      </li>
      <li>
        Open-i: An open access biomedical search engine.
        <a href="https://openi.nlm.nih.gov" target="_blank">https://openi.nlm.nih.gov</a>
      </li>
    </ol>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.markdown("### Evaluation metrics")
st.caption("Measured on a held-out test set that the model never saw during training.")

metrics = [
    ("Accuracy", "87.0%"),
    ("Precision", "54.3%"),
    ("Recall", "41.2%"),
    ("F1 Score", "46.8%"),
    ("Sensitivity", "41.2%"),
    ("Specificity", "94.4%"),
]

for row_start in range(0, len(metrics), 3):
    row = metrics[row_start:row_start + 3]
    row_cols = st.columns(3)
    for col, (label, value) in zip(row_cols, row):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="number">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")

st.markdown(
    """
    <div class="metric-card">
        <div class="label">ROC&ndash;AUC</div>
        <div class="number">0.798</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.info(
    "Recall/sensitivity on the suspicious class is limited, meaning the "
    "model misses a meaningful share of suspicious cases in this "
    "evaluation. This is exactly why the tool is a research demonstration, "
    "not a diagnostic aid."
)

st.warning(
    "\u26a0 PulmoScan AI is an educational and research demonstration only. "
    "It is not FDA approved, is not a medical device, and must never be "
    "used to diagnose disease or guide treatment. Always consult a "
    "licensed healthcare professional."
)
