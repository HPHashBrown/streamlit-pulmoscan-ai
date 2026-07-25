"""
predict.py

Image preprocessing and inference logic for the lung X-ray classifier.
Kept separate from app.py so the Flask routes stay thin and this logic
can be tested or reused independently.
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, UnidentifiedImageError
from torchvision import transforms

from model import CLASS_NAMES

# Must exactly mirror the validation/eval transform used in train_model.py
# so inference sees images preprocessed the same way the model was
# evaluated on during training.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Same resize/grayscale as above but no normalization — used as the plain
# RGB base image that the Grad-CAM heatmap gets blended on top of.
DISPLAY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
])

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


class InvalidImageError(Exception):
    """Raised when an uploaded file cannot be read as a valid image."""
    pass


def allowed_file(filename: str) -> bool:
    """Check whether a filename has a supported image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def load_image(file_bytes: bytes) -> Image.Image:
    """
    Open raw uploaded bytes as a PIL Image, raising a friendly
    InvalidImageError for anything corrupted or unreadable.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # force decode now so truncated/corrupt files fail here
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "The uploaded file could not be read as a valid image. "
            "It may be corrupted or in an unsupported format."
        ) from exc


def predict_image(model, device, image: Image.Image) -> dict:
    """
    Run the full preprocessing + inference pipeline on a single PIL image.

    Returns a dict with the predicted class, confidence, and the full
    probability breakdown across both classes.
    """
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item()) * 100

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 1),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i].item()) * 100, 1)
            for i in range(len(CLASS_NAMES))
        },
    }


def check_image_quality(image: Image.Image) -> dict:
    """
    A few cheap, genuine sanity checks on the uploaded image -- not a
    medical quality check, just basic technical adequacy for the model
    (resolution, aspect ratio). Used to populate the "Quality verified"
    step of the analysis checklist honestly rather than as a fake delay.
    """
    width, height = image.size
    warnings = []
    if width < 100 or height < 100:
        warnings.append("Image resolution is very low; results may be less reliable.")
    aspect_ratio = width / height if height else 0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        warnings.append("Unusual aspect ratio for a chest X-ray.")
    return {"ok": len(warnings) == 0, "warnings": warnings, "size": (width, height)}


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """
    Map a 2D array of values in [0, 1] to an RGB heat-style colormap
    (blue -> cyan -> green -> yellow -> red), without needing matplotlib.
    Returns an array of shape (H, W, 3) with values in [0, 1].
    """
    r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def predict_with_gradcam(model, device, image: Image.Image):
    """
    Run prediction AND Grad-CAM together, sharing a single forward pass
    through the model instead of two separate ones. This roughly halves
    the compute time/memory of a full analysis, which matters a lot on
    slow or memory-constrained hosts (e.g. free-tier cloud servers) where
    a naive "predict, then separately re-run for Grad-CAM" approach can
    be slow enough to trip a request timeout.

    Returns (result_dict, raw_cam). raw_cam is a (224, 224) numpy array
    of activation values in [0, 1] -- the *raw* Grad-CAM map, before any
    threshold/coloring/blending is applied. Keeping this separate from
    rendering means the visualization (heatmap overlay, threshold slider,
    bounding box, stats) can all be recomputed cheaply afterwards without
    re-running the model.
    """
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    activation = model.features(tensor)      # (1, C, 7, 7)
    activation.retain_grad()

    relu_out = F.relu(activation)
    pooled = F.adaptive_avg_pool2d(relu_out, (1, 1))
    pooled = torch.flatten(pooled, 1)
    logits = model.classifier(pooled)

    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(torch.argmax(logits, dim=1).item())
    confidence = float(probabilities[predicted_index].item()) * 100
    result = {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 1),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i].item()) * 100, 1)
            for i in range(len(CLASS_NAMES))
        },
    }

    score = logits[0, predicted_index]
    score.backward()

    gradient = activation.grad[0]             # (C, H, W)
    activation_values = activation[0].detach()  # (C, H, W)

    weights = gradient.mean(dim=(1, 2))        # (C,)
    cam = torch.relu((weights[:, None, None] * activation_values).sum(dim=0))

    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Free the graph/tensors from this request as soon as we're done
    # with them, rather than waiting for garbage collection.
    del tensor, activation, relu_out, pooled, logits, gradient, activation_values
    model.zero_grad(set_to_none=True)

    # Upsample the (7, 7) CAM to full image resolution.
    cam_tensor = torch.tensor(cam)[None, None, :, :]
    raw_cam = F.interpolate(
        cam_tensor, size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    return result, raw_cam


def compute_gradcam_stats(raw_cam: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Derive explainability stats from an existing raw Grad-CAM map --
    no model call needed. threshold is in [0, 1].
    """
    h, w = raw_cam.shape
    mask = raw_cam >= threshold

    peak_y, peak_x = np.unravel_index(np.argmax(raw_cam), raw_cam.shape)
    peak_value = float(raw_cam[peak_y, peak_x])

    # Human-readable region name from a 3x3 grid of the image.
    col = min(int(peak_x / w * 3), 2)
    row = min(int(peak_y / h * 3), 2)
    row_names = ["upper", "mid", "lower"]
    col_names = ["left", "center", "right"]
    if row == 1 and col == 1:
        region_name = "center"
    else:
        region_name = f"{row_names[row]} {col_names[col]}".strip()

    if mask.any():
        ys, xs = np.where(mask)
        bounding_box = {
            "x_min": int(xs.min()), "x_max": int(xs.max()),
            "y_min": int(ys.min()), "y_max": int(ys.max()),
        }
    else:
        bounding_box = None

    return {
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "peak_x_pct": round(float(peak_x) / w * 100, 1),
        "peak_y_pct": round(float(peak_y) / h * 100, 1),
        "peak_value": round(peak_value, 3),
        "region_name": region_name,
        "percent_highlighted": round(float(mask.mean()) * 100, 1),
        "bounding_box": bounding_box,
        "image_size": (w, h),
    }


def render_gradcam_overlay(
    raw_cam: np.ndarray,
    image: Image.Image,
    threshold: float = 0.0,
    alpha: float = 0.45,
    draw_bbox: bool = False,
) -> str:
    """
    Render a raw Grad-CAM map into a displayable base64 PNG data URI,
    blended onto the (resized) original image. Pure post-processing --
    no model inference -- so this is cheap enough to call on every move
    of a threshold slider.

    threshold: activations below this (0-1) are shown as plain X-ray,
               not tinted -- lets the user isolate only the strongest
               regions.
    draw_bbox: outline the bounding box of all activations >= threshold.
    """
    cam = raw_cam.copy()
    if threshold > 0:
        cam = np.where(cam >= threshold, cam, 0.0)

    heatmap_rgb = _colorize(cam)
    base_image = np.array(DISPLAY_TRANSFORM(image)).astype(np.float32) / 255.0

    mask = (cam > 0)[..., None]
    blended = np.where(mask, base_image * (1 - alpha) + heatmap_rgb * alpha, base_image)
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    overlay_image = Image.fromarray(blended)

    if draw_bbox:
        stats = compute_gradcam_stats(raw_cam, threshold=max(threshold, 0.5))
        bbox = stats["bounding_box"]
        if bbox:
            draw = ImageDraw.Draw(overlay_image)
            draw.rectangle(
                [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]],
                outline=(255, 255, 255),
                width=2,
            )

    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_gradcam(model, device, image: Image.Image) -> str:
    """Convenience wrapper: run Grad-CAM and return a default-rendered overlay."""
    _result, raw_cam = predict_with_gradcam(model, device, image)
    return render_gradcam_overlay(raw_cam, image)
