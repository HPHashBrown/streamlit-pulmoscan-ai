"""
dicom_utils.py

DICOM (.dcm) file support for the Streamlit app: converts pixel data to
a displayable/analyzable image and reads header metadata directly (this
is plain structured header parsing, NOT optical character recognition
-- DICOM stores patient/study info as data fields, not as pixels).

Kept isolated to the Streamlit project (not shared with the original
Flask app) so that deployment doesn't gain a new dependency it doesn't
need.
"""

import io

import numpy as np
import pydicom
from PIL import Image


class InvalidDicomError(Exception):
    """Raised when a .dcm file can't be read or converted to an image."""
    pass


# Header fields worth surfacing. DICOM tags are optional by spec, so any
# of these may be absent -- especially in de-identified/anonymized files.
METADATA_FIELDS = [
    ("PatientName", "Patient Name"),
    ("PatientID", "Patient ID"),
    ("PatientBirthDate", "Patient Birth Date"),
    ("PatientSex", "Patient Sex"),
    ("StudyDate", "Study Date"),
    ("Modality", "Modality"),
    ("InstitutionName", "Institution"),
    ("BodyPartExamined", "Body Part Examined"),
]


def is_dicom_file(filename: str) -> bool:
    return filename.lower().endswith(".dcm")


def load_dicom(file_bytes: bytes):
    """
    Read a DICOM file and return (image, metadata_dict).

    image: a PIL RGB image, normalized from the DICOM pixel array, ready
           to feed into the same preprocessing/model pipeline as a
           regular PNG/JPEG upload.
    metadata_dict: header fields that were present, in display order.
    """
    try:
        dataset = pydicom.dcmread(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise InvalidDicomError(
            "This file could not be read as DICOM. It may be corrupted "
            "or in an unsupported format."
        ) from exc

    try:
        pixel_array = dataset.pixel_array.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        # Most commonly hit with compressed transfer syntaxes (e.g.
        # JPEG-lossless) that need an extra optional codec package
        # (pylibjpeg/gdcm) which this lightweight deployment doesn't
        # install, to keep the app's footprint small.
        raise InvalidDicomError(
            "This DICOM file's pixel data could not be decoded. It may "
            "use a compressed format that isn't supported here."
        ) from exc

    pixel_array -= pixel_array.min()
    if pixel_array.max() > 0:
        pixel_array = pixel_array / pixel_array.max()
    pixel_array = (pixel_array * 255).astype(np.uint8)
    image = Image.fromarray(pixel_array).convert("RGB")

    metadata = {}
    for tag, label in METADATA_FIELDS:
        value = getattr(dataset, tag, None)
        if value not in (None, ""):
            metadata[label] = str(value)

    return image, metadata
