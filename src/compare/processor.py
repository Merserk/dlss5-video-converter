from __future__ import annotations

import numpy as np
from PIL import Image

from ..image.decoder import decode_image
from .models import DiffMetrics

# A pixel counts as "changed" once any channel moves more than this many levels
# (out of 255). A hard 0 would flag ordinary lossy-codec rounding noise as change.
CHANGED_PIXEL_THRESHOLD = 2


def _load_rgb(path: str) -> Image.Image:
    """Decode any image the app can produce or accept as input, as RGB.

    Tries the app's own decoder first (handles RAW/HEIC/SVG/etc., the same path
    used for uploads), then falls back to plain Pillow for anything it doesn't
    recognize (this covers every rendered output format, which are always
    plain raster files).
    """
    try:
        decoded = decode_image(path)
        return Image.fromarray(decoded.rgba, mode="RGBA").convert("RGB")
    except Exception:
        with Image.open(path) as handle:
            return handle.convert("RGB")


def compute_diff(reference_path: str, candidate_path: str, amplify: float = 4.0) -> tuple[Image.Image, DiffMetrics]:
    """Compare two images pixel-for-pixel.

    Only meaningful at 1:1 — if the candidate is a different resolution than the
    reference (e.g. an upscaled DLSS output), it's resampled down/up to the
    reference's size with Lanczos before diffing, purely so the arrays line up.
    `metrics.resampled` tells the caller this happened so the UI can flag it;
    the resulting numbers describe a like-for-like comparison at that resolution,
    not a claim about the candidate's native detail.
    """
    reference = _load_rgb(reference_path)
    candidate = _load_rgb(candidate_path)
    candidate_size = candidate.size
    resampled = candidate.size != reference.size
    compared_candidate = (
        candidate.resize(reference.size, Image.Resampling.LANCZOS) if resampled else candidate
    )

    ref_arr = np.asarray(reference, dtype=np.float32)
    cand_arr = np.asarray(compared_candidate, dtype=np.float32)
    delta = cand_arr - ref_arr
    abs_delta = np.abs(delta)
    per_pixel_max = abs_delta.max(axis=2)

    metrics = DiffMetrics(
        reference_size=reference.size,
        candidate_size=candidate_size,
        compared_size=reference.size,
        resampled=resampled,
        mean_abs_error=float(abs_delta.mean()),
        root_mean_square_error=float(np.sqrt(np.mean(delta ** 2))),
        changed_pixels_pct=float((per_pixel_max > CHANGED_PIXEL_THRESHOLD).mean() * 100.0),
        max_channel_delta=int(per_pixel_max.max()) if per_pixel_max.size else 0,
    )

    amplified = np.clip(per_pixel_max * max(float(amplify), 0.1), 0, 255).astype(np.uint8)
    diff_image = Image.fromarray(amplified, mode="L")
    return diff_image, metrics
