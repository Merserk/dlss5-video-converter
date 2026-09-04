from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ComparisonItem:
    """One image that can be picked as a reference or candidate in the Comparison tab.

    `label` is what shows up in the dropdowns (e.g. "Input: cat.png" or
    "Output: cat_DLSS5.png"); `path` is the real file on disk. For sent-from-Image-tab
    items this is always the full-resolution file, never a UI thumbnail.
    """

    label: str
    path: str


@dataclass(slots=True)
class DiffMetrics:
    """Numeric results of comparing two images. Purely computational — no display
    formatting or labels here; the UI layer turns this into rows for the user.
    """

    reference_size: tuple[int, int]
    candidate_size: tuple[int, int]
    compared_size: tuple[int, int]
    resampled: bool
    mean_abs_error: float
    root_mean_square_error: float
    changed_pixels_pct: float
    max_channel_delta: int
