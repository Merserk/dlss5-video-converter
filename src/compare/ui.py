from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from .models import ComparisonItem, DiffMetrics
from .processor import compute_diff

NO_SELECTION = "(none yet — send an image here from another tab)"


def _choices(pool: list[ComparisonItem]) -> list[str]:
    labels = [item.label for item in pool]
    return labels or [NO_SELECTION]


def _path_for(pool: list[ComparisonItem], label: str) -> str | None:
    for item in pool:
        if item.label == label:
            return item.path
    return None


def _metrics_rows(
    metrics: DiffMetrics, reference_label: str, candidate_label: str
) -> list[list[str]]:
    rows = [
        ["Reference", f"{reference_label}  ({metrics.reference_size[0]}×{metrics.reference_size[1]})"],
        ["Candidate", f"{candidate_label}  ({metrics.candidate_size[0]}×{metrics.candidate_size[1]})"],
        ["Mean abs error", f"{metrics.mean_abs_error:.3f} / 255"],
        ["RMSE", f"{metrics.root_mean_square_error:.3f} / 255"],
        ["Changed pixels", f"{metrics.changed_pixels_pct:.2f}%"],
        ["Max channel delta", f"{metrics.max_channel_delta} / 255"],
    ]
    if metrics.resampled:
        rows.append(
            [
                "Note",
                f"Candidate resampled to {metrics.compared_size[0]}×{metrics.compared_size[1]} to "
                "match the reference before comparing — resolutions differed, so this is a "
                "like-for-like check at that size, not a native-detail comparison.",
            ]
        )
    return rows


def refresh_comparison(
    pool: list[ComparisonItem], reference_label: str, candidate_label: str, amplify: float
):
    reference_path = _path_for(pool, reference_label)
    candidate_path = _path_for(pool, candidate_label)
    if not reference_path or not candidate_path:
        return None, [["Status", "Pick a reference and a candidate above to compare."]], None
    try:
        diff_image, metrics = compute_diff(reference_path, candidate_path, amplify=amplify)
    except Exception as exc:
        return None, [["Status", f"Couldn't compare those two: {exc}"]], None
    return (
        (reference_path, candidate_path),
        _metrics_rows(metrics, reference_label, candidate_label),
        diff_image,
    )


def swap_selection(reference_label: str, candidate_label: str):
    return candidate_label, reference_label


def receive_items(new_items: list[ComparisonItem]):
    labels = [item.label for item in new_items]
    default_reference = next((label for label in labels if label.startswith("Input: ")), None) or (
        labels[0] if labels else NO_SELECTION
    )
    default_candidate = next((label for label in labels if label.startswith("Output: ")), None) or (
        labels[-1] if labels else NO_SELECTION
    )
    return (
        new_items,
        gr.update(choices=labels or [NO_SELECTION], value=default_reference),
        gr.update(choices=labels or [NO_SELECTION], value=default_candidate),
        gr.Tabs(selected="compare"),
    )


@dataclass(slots=True)
class CompareTab:
    pool: object
    reference: object
    candidate: object
    swap: object
    diff_amplify: object
    slider: object
    metrics: object
    diff_image: object


def build_compare_tab() -> CompareTab:
    gr.Markdown(
        "Send an image here with the **Send to Comparison** button under a render's output "
        "(Image tab for now). Pick any two of the images that came along with it — an input, "
        "or any rendered output — to diff them against each other."
    )
    pool = gr.State([])
    with gr.Row():
        reference = gr.Dropdown(choices=[NO_SELECTION], value=NO_SELECTION, label="Reference (baseline)")
        candidate = gr.Dropdown(choices=[NO_SELECTION], value=NO_SELECTION, label="Candidate")
        swap = gr.Button("⇄ Swap", scale=0)
    slider = gr.ImageSlider(type="filepath", label="Before / after", height=520)
    with gr.Row():
        with gr.Column(scale=1):
            metrics = gr.Dataframe(
                headers=["Metric", "Value"], datatype=["str", "str"], interactive=False,
                label="Comparison metrics", wrap=True,
            )
        with gr.Column(scale=1):
            diff_amplify = gr.Slider(
                1, 16, value=4, step=1, label="Diff visualization amplify",
                info="Brightens the difference view below so subtle changes are easier to see. Doesn't affect the metrics.",
            )
            diff_image = gr.Image(
                type="pil", interactive=False, label="Difference (amplified, grayscale)", height=360,
            )
    return CompareTab(pool, reference, candidate, swap, diff_amplify, slider, metrics, diff_image)


def bind_comparison_events(compare_tab: CompareTab, tabs, image_tab=None) -> None:
    refresh_inputs = [compare_tab.pool, compare_tab.reference, compare_tab.candidate, compare_tab.diff_amplify]
    refresh_outputs = [compare_tab.slider, compare_tab.metrics, compare_tab.diff_image]
    # Decoding + diffing a large or RAW image takes real time, so these stay on the
    # normal queue (unlike the trivial instant updates elsewhere in this file) rather
    # than running with queue=False, which would block the app while it works.
    refresh_kwargs = dict(show_progress="minimal")
    compare_tab.reference.change(refresh_comparison, inputs=refresh_inputs, outputs=refresh_outputs, **refresh_kwargs)
    compare_tab.candidate.change(refresh_comparison, inputs=refresh_inputs, outputs=refresh_outputs, **refresh_kwargs)
    compare_tab.diff_amplify.release(refresh_comparison, inputs=refresh_inputs, outputs=refresh_outputs, **refresh_kwargs)
    compare_tab.swap.click(
        swap_selection, inputs=[compare_tab.reference, compare_tab.candidate],
        outputs=[compare_tab.reference, compare_tab.candidate], queue=False,
    )
    if image_tab is not None:
        image_tab.send_to_compare.click(
            receive_items, inputs=image_tab.comparison_items,
            outputs=[compare_tab.pool, compare_tab.reference, compare_tab.candidate, tabs],
            queue=False,
        )
