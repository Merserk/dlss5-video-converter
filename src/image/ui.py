from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from PIL import Image

from ..core.jobs import cancel_active_job
from ..core.naming import RENAME_MODES
from ..core.runtime import DLSS_MODEL_PRESETS, NR_PRESETS, NR_STYLES, UPSCALING_MODES
from ..settings.models import AUTOMATIC_MASK_CHOICES, UISettings, automatic_mask_choice, parse_automatic_mask
from ..settings.storage import processing_gpu_settings
from ..compare.models import ComparisonItem
from .decoder import decode_image
from .encoder import take_image_preview
from .batch import convert_images
from .models import IMAGE_FORMATS, RAW_EXTENSIONS, ImageConversionOptions


def rename_suffix_update(mode: str):
    return gr.update(interactive=mode == "Custom")


UPSCALING_CHOICES = tuple((mode["label"], factor) for factor, mode in UPSCALING_MODES.items())


def build_neural_controls(settings: UISettings):
    nr_preset = gr.Dropdown(
        list(NR_PRESETS), value=settings.nr_preset, label="NR Preset",
        info="Experimental content-dependent neural-rendering model hint. Default is recommended."
    )
    nr_style = gr.Radio(
        list(NR_STYLES), value=settings.nr_style, label="NR Style",
        info="Selects the native neural-rendering style."
    )
    upscaling_factor = gr.Dropdown(
        choices=list(UPSCALING_CHOICES), value=settings.upscaling_factor,
        label="Upscaling factor", info="Uses NVIDIA's fixed DLSS modes. DLAA keeps source resolution."
    )
    with gr.Row():
        nr_intensity = gr.Slider(
            0.0, 2.0, value=settings.nr_intensity, step=0.05, precision=2,
            label="NR Intensity", info="Overall neural-rendering strength.", buttons=["reset"]
        )
        local_tone_strength = gr.Slider(
            0.0, 2.0, value=settings.local_tone_strength, step=0.05, precision=2,
            label="Local Tone Strength", info="Local tone and contrast enhancement.", buttons=["reset"]
        )
    with gr.Row():
        local_structure_strength = gr.Slider(
            0.0, 2.0, value=settings.local_structure_strength, step=0.05, precision=2,
            label="Local Structure Strength", info="Local detail and texture structure.", buttons=["reset"]
        )
        skin_structure_strength = gr.Slider(
            -1.0, 2.0, value=settings.skin_structure_strength, step=0.05, precision=2,
            label="Skin Structure Strength", info="Skin-specific structure; -1.00 is the native default.",
            buttons=["reset"]
        )
    automatic_mask = gr.Radio(
        choices=AUTOMATIC_MASK_CHOICES,
        value=automatic_mask_choice(settings.automatic_mask),
        label="Automatic Mask",
        info=(
            "Experimental runtime-generated mask that changes where Neural Rendering is "
            "applied; it may cause flicker or inconsistent results."
        ),
    )
    return [
        nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask
    ]


def build_dlss_model_control(settings: UISettings):
    return gr.Dropdown(
        choices=list(DLSS_MODEL_PRESETS),
        value=settings.dlss_model_preset,
        label="DLSS Model Preset",
        info=(
            "Default lets NVIDIA select its normal mode-specific presets. "
            "J, K, L, or M forces that model preset for every DLSS scaling mode."
        ),
    )

def preview_input_images(paths: list[str] | str | None):
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    previews = []
    for raw_path in paths:
        try:
            decoded = decode_image(raw_path)
            image = Image.fromarray(decoded.rgba, mode="RGBA")
            image.thumbnail((1200, 900), Image.Resampling.LANCZOS)
            previews.append((image, Path(raw_path).name))
        except Exception:
            continue
    return previews


def render_image_batch(
    input_paths: list[str] | str | None,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    image_format: str,
    image_quality: float,
    rename_mode: str,
    custom_suffix: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not input_paths:
        raise gr.Error("Choose at least one image first.")
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    options = ImageConversionOptions(
        ai_gpu_uuid=processing_gpu_settings()[0],
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        output_format=image_format,
        quality=int(image_quality),
        rename_mode=rename_mode,
        custom_suffix=custom_suffix,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_images(input_paths, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        return [], [], None, [], f"Failed: {exc}", []

    gallery = []
    comparison_items = [
        ComparisonItem(f"Input: {Path(path).name}", path) for path in input_paths
    ]
    for item in result.successes:
        preview = take_image_preview(item.output_path)
        if preview is None:
            with Image.open(item.output_path) as output:
                preview = output.convert("RGBA")
                preview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
                preview = preview.copy()
        gallery.append((preview, Path(item.output_path).name))
        # Full-resolution output path, not the thumbnail above — comparisons need the real file.
        comparison_items.append(ComparisonItem(f"Output: {Path(item.output_path).name}", item.output_path))
    files = [item.output_path for item in result.successes]
    rows = [
        [Path(item.input_path).name, "Complete", Path(item.output_path).name, "; ".join(item.warnings)]
        for item in result.successes
    ]
    rows.extend(
        [Path(item.input_path).name, "Failed", "", item.error] for item in result.failures
    )
    state = "Cancelled" if result.cancelled else "Complete"
    status = (
        f"{state}: {len(result.successes)} image(s) rendered, {len(result.failures)} failed. "
        "Every successful output returned feature-18 success and has a diagnostic report."
    )
    if result.failures:
        status += f"\nFirst error: {result.failures[0].error}"
    return gallery, files, result.zip_path, rows, status, comparison_items

@dataclass(slots=True)
class ImageTab:
    sources: object
    input_gallery: object
    neural: list[object]
    model_preset: object
    output_format: object
    quality: object
    rename_mode: object
    custom_suffix: object
    render: object
    stop: object
    reset: object
    output_gallery: object
    send_to_compare: object
    comparison_items: object
    output_files: object
    zip_download: object
    status: object
    results: object

    @property
    def render_inputs(self) -> list[object]:
        return [self.sources, *self.neural, self.model_preset, self.output_format, self.quality, self.rename_mode, self.custom_suffix]

    @property
    def settings_inputs(self) -> list[object]:
        return [*self.neural, self.model_preset, self.output_format, self.quality, self.rename_mode, self.custom_suffix]


def build_image_tab(settings: UISettings) -> ImageTab:
    upload_types = ["image", ".svg", ".heic", ".heif", *sorted(RAW_EXTENSIONS)]
    # The uploader lives in its own full-width row, above the input/output columns,
    # so both preview panes are the first element in their column and line up
    # vertically regardless of how many files are queued in the uploader.
    sources = gr.File(
        label="Input image(s)", file_count="multiple", file_types=upload_types,
        type="filepath", allow_reordering=True, elem_id="image-upload-list",
    )
    with gr.Row():
        with gr.Column(scale=3):
            input_gallery = gr.Gallery(
                label="Input preview", columns=3, height=520, object_fit="contain",
                interactive=False, buttons=["fullscreen"], elem_id="image-input-preview",
            )
            with gr.Accordion("DLSS 5 Neural Rendering Settings", open=True):
                neural = build_neural_controls(settings)
            with gr.Accordion("DLSS 5 Settings", open=True):
                model_preset = build_dlss_model_control(settings)
            with gr.Row():
                output_format = gr.Dropdown(
                    list(IMAGE_FORMATS), value=settings.image_format, label="Output format",
                    info="PNG and TIFF are lossless; JPEG composites transparency over white.",
                )
                quality = gr.Slider(
                    1, 100, value=settings.image_quality, step=1, precision=0,
                    label="Lossy quality", info="Used by JPEG, WebP, and AVIF; ignored by PNG/TIFF.",
                )
            with gr.Row():
                rename_mode = gr.Radio(
                    RENAME_MODES, value=settings.image_rename_mode, label="Rename",
                    info="Auto adds the current DLSS5 timestamp; Copy keeps the original base name; Custom appends your suffix.",
                )
                custom_suffix = gr.Textbox(
                    value=settings.image_custom_suffix, label="Custom suffix", placeholder="_DLSS5",
                    interactive=settings.image_rename_mode == "Custom",
                )
            with gr.Row():
                render = gr.Button("Render image(s)", variant="primary")
                stop = gr.Button("Stop", variant="stop")
                reset = gr.Button("Reset settings")
        with gr.Column(scale=3):
            output_gallery = gr.Gallery(
                label="Enhanced previews", columns=2, height=520, object_fit="contain",
                interactive=False, buttons=["download", "download_all", "fullscreen"],
                elem_id="image-output-preview",
            )
            # Holds the real (non-thumbnail) input/output paths from the last render,
            # so "Send to Comparison" always hands off full-resolution files.
            comparison_items = gr.State([])
            send_to_compare = gr.Button("Send to Comparison")
            output_files = gr.File(
                label="Rendered image files", file_count="multiple", interactive=False,
                elem_id="image-output-list",
            )
            zip_download = gr.DownloadButton("Download successful images as ZIP")
            status = gr.Textbox(label="Status", interactive=False)
            results = gr.Dataframe(
                headers=["Input", "Result", "Output", "Details"],
                datatype=["str", "str", "str", "str"], interactive=False,
                label="Batch results", wrap=True,
            )
    tab = ImageTab(
        sources, input_gallery, neural, model_preset, output_format, quality, rename_mode,
        custom_suffix, render, stop, reset, output_gallery, send_to_compare, comparison_items,
        output_files, zip_download, status, results
    )
    bind_image_events(tab)
    return tab


def bind_image_events(tab: ImageTab) -> None:
    tab.sources.change(
        preview_input_images, inputs=tab.sources, outputs=tab.input_gallery,
        queue=False, show_progress="hidden",
    )
    tab.rename_mode.change(
        rename_suffix_update, inputs=tab.rename_mode, outputs=tab.custom_suffix, queue=False,
    )
    tab.render.click(
        render_image_batch, inputs=tab.render_inputs,
        outputs=[
            tab.output_gallery, tab.output_files, tab.zip_download, tab.results, tab.status,
            tab.comparison_items,
        ],
        concurrency_limit=1, show_progress="full", show_progress_on=tab.output_gallery,
    )
    tab.stop.click(cancel_active_job, outputs=tab.status, queue=False)

