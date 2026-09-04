from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from ..core.ffmpeg import hdr_mode_supported
from ..core.ffmpeg.preview import normalize_preview_encoding, resolve_final_preview
from ..core.jobs import cancel_active_job
from ..core.naming import RENAME_MODES
from ..core.runtime import DLSS_MODEL_PRESETS, NR_PRESETS, NR_STYLES, UPSCALING_MODES
from ..settings.models import (
    AUTOMATIC_MASK_CHOICES, CODEC_CHOICES, CONTAINER_CHOICES, QUALITY_CHOICES, UISettings,
    automatic_mask_choice, coerce_hdr_mode, parse_automatic_mask,
)
from ..settings.storage import current_preview_encoding, processing_gpu_settings
from .batch import convert_videos
from .models import ConversionOptions
from .preview import (
    _process_video, normalize_video_paths, preview_one_frame, preview_video, update_video_preview_mode,
)


def hdr_mode_update(codec: str):
    allowed = hdr_mode_supported(codec)
    if not allowed:
        return gr.update(value=False, interactive=False)
    return gr.update(interactive=True)


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

def render_video(
    input_path: str,
    nr_preset: str,
    nr_style: str,
    nr_intensity: float,
    local_tone_strength: float,
    local_structure_strength: float,
    skin_structure_strength: float,
    upscaling_factor: float,
    automatic_mask: str,
    dlss_model_preset: str,
    codec: str,
    container: str,
    quality: str,
    hdr_mode: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    return _process_video(
        input_path, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality, hdr_mode,
        progress, None, None
    )

def render_video_batch(
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
    codec: str,
    container: str,
    quality: str,
    hdr_mode: bool,
    rename_mode: str,
    custom_suffix: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[object, list[str], list[list[str]], str]:
    paths = normalize_video_paths(input_paths)
    if not paths:
        raise gr.Error("Choose at least one video first.")
    effective_hdr = coerce_hdr_mode(codec, hdr_mode)
    options = ConversionOptions(
        ai_gpu_uuid=processing_gpu_settings()[0],
        video_gpu_uuid=processing_gpu_settings()[1],
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        codec=codec,
        container=container,
        quality=quality,
        preserve_hdr=effective_hdr,
        rename_mode=rename_mode,
        custom_suffix=custom_suffix,
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_videos(paths, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        return gr.update(value=None, visible=len(paths) == 1), [], [], f"Failed: {exc}"

    ordered_rows: list[tuple[int, list[str]]] = []
    for item in result.successes:
        conversion = item.result
        details = (
            f"{conversion.frames} frames in {conversion.elapsed_seconds:.1f}s; "
            f"DLSS {conversion.dlss_mode}: "
            f"{conversion.render_width}×{conversion.render_height} → "
            f"{conversion.output_width}×{conversion.output_height}; "
            f"report: {conversion.report_path}"
        )
        ordered_rows.append(
            (
                item.index,
                [
                    Path(item.input_path).name,
                    "Complete",
                    Path(conversion.output_path).name,
                    details,
                ],
            )
        )
    for item in result.failures:
        state = "Skipped" if item.error == "Cancelled before rendering." else (
            "Cancelled" if item.cancelled else "Failed"
        )
        ordered_rows.append(
            (item.index, [Path(item.input_path).name, state, "", item.error])
        )
    rows = [row for _index, row in sorted(ordered_rows, key=lambda entry: entry[0])]
    files = [item.result.output_path for item in result.successes]
    try:
        preview_mode = normalize_preview_encoding(current_preview_encoding())
    except Exception:
        preview_mode = "Auto"
    output_preview = None
    used_derivative = False
    if len(paths) == 1 and result.successes:
        candidate = result.successes[0].result.output_path
        output_preview, used_derivative = resolve_final_preview(
            candidate, preview_mode
        )
    failed_count = sum(not item.cancelled for item in result.failures)
    cancelled_count = sum(
        item.cancelled and item.error != "Cancelled before rendering."
        for item in result.failures
    )
    skipped_count = sum(
        item.error == "Cancelled before rendering." for item in result.failures
    )
    state = "Cancelled" if result.cancelled else "Complete"
    status = (
        f"{state}: {len(result.successes)} completed, {failed_count} failed, "
        f"{cancelled_count} cancelled, {skipped_count} skipped. "
        f"Batch manifest: {result.manifest_path}"
    )
    if result.failures:
        status += f"\nFirst error: {result.failures[0].error}"
    if used_derivative:
        status += "\nBrowser preview transcoded to H.264; the original file is unchanged."
    return gr.update(value=output_preview, visible=len(paths) == 1), files, rows, status

@dataclass(slots=True)
class VideoTab:
    sources: object
    input_preview: object
    neural: list[object]
    model_preset: object
    quality: object
    codec: object
    container: object
    rename_mode: object
    custom_suffix: object
    hdr_mode: object
    preview_frame: object
    preview: object
    render: object
    stop: object
    reset: object
    output_video: object
    output_files: object
    status: object
    results: object

    @property
    def render_inputs(self) -> list[object]:
        return [
            self.sources, *self.neural, self.model_preset, self.codec, self.container,
            self.quality, self.hdr_mode, self.rename_mode, self.custom_suffix,
        ]

    @property
    def preview_inputs(self) -> list[object]:
        return [
            self.sources, *self.neural, self.model_preset, self.codec, self.container,
            self.quality, self.hdr_mode,
        ]

    @property
    def settings_inputs(self) -> list[object]:
        return [
            *self.neural, self.model_preset, self.codec, self.container, self.quality,
            self.hdr_mode, self.rename_mode, self.custom_suffix,
        ]


def build_video_tab(settings: UISettings) -> VideoTab:
    # The uploader lives in its own full-width row, above the input/output columns,
    # so both preview panes are the first element in their column and line up
    # vertically regardless of how many files are queued in the uploader.
    sources = gr.File(
        label="Input video(s)", file_count="multiple", file_types=["video"],
        type="filepath", allow_reordering=True, elem_id="video-upload-list",
    )
    with gr.Row():
        with gr.Column(scale=3):
            input_preview = gr.Video(label="Input video preview", interactive=False, visible=False)
            with gr.Accordion("DLSS 5 Neural Rendering Settings", open=True):
                neural = build_neural_controls(settings)
            with gr.Accordion("DLSS 5 Settings", open=True):
                model_preset = build_dlss_model_control(settings)
            quality = gr.Radio(
                QUALITY_CHOICES, value=settings.quality, label="Encoding quality",
                info="Auto uses output resolution, FPS, and codec. Good = Auto×2, Best = Auto×4, Max = CQ 0.",
            )
            with gr.Row():
                codec = gr.Dropdown(
                    CODEC_CHOICES, value=settings.codec, label="Video codec",
                    info="Plain = CPU (libx264 / libx265 / libsvtav1). Suffixed (NVIDIA NVENC) = GPU. ProRes Proxy uses 10-bit 4:2:2 and requires MOV or MKV.",
                )
                container = gr.Dropdown(CONTAINER_CHOICES, value=settings.container, label="Container")
            with gr.Row():
                rename_mode = gr.Radio(
                    RENAME_MODES, value=settings.video_rename_mode, label="Rename",
                    info="Auto adds the current DLSS5 timestamp; Copy keeps the original base name; Custom appends your suffix.",
                )
                custom_suffix = gr.Textbox(
                    value=settings.video_custom_suffix, label="Custom suffix", placeholder="_DLSS5",
                    interactive=settings.video_rename_mode == "Custom",
                )
            hdr_mode = gr.Checkbox(
                value=settings.hdr_mode and hdr_mode_supported(settings.codec),
                label="HDR Mode",
                info="When on: 10-bit output, copies input colorspace; keeps HDR if input is HDR. Only for H.265 / AV1 / ProRes.",
                interactive=hdr_mode_supported(settings.codec),
            )
            with gr.Row():
                preview_frame = gr.Button("Preview 1 frame", visible=False)
                preview = gr.Button("Preview 3 sec", visible=False)
                render = gr.Button("Render video(s)", variant="primary")
                stop = gr.Button("Stop", variant="stop")
                reset = gr.Button("Reset settings")
        with gr.Column(scale=3):
            output_video = gr.Video(label="Output video", interactive=False, visible=False)
            output_files = gr.File(
                label="Rendered video files", file_count="multiple", interactive=False,
                elem_id="video-output-list",
            )
            status = gr.Textbox(label="Status", interactive=False)
            results = gr.Dataframe(
                headers=["Input", "Result", "Output", "Details"],
                datatype=["str", "str", "str", "str"], interactive=False,
                label="Batch results", wrap=True,
            )
    tab = VideoTab(
        sources, input_preview, neural, model_preset, quality, codec, container, rename_mode,
        custom_suffix, hdr_mode, preview_frame, preview, render, stop, reset, output_video,
        output_files, status, results
    )
    bind_video_events(tab)
    return tab


def bind_video_events(tab: VideoTab) -> None:
    tab.sources.change(
        update_video_preview_mode, inputs=tab.sources,
        outputs=[tab.input_preview, tab.output_video, tab.preview_frame, tab.preview],
        queue=False, show_progress="hidden",
    )
    tab.rename_mode.change(rename_suffix_update, inputs=tab.rename_mode, outputs=tab.custom_suffix, queue=False)
    tab.codec.change(hdr_mode_update, inputs=tab.codec, outputs=tab.hdr_mode, queue=False)
    tab.render.click(
        render_video_batch, inputs=tab.render_inputs,
        outputs=[tab.output_video, tab.output_files, tab.results, tab.status],
        concurrency_limit=1, show_progress="full", show_progress_on=tab.output_video,
    )
    tab.preview.click(
        preview_video, inputs=tab.preview_inputs, outputs=[tab.output_video, tab.status],
        concurrency_limit=1, show_progress="full", show_progress_on=tab.output_video,
    )
    tab.preview_frame.click(
        preview_one_frame, inputs=tab.preview_inputs, outputs=[tab.output_video, tab.status],
        concurrency_limit=1, show_progress="full", show_progress_on=tab.output_video,
    )
    tab.stop.click(cancel_active_job, outputs=tab.status, queue=False)

