from __future__ import annotations

import threading
import traceback
import sys
from dataclasses import replace
from pathlib import Path

import gradio as gr
from PIL import Image

from src.images import (
    IMAGE_FORMATS,
    RAW_EXTENSIONS,
    ImageConversionOptions,
    convert_images,
    decode_image,
    take_image_preview,
)
from src.naming import RENAME_MODES
from src.runtime import (
    AUTO_GPU,
    LOGS,
    OUTPUTS,
    cancel_active_job,
    gpu_choices,
    set_preferred_gpu,
)
from src.prepare import prepare_runtime
from src.settings import (
    CODEC_CHOICES,
    CONTAINER_CHOICES,
    DEFAULT_SETTINGS,
    QUALITY_CHOICES,
    UISettings,
    load_settings,
    save_settings,
)
from src.video import (
    ConversionOptions,
    DLSS_MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    UPSCALING_CHOICES,
    convert_video,
    convert_videos,
)


CONFIG_PATH = Path(__file__).resolve().with_name("config.ini")
PREVIEW_SECONDS = 3.0
AUTOMATIC_MASK_CHOICES = ("Off", "On")
_CONFIG_LOCK = threading.Lock()
_CURRENT_SETTINGS: UISettings | None = None

APP_CSS = """
/* Keep the header links identical even when one URL has been visited. */
#app-title a,
#app-title a:visited,
#app-title a:hover,
#app-title a:active {
    color: #00bfff !important;
    opacity: 1 !important;
}

/* Keep large upload batches compact: roughly three file rows, then scroll. */
#image-upload-list .file-preview-holder,
#image-output-list .file-preview-holder,
#video-upload-list .file-preview-holder,
#video-output-list .file-preview-holder {
    max-height: 210px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
}

#image-upload-list .file-preview,
#image-output-list .file-preview,
#video-upload-list .file-preview,
#video-output-list .file-preview {
    max-height: none !important;
}

/* A single input or output preview is a full 16:9 viewport without scrolling. */
#image-input-preview:has(.gallery-item:only-child),
#image-output-preview:has(.gallery-item:only-child) {
    aspect-ratio: 16 / 9;
    height: auto !important;
    min-height: 0 !important;
}

#image-input-preview:has(.gallery-item:only-child) .gallery-container,
#image-input-preview:has(.gallery-item:only-child) .grid-wrap,
#image-input-preview:has(.gallery-item:only-child) .grid-container,
#image-input-preview:has(.gallery-item:only-child) .gallery-item,
#image-input-preview:has(.gallery-item:only-child) .thumbnail-lg,
#image-output-preview:has(.gallery-item:only-child) .gallery-container,
#image-output-preview:has(.gallery-item:only-child) .grid-wrap,
#image-output-preview:has(.gallery-item:only-child) .grid-container,
#image-output-preview:has(.gallery-item:only-child) .gallery-item,
#image-output-preview:has(.gallery-item:only-child) .thumbnail-lg {
    box-sizing: border-box;
    height: 100% !important;
    min-height: 0 !important;
}

#image-input-preview:has(.gallery-item:only-child) .grid-wrap,
#image-output-preview:has(.gallery-item:only-child) .grid-wrap {
    overflow: hidden !important;
}

#image-input-preview:has(.gallery-item:only-child) .grid-container,
#image-output-preview:has(.gallery-item:only-child) .grid-container {
    grid-template-rows: minmax(0, 1fr) !important;
    grid-auto-rows: minmax(0, 1fr) !important;
}

#image-input-preview:has(.gallery-item:only-child) img,
#image-output-preview:has(.gallery-item:only-child) img {
    height: 100% !important;
    width: 100% !important;
    object-fit: contain !important;
}
"""


def _automatic_mask_choice(enabled: bool) -> str:
    return "On" if enabled else "Off"


def _parse_automatic_mask(value: str) -> bool:
    if value not in AUTOMATIC_MASK_CHOICES:
        choices = ", ".join(AUTOMATIC_MASK_CHOICES)
        raise ValueError(f"Automatic Mask must be one of: {choices}.")
    return value == "On"


def rename_suffix_update(mode: str):
    return gr.update(interactive=mode == "Custom")


def _neural_values(
    settings: UISettings,
) -> tuple[str, str, float, float, float, float, float, str]:
    return (
        settings.nr_preset,
        settings.nr_style,
        settings.nr_intensity,
        settings.local_tone_strength,
        settings.local_structure_strength,
        settings.skin_structure_strength,
        settings.upscaling_factor,
        _automatic_mask_choice(settings.automatic_mask),
    )


def _shared_dlss_values(
    settings: UISettings,
) -> tuple[str, str, float, float, float, float, float, str, str]:
    return (*_neural_values(settings), settings.dlss_model_preset)


def persist_image_settings(
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
) -> tuple[str, str, float, float, float, float, float, str, str]:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(
            current,
            nr_preset=nr_preset,
            nr_style=nr_style,
            nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            upscaling_factor=upscaling_factor,
            automatic_mask=_parse_automatic_mask(automatic_mask),
            dlss_model_preset=dlss_model_preset,
            image_format=image_format,
            image_quality=int(image_quality),
            image_rename_mode=rename_mode,
            image_custom_suffix=custom_suffix,
        )
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    return _shared_dlss_values(settings)


def persist_video_settings(
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
    rename_mode: str,
    custom_suffix: str,
) -> tuple[str, str, float, float, float, float, float, str, str]:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(
            current,
            nr_preset=nr_preset,
            nr_style=nr_style,
            nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            upscaling_factor=upscaling_factor,
            automatic_mask=_parse_automatic_mask(automatic_mask),
            dlss_model_preset=dlss_model_preset,
            codec=codec,
            container=container,
            quality=quality,
            video_rename_mode=rename_mode,
            video_custom_suffix=custom_suffix,
        )
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    return _shared_dlss_values(settings)


def _gpu_dropdown_state(selection: str) -> tuple[list[str], str]:
    """Current GPU choices, falling back to automatic when the saved GPU is gone."""
    choices = gpu_choices()
    return choices, selection if selection in choices else AUTO_GPU


def build_gpu_control(settings: UISettings):
    choices, value = _gpu_dropdown_state(settings.gpu)
    return gr.Dropdown(
        choices=choices,
        value=value,
        label="GPU",
        info=(
            "Which detected RTX GPU renders. Auto uses the first one. The render "
            "fails fast if the native worker cannot bind the selected card."
        ),
    )


def persist_gpu(gpu: str):
    """Apply and save the GPU choice, then mirror it onto the other tab."""
    global _CURRENT_SETTINGS
    choices, value = _gpu_dropdown_state(gpu)
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(current, gpu=value)
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    set_preferred_gpu(value)
    return gr.update(choices=choices, value=value)


def refresh_gpus(gpu: str):
    """Re-enumerate GPUs so a newly available card can be picked without a restart."""
    return persist_gpu(gpu)


def persist_dual_gpu_encode(enabled: bool) -> bool:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        current = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
        settings = replace(current, dual_gpu_encode=bool(enabled))
        if settings != current:
            save_settings(CONFIG_PATH, settings)
        _CURRENT_SETTINGS = settings
    return bool(enabled)


def _saved_dual_gpu_encode() -> bool:
    with _CONFIG_LOCK:
        settings = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
    return settings.dual_gpu_encode


def reset_saved_settings() -> tuple:
    global _CURRENT_SETTINGS
    with _CONFIG_LOCK:
        save_settings(CONFIG_PATH, DEFAULT_SETTINGS)
        _CURRENT_SETTINGS = DEFAULT_SETTINGS
    set_preferred_gpu(DEFAULT_SETTINGS.gpu)
    gpu_choices_list, gpu_value = _gpu_dropdown_state(DEFAULT_SETTINGS.gpu)
    shared = _shared_dlss_values(DEFAULT_SETTINGS)
    message = "All Image and Video settings were reset to defaults."
    return (
        *shared,
        *shared,
        DEFAULT_SETTINGS.image_format,
        DEFAULT_SETTINGS.image_quality,
        DEFAULT_SETTINGS.image_rename_mode,
        gr.update(value=DEFAULT_SETTINGS.image_custom_suffix, interactive=False),
        DEFAULT_SETTINGS.codec,
        DEFAULT_SETTINGS.container,
        DEFAULT_SETTINGS.quality,
        DEFAULT_SETTINGS.video_rename_mode,
        gr.update(value=DEFAULT_SETTINGS.video_custom_suffix, interactive=False),
        gr.update(choices=gpu_choices_list, value=gpu_value),
        gr.update(choices=gpu_choices_list, value=gpu_value),
        DEFAULT_SETTINGS.dual_gpu_encode,
        message,
        message,
    )


def _process_video(
    input_path: str | None,
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
    progress,
    preview_seconds: float | None,
    preview_frames: int | None,
) -> tuple[str | None, str]:
    if not input_path:
        raise gr.Error("Choose a video first.")
    is_preview = preview_seconds is not None or preview_frames is not None
    options = ConversionOptions(
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=_parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        codec="H.264" if is_preview else codec,
        container="MP4" if is_preview else container,
        quality=quality,
        preview_seconds=preview_seconds,
        preview_frames=preview_frames,
        dual_gpu_encode=_saved_dual_gpu_encode(),
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_video(input_path, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc
    output_preview = result.output_path if options.container == "MP4" else None
    source_name = Path(input_path).name
    if preview_frames is not None:
        return output_preview, (
            f"One-frame preview complete for {source_name} on {result.gpu} "
            f"in {result.elapsed_seconds:.1f}s. "
            f"DLSS {result.dlss_mode}: {result.render_width}×{result.render_height} → "
            f"{result.output_width}×{result.output_height}. Signed feature 18 confirmed."
        )
    if is_preview:
        return output_preview, (
            f"Preview complete for {source_name}: {result.frames} frames from the first "
            f"{PREVIEW_SECONDS:g} seconds processed "
            f"on {result.gpu} in {result.elapsed_seconds:.1f}s. DLSS {result.dlss_mode}: "
            f"{result.render_width}×{result.render_height} → {result.output_width}×{result.output_height}. "
            "All frames returned success with signed feature 18 confirmed."
        )
    status = (
        f"Complete: {result.frames} frames processed on {result.gpu} in {result.elapsed_seconds:.1f}s. "
        f"All {result.nr_count_evidence} frames returned success with signed feature 18 confirmed. "
        f"DLSS {result.dlss_mode}: {result.render_width}×{result.render_height} → "
        f"{result.output_width}×{result.output_height}."
    )
    if container != "MP4":
        status += f" {container} output was created successfully, but browser preview is unavailable."
    return output_preview, status


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
    progress=gr.Progress(track_tqdm=False),
):
    return _process_video(
        input_path, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality,
        progress, None, None
    )


def _normalize_video_paths(paths: list[str] | str | None) -> list[str]:
    if not paths:
        return []
    return [paths] if isinstance(paths, str) else list(paths)


def first_video_path(paths: list[str] | str | None) -> str | None:
    normalized = _normalize_video_paths(paths)
    return normalized[0] if normalized else None


def update_video_preview_mode(paths: list[str] | str | None):
    normalized = _normalize_video_paths(paths)
    single = len(normalized) == 1
    input_value = normalized[0] if single else None
    return (
        gr.update(value=input_value, visible=single),
        gr.update(value=None, visible=single),
        gr.update(visible=single),
        gr.update(visible=single),
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
    rename_mode: str,
    custom_suffix: str,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[object, list[str], list[list[str]], str]:
    paths = _normalize_video_paths(input_paths)
    if not paths:
        raise gr.Error("Choose at least one video first.")
    options = ConversionOptions(
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=_parse_automatic_mask(automatic_mask),
        dlss_model_preset=dlss_model_preset,
        upscaling_factor=upscaling_factor,
        codec=codec,
        container=container,
        quality=quality,
        rename_mode=rename_mode,
        custom_suffix=custom_suffix,
        dual_gpu_encode=_saved_dual_gpu_encode(),
    )

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        result = convert_videos(paths, options, progress=report)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc)) from exc

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
    output_preview = None
    if len(paths) == 1 and result.successes:
        candidate = result.successes[0].result.output_path
        if Path(candidate).suffix.lower() == ".mp4":
            output_preview = candidate
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
    return gr.update(value=output_preview, visible=len(paths) == 1), files, rows, status


def preview_video(
    input_path: list[str] | str | None,
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
    progress=gr.Progress(track_tqdm=False),
):
    selected = first_video_path(input_path)
    return _process_video(
        selected, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality,
        progress, PREVIEW_SECONDS, None
    )


def preview_one_frame(
    input_path: list[str] | str | None,
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
    progress=gr.Progress(track_tqdm=False),
):
    selected = first_video_path(input_path)
    return _process_video(
        selected, nr_preset, nr_style, nr_intensity, local_tone_strength, local_structure_strength,
        skin_structure_strength, upscaling_factor, automatic_mask, dlss_model_preset,
        codec, container, quality,
        progress, None, 1
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
        nr_preset=nr_preset,
        nr_style=nr_style,
        nr_intensity=nr_intensity,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        skin_structure_strength=skin_structure_strength,
        automatic_mask=_parse_automatic_mask(automatic_mask),
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
        raise gr.Error(str(exc)) from exc

    gallery = []
    for item in result.successes:
        preview = take_image_preview(item.output_path)
        if preview is None:
            with Image.open(item.output_path) as output:
                preview = output.convert("RGBA")
                preview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
                preview = preview.copy()
        gallery.append((preview, Path(item.output_path).name))
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
    return gallery, files, result.zip_path, rows, status


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
        value=_automatic_mask_choice(settings.automatic_mask),
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


def build_app() -> gr.Blocks:
    """Build the UI from the cached settings without rewriting configuration."""
    global _CURRENT_SETTINGS
    settings = _CURRENT_SETTINGS or load_settings(CONFIG_PATH)
    _CURRENT_SETTINGS = settings
    set_preferred_gpu(settings.gpu)
    upload_types = ["image", ".svg", ".heic", ".heif", *sorted(RAW_EXTENSIONS)]

    with gr.Blocks(title="DLSS 5 Visual Enhancer") as demo:
        gr.Markdown(
            "# DLSS 5 Visual Enhancer\n"
            "[Support on Patreon](https://www.patreon.com/MM744) | "
            "[GitHub](https://github.com/Merserk/dlss5-visual-enhancer)",
            elem_id="app-title",
        )
        with gr.Tabs(selected="image"):
            with gr.Tab("Image", id="image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_sources = gr.File(
                            label="Input image(s)",
                            file_count="multiple",
                            file_types=upload_types,
                            type="filepath",
                            allow_reordering=True,
                            elem_id="image-upload-list",
                        )
                        image_input_gallery = gr.Gallery(
                            label="Input preview",
                            columns=3,
                            height=320,
                            object_fit="contain",
                            interactive=False,
                            buttons=["fullscreen"],
                            elem_id="image-input-preview",
                        )
                        with gr.Accordion(
                            "DLSS 5 Neural Rendering Settings", open=True
                        ):
                            image_neural = build_neural_controls(settings)
                        with gr.Accordion("DLSS 5 Settings", open=True):
                            image_model_preset = build_dlss_model_control(settings)
                        with gr.Row():
                            image_format = gr.Dropdown(
                                list(IMAGE_FORMATS),
                                value=settings.image_format,
                                label="Output format",
                                info=(
                                    "PNG and TIFF are lossless; JPEG composites transparency "
                                    "over white."
                                ),
                            )
                            image_quality = gr.Slider(
                                1,
                                100,
                                value=settings.image_quality,
                                step=1,
                                precision=0,
                                label="Lossy quality",
                                info=(
                                    "Used by JPEG, WebP, and AVIF; ignored by PNG/TIFF."
                                ),
                            )
                        with gr.Row():
                            image_rename_mode = gr.Radio(
                                RENAME_MODES,
                                value=settings.image_rename_mode,
                                label="Rename",
                                info=(
                                    "Auto adds the current DLSS5 timestamp; Copy keeps the "
                                    "original base name; Custom appends your suffix."
                                ),
                            )
                            image_custom_suffix = gr.Textbox(
                                value=settings.image_custom_suffix,
                                label="Custom suffix",
                                placeholder="_DLSS5",
                                interactive=settings.image_rename_mode == "Custom",
                            )
                        with gr.Row():
                            image_gpu = build_gpu_control(settings)
                            image_gpu_refresh = gr.Button("Refresh GPUs")
                        gr.Markdown(
                            "Images are processed as 8-bit SDR sRGB. Animated and multipage "
                            "files use their first frame/page."
                        )
                        with gr.Row():
                            image_render = gr.Button("Render image(s)", variant="primary")
                            image_stop = gr.Button("Stop", variant="stop")
                            image_reset = gr.Button("Reset settings")
                        image_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=3):
                        image_output_gallery = gr.Gallery(
                            label="Enhanced previews",
                            columns=2,
                            height=520,
                            object_fit="contain",
                            interactive=False,
                            buttons=["download", "download_all", "fullscreen"],
                            elem_id="image-output-preview",
                        )
                        image_output_files = gr.File(
                            label="Rendered image files",
                            file_count="multiple",
                            interactive=False,
                            elem_id="image-output-list",
                        )
                        image_zip = gr.DownloadButton(
                            "Download successful images as ZIP"
                        )
                        image_results = gr.Dataframe(
                            headers=["Input", "Result", "Output", "Details"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Batch results",
                            wrap=True,
                        )

            with gr.Tab("Video", id="video"):
                with gr.Row():
                    with gr.Column(scale=3):
                        video_sources = gr.File(
                            label="Input video(s)",
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                            allow_reordering=True,
                            elem_id="video-upload-list",
                        )
                        video_input_preview = gr.Video(
                            label="Input video preview",
                            interactive=False,
                            visible=False,
                        )
                        with gr.Accordion(
                            "DLSS 5 Neural Rendering Settings", open=True
                        ):
                            video_neural = build_neural_controls(settings)
                        with gr.Accordion("DLSS 5 Settings", open=True):
                            video_model_preset = build_dlss_model_control(settings)
                        video_quality = gr.Radio(
                            QUALITY_CHOICES,
                            value=settings.quality,
                            label="Encoding quality",
                            info=(
                                "Auto uses output resolution, FPS, and codec. Good = Auto×2, "
                                "Best = Auto×4, Max = CQ 0."
                            ),
                        )
                        with gr.Row():
                            video_codec = gr.Dropdown(
                                CODEC_CHOICES,
                                value=settings.codec,
                                label="Video codec",
                                info=(
                                    "ProRes Proxy uses 10-bit 4:2:2 and requires MOV or MKV."
                                ),
                            )
                            video_container = gr.Dropdown(
                                CONTAINER_CHOICES,
                                value=settings.container,
                                label="Container",
                            )
                        with gr.Row():
                            video_rename_mode = gr.Radio(
                                RENAME_MODES,
                                value=settings.video_rename_mode,
                                label="Rename",
                                info=(
                                    "Auto adds the current DLSS5 timestamp; Copy keeps the "
                                    "original base name; Custom appends your suffix."
                                ),
                            )
                            video_custom_suffix = gr.Textbox(
                                value=settings.video_custom_suffix,
                                label="Custom suffix",
                                placeholder="_DLSS5",
                                interactive=settings.video_rename_mode == "Custom",
                            )
                        with gr.Row():
                            video_gpu = build_gpu_control(settings)
                            video_gpu_refresh = gr.Button("Refresh GPUs")
                        video_dual_gpu = gr.Checkbox(
                            value=settings.dual_gpu_encode,
                            label="Use both GPUs (encode on the other GPU)",
                            info=(
                                "DLSS renders on the GPU above; NVENC encoding runs on the "
                                "other detected RTX card. Ignored with a single GPU."
                            ),
                        )
                        gr.Checkbox(
                            value=False,
                            interactive=False,
                            label=(
                                "Preserve HDR (disabled: verified feature-18 path is RGBA8; "
                                "HDR safely outputs as SDR)"
                            ),
                        )
                        with gr.Row():
                            video_preview_frame = gr.Button(
                                "Preview 1 frame", visible=False
                            )
                            video_preview = gr.Button("Preview 3 sec", visible=False)
                            video_render = gr.Button("Render video(s)", variant="primary")
                            video_stop = gr.Button("Stop", variant="stop")
                            video_reset = gr.Button("Reset settings")
                        video_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=3):
                        output_video = gr.Video(
                            label="Output video",
                            interactive=False,
                            visible=False,
                        )
                        video_output_files = gr.File(
                            label="Rendered video files",
                            file_count="multiple",
                            interactive=False,
                            elem_id="video-output-list",
                        )
                        video_results = gr.Dataframe(
                            headers=["Input", "Result", "Output", "Details"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            label="Batch results",
                            wrap=True,
                        )

        image_inputs = [
            image_sources,
            *image_neural,
            image_model_preset,
            image_format,
            image_quality,
            image_rename_mode,
            image_custom_suffix,
        ]
        video_inputs = [
            video_sources,
            *video_neural,
            video_model_preset,
            video_codec,
            video_container,
            video_quality,
            video_rename_mode,
            video_custom_suffix,
        ]
        video_preview_inputs = [
            video_sources,
            *video_neural,
            video_model_preset,
            video_codec,
            video_container,
            video_quality,
        ]
        image_settings_inputs = [
            *image_neural,
            image_model_preset,
            image_format,
            image_quality,
            image_rename_mode,
            image_custom_suffix,
        ]
        video_settings_inputs = [
            *video_neural,
            video_model_preset,
            video_codec,
            video_container,
            video_quality,
            video_rename_mode,
            video_custom_suffix,
        ]

        image_sources.change(
            preview_input_images,
            inputs=image_sources,
            outputs=image_input_gallery,
            queue=False,
            show_progress="hidden",
        )
        video_sources.change(
            update_video_preview_mode,
            inputs=video_sources,
            outputs=[
                video_input_preview,
                output_video,
                video_preview_frame,
                video_preview,
            ],
            queue=False,
            show_progress="hidden",
        )
        image_rename_mode.change(
            rename_suffix_update,
            inputs=image_rename_mode,
            outputs=image_custom_suffix,
            queue=False,
        )
        video_rename_mode.change(
            rename_suffix_update,
            inputs=video_rename_mode,
            outputs=video_custom_suffix,
            queue=False,
        )
        for component in image_settings_inputs:
            component.input(
                persist_image_settings,
                inputs=image_settings_inputs,
                outputs=[*video_neural, video_model_preset],
                queue=False,
            )
        for component in video_settings_inputs:
            component.input(
                persist_video_settings,
                inputs=video_settings_inputs,
                outputs=[*image_neural, image_model_preset],
                queue=False,
            )

        image_gpu.input(persist_gpu, inputs=image_gpu, outputs=video_gpu, queue=False)
        video_gpu.input(persist_gpu, inputs=video_gpu, outputs=image_gpu, queue=False)
        image_gpu_refresh.click(
            refresh_gpus, inputs=image_gpu, outputs=image_gpu, queue=False
        )
        video_gpu_refresh.click(
            refresh_gpus, inputs=video_gpu, outputs=video_gpu, queue=False
        )
        video_dual_gpu.input(
            persist_dual_gpu_encode,
            inputs=video_dual_gpu,
            outputs=video_dual_gpu,
            queue=False,
        )

        image_render.click(
            render_image_batch,
            inputs=image_inputs,
            outputs=[
                image_output_gallery,
                image_output_files,
                image_zip,
                image_results,
                image_status,
            ],
            concurrency_limit=1,
        )
        image_stop.click(cancel_active_job, outputs=image_status, queue=False)

        video_render.click(
            render_video_batch,
            inputs=video_inputs,
            outputs=[output_video, video_output_files, video_results, video_status],
            concurrency_limit=1,
        )
        video_preview.click(
            preview_video,
            inputs=video_preview_inputs,
            outputs=[output_video, video_status],
            concurrency_limit=1,
        )
        video_preview_frame.click(
            preview_one_frame,
            inputs=video_preview_inputs,
            outputs=[output_video, video_status],
            concurrency_limit=1,
        )
        video_stop.click(cancel_active_job, outputs=video_status, queue=False)

        reset_outputs = [
            *image_neural,
            image_model_preset,
            *video_neural,
            video_model_preset,
            image_format,
            image_quality,
            image_rename_mode,
            image_custom_suffix,
            video_codec,
            video_container,
            video_quality,
            video_rename_mode,
            video_custom_suffix,
            image_gpu,
            video_gpu,
            video_dual_gpu,
            image_status,
            video_status,
        ]
        image_reset.click(reset_saved_settings, outputs=reset_outputs, queue=False)
        video_reset.click(reset_saved_settings, outputs=reset_outputs, queue=False)
    return demo


def main() -> None:
    print("Preparing DLSS, GPU, image, and FFmpeg runtime before launching the UI...", flush=True)
    try:
        prepared = prepare_runtime()
    except Exception as exc:
        print(f"Startup preparation failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    print(
        f"Runtime ready on {prepared.gpu['display_name']}; launching Gradio.",
        flush=True,
    )
    OUTPUTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        css=APP_CSS,
        theme=gr.themes.Ocean(),
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        allowed_paths=[str(OUTPUTS.resolve())],
        show_error=True,
    )


if __name__ == "__main__":
    main()
