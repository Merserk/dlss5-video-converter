from __future__ import annotations

from pathlib import Path

# Show LOADING immediately before heavy imports (gradio etc.) to avoid black screen after start.bat
import time as _early_time

try:
    from src.core.terminal import TerminalUI as _EarlyTerminalUI

    _early_ui = _EarlyTerminalUI(Path(__file__).with_name("logs"))
    _early_ui.enable_vt_mode()
    _early_time.sleep(0.05)
    _early_ui.render_loading()
    _EARLY_TS = _early_time.time()
except Exception:
    _early_ui = None
    _EARLY_TS = 0.0

import gradio as gr

from src.compare.ui import bind_comparison_events, build_compare_tab
from src.core.paths import LOGS, OUTPUTS
from src.core.runtime import prepare_runtime
from src.core.terminal import init_console
from src.frame_interpolation.ui import build_frame_interpolation_tab
from src.image.decoder import initialize_image_runtime
from src.image.ui import build_image_tab
from src.settings.ui import bind_settings_events, build_settings_tab, initialize_settings
from src.video.ui import build_video_tab

APP_CSS = r"""
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
#video-output-list .file-preview-holder,
#frame-interpolation-upload-list .file-preview-holder,
#frame-interpolation-output-list .file-preview-holder {
    max-height: 210px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
}

#image-upload-list .file-preview,
#image-output-list .file-preview,
#video-upload-list .file-preview,
#video-output-list .file-preview,
#frame-interpolation-upload-list .file-preview,
#frame-interpolation-output-list .file-preview {
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


def build_app() -> gr.Blocks:
    """Build the UI from cached settings and feature-owned tab modules."""
    prepared = prepare_runtime()
    initialize_image_runtime()
    settings, _gpu_warning, ai_gpu_choices, video_gpu_choices = initialize_settings(prepared)

    with gr.Blocks(title="DLSS 5 Visual Enhancer") as demo:
        gr.Markdown(
            "# DLSS 5 Visual Enhancer\n"
            "[Support on Patreon](https://www.patreon.com/MM744) | "
            "[GitHub](https://github.com/Merserk/dlss5-visual-enhancer)",
            elem_id="app-title",
        )
        with gr.Tabs(selected="image") as tabs:
            with gr.Tab("Image", id="image"):
                image_tab = build_image_tab(settings)
            with gr.Tab("Video", id="video"):
                video_tab = build_video_tab(settings)
            with gr.Tab("Frame Interpolation", id="frame-interpolation"):
                frame_tab = build_frame_interpolation_tab(settings)
            with gr.Tab("Comparison", id="compare"):
                compare_tab = build_compare_tab()
            with gr.Tab("Settings", id="settings"):
                settings_tab = build_settings_tab(settings, ai_gpu_choices, video_gpu_choices)

        bind_settings_events(settings_tab, image_tab, video_tab, frame_tab)
        # Video and Frame Interpolation aren't wired into Comparison yet (they need a synced
        # player, not the image before/after slider) — that's a later phase.
        bind_comparison_events(compare_tab, tabs, image_tab=image_tab)
    return demo


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    # Reuse early loading UI if it was already rendered at import time (avoids second flash and keeps alt buffer)
    global _early_ui  # type: ignore
    if "_EARLY_UI" in globals() and _early_ui is not None and getattr(_early_ui, "_alt_active", False):
        ui = _early_ui  # type: ignore
        # Complete init that early block did not do (listener + redirect)
        try:
            ui.start_input_listener()
        except Exception:
            pass
        try:
            ui.silence_and_redirect()
        except Exception:
            pass
        try:
            import atexit

            atexit.register(ui.restore_cursor)
        except Exception:
            pass
    else:
        ui = init_console(LOGS)
    try:
        prepare_runtime()
    except Exception as exc:
        with open(LOGS / "startup_error.log", "a", encoding="utf-8") as f:
            f.write(f"Startup preparation failed: {exc}\n")
        raise SystemExit(1) from exc

    demo = build_app()
    # Replace loading screen with final DLSS 5 Visual Enhancer splash
    try:
        ui.render_screen()
    except Exception:
        pass
    try:
        demo.queue(default_concurrency_limit=1).launch(
            css=APP_CSS,
            theme=gr.themes.Ocean(),
            server_name="127.0.0.1",
            inbrowser=True,
            share=False,
            allowed_paths=[str(OUTPUTS.resolve())],
            show_error=True,
            quiet=True,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
