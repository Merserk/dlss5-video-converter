from __future__ import annotations

import json
import math
import os
import queue
import shutil
import threading
import time
from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable

import av
import cv2
import numpy as np

from . import ffmpeg
from .naming import output_filename, require_available_output, validate_rename
from .runtime import (
    detect_gpu,
    list_gpus,
    validate_gpu_runtime,
    JOBS,
    LOGS,
    OUTPUTS,
    Cancelled,
    DLSSFrameSession,
    active_job,
    resize_fit,
    rotate_frame,
    verify_feature_18,
    write_failure_report,
)


ENCODING_QUALITIES = ffmpeg.ENCODING_QUALITIES
AUTO_BITRATE_DIVISORS = ffmpeg.AUTO_BITRATE_DIVISORS
calculate_auto_bitrate_kbps = ffmpeg.calculate_auto_bitrate_kbps
probe_video = ffmpeg.probe_video
validate_codec_container = ffmpeg.validate_codec_container


@dataclass(slots=True)
class ConversionOptions:
    nr_style: str = "Default"
    nr_intensity: float = 1.0
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    upscaling_factor: float = 1.0
    codec: str = "H.264"
    container: str = "MP4"
    quality: str = "Auto (Default)"
    preserve_hdr: bool = False
    warmup_frames: int = 0
    preview_seconds: float | None = None
    preview_frames: int | None = None
    nr_preset: str = "Default"
    automatic_mask: bool = False
    rename_mode: str = "Auto"
    custom_suffix: str = "_DLSS5"
    dlss_model_preset: str = "Default"
    dual_gpu_encode: bool = True


MIN_ENCODE_GPU_FREE_MB = 2048

NR_PRESETS = {
    "Default": 0,
    "Preset #1": 1,
    "Preset #2": 2,
    "Preset #3": 3,
}

NR_STYLES = {
    "Default": 0,
    "Natural": 1,
    "Cinematic": 2,
}

DLSS_MODEL_PRESETS = {
    "Default": 0,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
}

UPSCALING_MODES = {
    1.0: {"label": "1× (DLAA / native)", "name": "DLAA", "perf_quality": 5},
    1.5: {"label": "1.5× (Quality)", "name": "Quality", "perf_quality": 2},
    1.724: {"label": "1.724× (Balanced)", "name": "Balanced", "perf_quality": 1},
    2.0: {"label": "2× (Performance)", "name": "Performance", "perf_quality": 0},
    3.0: {
        "label": "3× (Ultra Performance)",
        "name": "Ultra Performance",
        "perf_quality": 3,
    },
}
UPSCALING_CHOICES = tuple(
    (mode["label"], factor) for factor, mode in UPSCALING_MODES.items()
)


def resolve_upscaling_mode(raw_factor: float) -> tuple[float, dict[str, str | int]]:
    try:
        factor = float(raw_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Upscaling factor must be one of the supported NVIDIA DLSS modes."
        ) from exc
    if not math.isfinite(factor):
        raise ValueError("Upscaling factor must be one of the supported NVIDIA DLSS modes.")
    for supported, mode in UPSCALING_MODES.items():
        if math.isclose(factor, supported, rel_tol=0.0, abs_tol=1e-9):
            return supported, mode
    choices = ", ".join(f"{factor:g}×" for factor in UPSCALING_MODES)
    raise ValueError(f"Unsupported upscaling factor {factor:g}×. Choose one of: {choices}.")


def _nearest_even(value: float) -> int:
    return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)


def resolve_output_size(width: int, height: int, factor: float) -> tuple[int, int]:
    factor, _ = resolve_upscaling_mode(factor)
    output_width = _nearest_even(int(width) * factor)
    output_height = _nearest_even(int(height) * factor)
    long_edge = max(output_width, output_height)
    short_edge = min(output_width, output_height)
    if long_edge > 7680 or short_edge > 4320:
        valid = [
            candidate
            for candidate in UPSCALING_MODES
            if max(_nearest_even(width * candidate), _nearest_even(height * candidate)) <= 7680
                       and min(_nearest_even(width * candidate), _nearest_even(height * candidate))
            <= 4320
        ]
        recommendation = max(valid) if valid else None
        hint = (
            f" Choose {recommendation:g}× or lower for this video."
            if recommendation is not None
            else " The source already exceeds the supported 8K boundary."
        )
        raise ValueError(
            f"The requested {output_width}×{output_height} output exceeds the supported "
            f"7680×4320 boundary.{hint}"
        )
    return output_width, output_height


def resolve_encoding_quality(
    options: ConversionOptions, width: int, height: int, fps: float
) -> dict:
    return ffmpeg.resolve_encoding_quality(
        options.quality, options.codec, width, height, fps
    )


def resolve_native_settings(options: ConversionOptions) -> dict[str, int | float]:
    """Validate public NR controls and translate them to the worker protocol."""
    try:
        preset = NR_PRESETS[options.nr_preset]
    except KeyError as exc:
        choices = ", ".join(NR_PRESETS)
        raise ValueError(
            f"Unknown NR Preset: {options.nr_preset!r}. Choose one of: {choices}."
        ) from exc

    try:
        style = NR_STYLES[options.nr_style]
    except KeyError as exc:
        choices = ", ".join(NR_STYLES)
        raise ValueError(
            f"Unknown NR Style: {options.nr_style!r}. Choose one of: {choices}."
        ) from exc

    try:
        model_preset = DLSS_MODEL_PRESETS[options.dlss_model_preset]
    except KeyError as exc:
        choices = ", ".join(DLSS_MODEL_PRESETS)
        raise ValueError(
            f"Unknown DLSS Model Preset: {options.dlss_model_preset!r}. "
            f"Choose one of: {choices}."
        ) from exc

    controls = {
        "NR Intensity": (options.nr_intensity, 0.0, 2.0),
        "Local Tone Strength": (options.local_tone_strength, 0.0, 2.0),
        "Local Structure Strength": (options.local_structure_strength, 0.0, 2.0),
        "Skin Structure Strength": (options.skin_structure_strength, -1.0, 2.0),
    }
    validated: dict[str, float] = {}
    for label, (raw_value, minimum, maximum) in controls.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must be a number between {minimum:g} and {maximum:g}."
            ) from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
        validated[label] = value

    if not isinstance(options.automatic_mask, bool):
        raise ValueError("Automatic Mask must be a boolean value.")

    return {
        "profile": 0,
        "preset": preset,
        "style": style,
        "auto_mask": int(options.automatic_mask),
        "ui_correction": 0,
        "intensity": validated["NR Intensity"],
        "local_tone": validated["Local Tone Strength"],
        "local_structure": validated["Local Structure Strength"],
        "skin_structure": validated["Skin Structure Strength"],
        "dlss_model_preset": model_preset,
    }


@dataclass(slots=True)
class ConversionResult:
    output_path: str
    report_path: str
    frames: int
    nr_count_evidence: int
    elapsed_seconds: float
    gpu: str
    input_width: int
    input_height: int
    render_width: int
    render_height: int
    output_width: int
    output_height: int
    upscaling_factor: float
    dlss_mode: str
    dlss_model_preset: str = "Default"
    applied_dlss_model_preset: int = 0


@dataclass(slots=True)
class VideoConversionSuccess:
    index: int
    input_path: str
    result: ConversionResult


@dataclass(slots=True)
class VideoConversionFailure:
    index: int
    input_path: str
    error: str
    cancelled: bool = False


@dataclass(slots=True)
class VideoBatchResult:
    successes: list[VideoConversionSuccess]
    failures: list[VideoConversionFailure]
    cancelled: bool
    manifest_path: str


_BATCH_CONTEXT = threading.local()


@dataclass(slots=True)
class GuideFrame:
    motion: np.ndarray
    reset: bool
    scene_score: float


class TemporalGuideGenerator:
    """Estimate the guide buffers an encoded video does not contain."""

    def __init__(self, width: int, height: int, flow_width: int = 640) -> None:
        self.width = width
        self.height = height
        scale = min(1.0, flow_width / width)
        self.flow_width = max(64, int(round(width * scale / 2) * 2))
        self.flow_height = max(64, int(round(height * scale / 2) * 2))
        self.previous_gray: np.ndarray | None = None
        self.zero_motion = np.zeros((height, width, 2), dtype=np.float16)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.dis.setUseSpatialPropagation(True)
        self.dis.setFinestScale(1)

    def _small_gray(self, rgba: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        return cv2.resize(
            gray,
            (self.flow_width, self.flow_height),
            interpolation=cv2.INTER_AREA,
        )

    def process(self, rgba: np.ndarray) -> GuideFrame:
        current = self._small_gray(rgba)
        if self.previous_gray is None:
            motion = self.zero_motion
            reset = True
            scene_score = 1.0
        else:
            scene_score = float(np.mean(cv2.absdiff(current, self.previous_gray))) / 255.0
            reset = scene_score > 0.24
            if reset:
                motion = self.zero_motion
            else:
                motion = self.dis.calc(current, self.previous_gray, None)
                motion = cv2.resize(
                    motion,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                motion[..., 0] *= self.width / self.flow_width
                motion[..., 1] *= self.height / self.flow_height
                motion = np.ascontiguousarray(motion.astype(np.float16))
        self.previous_gray = current
        return GuideFrame(
            motion=motion,
            reset=reset,
            scene_score=scene_score,
        )


def _validate_preview_options(
    options: ConversionOptions,
) -> tuple[float | None, int | None]:
    preview_seconds: float | None = None
    if options.preview_seconds is not None:
        try:
            preview_seconds = float(options.preview_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("Preview duration must be a positive number of seconds.") from exc
        if not math.isfinite(preview_seconds) or preview_seconds <= 0:
            raise ValueError("Preview duration must be a positive number of seconds.")

    preview_frames: int | None = None
    if options.preview_frames is not None:
        if isinstance(options.preview_frames, bool):
            raise ValueError("Preview frame count must be a positive integer.")
        try:
            preview_frames = int(options.preview_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("Preview frame count must be a positive integer.") from exc
        if preview_frames <= 0 or preview_frames != options.preview_frames:
            raise ValueError("Preview frame count must be a positive integer.")
    if preview_seconds is not None and preview_frames is not None:
        raise ValueError("Choose either a timed preview or a frame preview, not both.")
    return preview_seconds, preview_frames


def convert_video(
    input_path: str | os.PathLike[str],
    options: ConversionOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> ConversionResult:
    from .prepare import prepare_runtime

    options = options or ConversionOptions()
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    validate_codec_container(options.codec, options.container)
    validate_rename(options.rename_mode, options.custom_suffix)
    preview_seconds, preview_frames = _validate_preview_options(options)
    is_preview = preview_seconds is not None or preview_frames is not None
    if options.preserve_hdr:
        raise RuntimeError(
            "HDR preservation is disabled in this build because the verified DLSSNR path is "
            "RGBA8. HDR input is converted to SDR instead of being mislabeled as HDR."
        )
    prepared_runtime = getattr(_BATCH_CONTEXT, "prepared_runtime", None)
    if prepared_runtime is None:
        prepared_runtime = prepare_runtime()
    batch_controller = getattr(_BATCH_CONTEXT, "controller", None)
    job_context = nullcontext(batch_controller) if batch_controller is not None else active_job()

    with job_context as controller:
        assert controller is not None
        started = time.perf_counter()
        timings: dict[str, float] = {}
        job_dir: Path | None = None
        output: Path | None = None
        session: DLSSFrameSession | None = None
        encoder_failure_logs: list[str] = []
        # Detect fresh instead of reusing the prepared GPU: the user's GPU
        # selection can change between renders, and the session later corrects
        # this to whichever adapter the worker really bound.
        gpu: dict | None = detect_gpu()
        runtime_bundle: dict | None = prepared_runtime.runtime_bundle
        validate_gpu_runtime(gpu, runtime_bundle)
        encoder = None
        encoder_setup_thread: threading.Thread | None = None
        producer_thread: threading.Thread | None = None
        sender_thread: threading.Thread | None = None
        writer_thread: threading.Thread | None = None
        pipeline_stop = threading.Event()
        pipeline_errors: queue.Queue[BaseException] = queue.Queue(maxsize=4)

        def record_pipeline_error(exc: BaseException) -> None:
            pipeline_stop.set()
            try:
                pipeline_errors.put_nowait(exc)
            except queue.Full:
                pass

        try:
            stage_started = time.perf_counter()
            metadata = ffmpeg.probe_video(source, count_mode="metadata")
            if preview_frames is not None:
                known_frames = int(metadata["frames"])
                frame_count = min(known_frames, preview_frames) if known_frames else preview_frames
            elif preview_seconds is not None:
                frame_count = ffmpeg.preview_frame_count(source, preview_seconds)
            else:
                frame_count = int(metadata["frames"])
                if frame_count <= 0:
                    exact = ffmpeg.probe_video(source, count_mode="exact")
                    frame_count = int(exact["frames"])
                    metadata["frames"] = frame_count
                    metadata["frame_count_source"] = exact["frame_count_source"]
            timings["probe_seconds"] = time.perf_counter() - stage_started
            input_width = int(metadata["width"])
            input_height = int(metadata["height"])
            factor, mode = resolve_upscaling_mode(options.upscaling_factor)
            output_width, output_height = resolve_output_size(
                input_width, input_height, factor
            )
            OUTPUTS.mkdir(exist_ok=True)
            LOGS.mkdir(exist_ok=True)
            JOBS.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
            job_dir = JOBS / f"{source.stem}-{stamp}-{os.getpid()}"
            job_dir.mkdir(parents=True, exist_ok=False)
            extension = {"MP4": ".mp4", "MKV": ".mkv", "MOV": ".mov"}.get(
                options.container
            )
            if extension is None:
                raise ValueError(f"Unknown output container: {options.container!r}.")
            output_kind = (
                "DLSS5"
                if not is_preview
                else (
                    "DLSS5_PREVIEW_FRAME"
                    if preview_frames is not None
                    else "DLSS5_PREVIEW"
                )
            )
            output = OUTPUTS / output_filename(
                source,
                extension,
                options.rename_mode,
                options.custom_suffix,
                f"{source.stem}_{output_kind}_{stamp}",
            )
            require_available_output(output)
            temp_video = job_dir / "processed-video.mkv"
            native = resolve_native_settings(options)
            if progress:
                progress(0.01, "Starting feature 18 (the native worker picks the adapter)")

            encoder_setup: list[tuple] = []
            encoding_stage_started = time.perf_counter()
            encode_gpu: dict | None = None

            def prepare_encoder() -> None:
                encoder_started = time.perf_counter()
                try:
                    encoder_setup.append(
                        ffmpeg.start_encoder(
                            temp_video,
                            options.codec,
                            options.quality,
                            controller,
                            output_width,
                            output_height,
                            float(metadata["fps"]),
                            encode_gpu=encode_gpu,
                        )
                    )
                except BaseException as exc:
                    record_pipeline_error(exc)
                finally:
                    timings["encoder_setup_seconds"] = (
                        time.perf_counter() - encoder_started
                    )

            # With two cards the encoder must wait for the worker handshake to
            # learn the render adapter, so NVENC can run on the other card.
            defer_encoder = options.dual_gpu_encode and len(list_gpus()) > 1
            if not defer_encoder:
                encoder_setup_thread = threading.Thread(
                    target=prepare_encoder, name="dlss5-encoder-setup", daemon=True
                )
                encoder_setup_thread.start()
            session_started = time.perf_counter()
            session = DLSSFrameSession(
                input_width=input_width,
                input_height=input_height,
                output_width=output_width,
                output_height=output_height,
                frame_count=frame_count,
                warmup_frames=options.warmup_frames,
                factor=factor,
                mode=mode,
                native_settings=native,
                gpu=gpu,
                runtime_bundle=runtime_bundle,
                controller=controller,
            )
            timings["native_setup_seconds"] = time.perf_counter() - session_started
            if defer_encoder:
                bound = str(gpu.get("bound_adapter") or gpu["name"]).casefold()
                encode_gpu = next(
                    (
                        entry
                        for entry in list_gpus()
                        if entry["name"].casefold() != bound
                    ),
                    None,
                )
                if encode_gpu is not None and (
                    int(encode_gpu.get("memory_free_mb", 0)) < MIN_ENCODE_GPU_FREE_MB
                ):
                    # An NVENC session that cannot allocate dies mid-encode and
                    # takes the whole render with it; stay on the render GPU.
                    if progress:
                        progress(
                            0.02,
                            f"Dual GPU skipped: {encode_gpu['name']} has only "
                            f"{encode_gpu.get('memory_free_mb', 0)} MB VRAM free "
                            f"(needs {MIN_ENCODE_GPU_FREE_MB} MB); encoding on "
                            f"{gpu['display_name']}",
                        )
                    encode_gpu = None
                elif encode_gpu is not None and progress:
                    progress(
                        0.02,
                        f"Dual GPU: rendering on {gpu['display_name']}, "
                        f"encoding on {encode_gpu['name']}",
                    )
                prepare_encoder()
            else:
                encoder_setup_thread.join()
            encoder_setup_thread = None
            timings["setup_seconds"] = max(
                timings["native_setup_seconds"],
                timings.get("encoder_setup_seconds", 0.0),
            )
            if not pipeline_errors.empty():
                raise pipeline_errors.get_nowait()
            if not encoder_setup:
                raise RuntimeError("Video encoder did not finish preparing.")
            render_width = session.render_width
            render_height = session.render_height
            setup_result = session.setup_result
            minimum_width = session.minimum_width
            minimum_height = session.minimum_height
            maximum_width = session.maximum_width
            maximum_height = session.maximum_height
            if progress:
                progress(
                    0.03,
                    f"DLSS {mode['name']}: {render_width}×{render_height} → "
                    f"{output_width}×{output_height}",
                )

            (
                encoder,
                encoder_log_thread,
                encoder_logs,
                selected_encoder,
                encoding_quality,
            ) = encoder_setup[0]
            encoder_failure_logs = encoder_logs
            assert encoder.stdin is not None

            prepared_bytes = render_width * render_height * 8
            rendered_bytes = output_width * output_height * 4
            queue_slots = max(
                1,
                min(3, (384 * 1024 * 1024) // max(1, prepared_bytes + rendered_bytes)),
            )
            prepared_frames: queue.Queue[object] = queue.Queue(maxsize=queue_slots)
            rendered_frames: queue.Queue[object] = queue.Queue(maxsize=queue_slots)
            stop_marker = object()
            producer_stats: dict[str, float | int] = {}
            writer_stats: dict[str, float | int] = {}

            def put_pipeline(target: queue.Queue[object], item: object) -> bool:
                while not pipeline_stop.is_set():
                    if controller.cancel.is_set():
                        return False
                    try:
                        target.put(item, timeout=0.1)
                        return True
                    except queue.Full:
                        continue
                return False

            def produce_frames() -> None:
                producer_started = time.perf_counter()
                decoded = 0
                container = None
                try:
                    container = av.open(str(source))
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    guides = TemporalGuideGenerator(render_width, render_height)
                    for index, frame in enumerate(container.decode(stream)):
                        if index >= frame_count or pipeline_stop.is_set():
                            break
                        if controller.cancel.is_set():
                            raise Cancelled("Render stopped by user.")
                        rgba = rotate_frame(
                            frame.to_ndarray(format="rgba"), metadata["rotation"]
                        )
                        if rgba.shape[1] != render_width or rgba.shape[0] != render_height:
                            rgba = resize_fit(rgba, render_width, render_height)
                        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
                        guide = guides.process(rgba)
                        pts = int(frame.pts if frame.pts is not None else index)
                        if not put_pipeline(
                            prepared_frames, (index, rgba, guide, pts)
                        ):
                            return
                        decoded += 1
                    producer_stats["decoded_frames"] = decoded
                    put_pipeline(prepared_frames, stop_marker)
                except BaseException as exc:
                    record_pipeline_error(exc)
                    put_pipeline(prepared_frames, stop_marker)
                finally:
                    if container is not None:
                        with suppress(Exception):
                            container.close()
                    producer_stats["seconds"] = time.perf_counter() - producer_started

            def write_frames() -> None:
                writer_started = time.perf_counter()
                written = 0
                nut = None
                try:
                    nut = av.open(encoder.stdin, mode="w", format="nut")
                    raw_stream = nut.add_stream("rawvideo", rate=metadata["rate"])
                    raw_stream.width = output_width
                    raw_stream.height = output_height
                    raw_stream.pix_fmt = "rgba"
                    raw_stream.time_base = metadata["time_base"]
                    # Keep the source's fine time base in the encoder context
                    # too. The default (1/rate) quantizes VFR timestamps to
                    # frame-number ticks, and two close frames then collide
                    # into one tick, which the NUT muxer rejects as
                    # non-monotonic dts.
                    raw_stream.codec_context.time_base = metadata["time_base"]
                    last_mux_pts: int | None = None

                    def mux_packet(packet) -> None:
                        """Surface encoder death instead of a bare mux EINVAL."""
                        try:
                            nut.mux(packet)
                        except Exception as mux_exc:
                            encoder_code = encoder.poll()
                            if encoder_code is None:
                                raise
                            raise RuntimeError(
                                f"The video encoder ({selected_encoder}) exited "
                                f"with code {encoder_code} during frame "
                                f"{written + 1}/{frame_count}:\n"
                                + (
                                    "\n".join(encoder_logs[-40:])
                                    or "It produced no output."
                                )
                            ) from mux_exc

                    while not pipeline_stop.is_set():
                        if controller.cancel.is_set():
                            raise Cancelled("Render stopped by user.")
                        try:
                            item = rendered_frames.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if item is stop_marker:
                            break
                        processed, output_pts = item
                        output_frame = av.VideoFrame.from_ndarray(processed, format="rgba")
                        output_frame.pts = output_pts
                        output_frame.time_base = metadata["time_base"]
                        if last_mux_pts is not None and output_frame.pts <= last_mux_pts:
                            output_frame.pts = last_mux_pts + 1
                            writer_stats["pts_collisions"] = (
                                int(writer_stats.get("pts_collisions", 0)) + 1
                            )
                        last_mux_pts = output_frame.pts
                        for packet in raw_stream.encode(output_frame):
                            mux_packet(packet)
                        written += 1
                    if not pipeline_stop.is_set():
                        for packet in raw_stream.encode():
                            mux_packet(packet)
                        nut.close()
                        nut = None
                    writer_stats["written_frames"] = written
                except BaseException as exc:
                    record_pipeline_error(exc)
                finally:
                    if nut is not None:
                        with suppress(Exception):
                            nut.close()
                    writer_stats["seconds"] = time.perf_counter() - writer_started

            sent_frames: queue.Queue[object] = queue.Queue(maxsize=2)

            def send_frames() -> None:
                # A dedicated sender lets the main thread keep draining the
                # worker's stdout; sending and receiving from one thread would
                # deadlock on full pipes once a frame is in flight.
                try:
                    while not pipeline_stop.is_set():
                        if controller.cancel.is_set():
                            raise Cancelled("Render stopped by user.")
                        try:
                            item = prepared_frames.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if item is stop_marker:
                            put_pipeline(sent_frames, stop_marker)
                            return
                        index, rgba, guide, pts = item
                        session.send_frame(
                            index=index,
                            rgba=rgba,
                            motion=guide.motion,
                            reset=guide.reset,
                            pts=pts,
                        )
                        if not put_pipeline(sent_frames, (index, guide.reset)):
                            return
                except BaseException as exc:
                    record_pipeline_error(exc)
                    put_pipeline(sent_frames, stop_marker)

            producer_thread = threading.Thread(
                target=produce_frames, name="dlss5-video-producer", daemon=True
            )
            sender_thread = threading.Thread(
                target=send_frames, name="dlss5-worker-sender", daemon=True
            )
            writer_thread = threading.Thread(
                target=write_frames, name="dlss5-video-writer", daemon=True
            )
            producer_thread.start()
            sender_thread.start()
            writer_thread.start()
            delivered = 0
            scene_resets = 0
            preview_pts_origin: int | None = None
            dlss_seconds = 0.0
            last_progress_update = 0.0
            while True:
                if controller.cancel.is_set():
                    raise Cancelled("Render stopped by user.")
                if not pipeline_errors.empty():
                    raise pipeline_errors.get_nowait()
                try:
                    item = sent_frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is stop_marker:
                    break
                index, was_reset = item
                scene_resets += int(was_reset and index != 0)
                dlss_started = time.perf_counter()
                processed, out_pts = session.receive_frame(index)
                dlss_seconds += time.perf_counter() - dlss_started
                if is_preview:
                    if preview_pts_origin is None:
                        preview_pts_origin = out_pts
                    out_pts -= preview_pts_origin
                if not put_pipeline(rendered_frames, (processed, out_pts)):
                    if not pipeline_errors.empty():
                        raise pipeline_errors.get_nowait()
                    raise Cancelled("Render stopped by user.")
                delivered += 1
                now = time.perf_counter()
                if progress and (delivered == frame_count or now - last_progress_update >= 0.1):
                    progress(
                        0.04 + 0.84 * delivered / frame_count,
                        f"DLSS 5 frame {delivered}/{frame_count}",
                    )
                    last_progress_update = now

            if delivered != frame_count:
                raise RuntimeError(
                    f"Decoded {delivered} frames instead of the expected {frame_count}; "
                    "refusing an incomplete render."
                )
            if not put_pipeline(rendered_frames, stop_marker):
                if not pipeline_errors.empty():
                    raise pipeline_errors.get_nowait()
                raise Cancelled("Render stopped by user.")
            producer_thread.join()
            producer_thread = None
            sender_thread.join()
            sender_thread = None
            writer_thread.join()
            writer_thread = None
            if not pipeline_errors.empty():
                raise pipeline_errors.get_nowait()
            timings["producer_seconds"] = float(producer_stats.get("seconds", 0.0))
            timings["decode_and_guide_seconds"] = timings["producer_seconds"]
            timings["dlss_seconds"] = dlss_seconds
            timings["encoder_feed_seconds"] = float(writer_stats.get("seconds", 0.0))
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            session.close()
            encoder_code = encoder.wait(timeout=120)
            encoder_log_thread.join(timeout=2)
            controller.unregister(encoder)
            timings["encoding_seconds"] = time.perf_counter() - encoding_stage_started
            if encoder_code:
                raise RuntimeError(
                    "Video encoder failed:\n" + "\n".join(encoder_logs[-40:])
                )

            feature_evidence = verify_feature_18(
                session.worker_logs, session.reshade_log_text()
            )
            nr_count = delivered
            nr_upscaling_requested = factor > 1.0
            nr_upscaling_active = bool(feature_evidence["nr_upscaling_active"])
            nr_native_fallback = bool(feature_evidence["nr_native_fallback"])
            carrier_create_result = str(feature_evidence["carrier_create_result"])
            if progress:
                progress(0.91, "Muxing original audio and metadata")
            mux_started = time.perf_counter()
            ffmpeg.final_mux(temp_video, source, output, options.container, controller)
            timings["final_mux_seconds"] = time.perf_counter() - mux_started
            timings["muxing_seconds"] = timings["final_mux_seconds"]
            verify_started = time.perf_counter()
            verified = ffmpeg.probe_video(output, count_mode="packets")
            if verified["frames"] != delivered:
                verified = ffmpeg.probe_video(output, count_mode="exact")
            if verified["frames"] != delivered:
                raise RuntimeError(
                    f"Output verification found {verified['frames']} frames instead of "
                    f"{delivered}."
                )
            if (verified["width"], verified["height"]) != (
                output_width,
                output_height,
            ):
                raise RuntimeError(
                    f"Output verification found {verified['width']}×{verified['height']} "
                    f"instead of {output_width}×{output_height}."
                )
            timings["verification_seconds"] = time.perf_counter() - verify_started

            elapsed = time.perf_counter() - started
            report = {
                "status": "success",
                "input": str(source),
                "output": str(output),
                "options": asdict(options),
                "input_metadata": {
                    key: str(value) if isinstance(value, Fraction) else value
                    for key, value in metadata.items()
                },
                "output_metadata": {
                    key: str(value) if isinstance(value, Fraction) else value
                    for key, value in verified.items()
                },
                "gpu": gpu,
                "encoder": selected_encoder,
                "encoding_quality": encoding_quality,
                "frames_processed": delivered,
                "render_mode": (
                    "full"
                    if not is_preview
                    else ("preview-frame" if preview_frames is not None else "preview")
                ),
                "dlss_mode": mode["name"],
                "dlss_model_preset": options.dlss_model_preset,
                "requested_dlss_model_preset": options.dlss_model_preset,
                "requested_dlss_model_preset_code": native["dlss_model_preset"],
                "applied_dlss_model_preset": session.applied_dlss_model_preset,
                "applied_dlss_model_preset_name": next(
                    name
                    for name, code in DLSS_MODEL_PRESETS.items()
                    if code == session.applied_dlss_model_preset
                ),
                "requested_upscaling_factor": factor,
                "input_dimensions": {"width": input_width, "height": input_height},
                "negotiated_render_dimensions": {
                    "width": render_width,
                    "height": render_height,
                },
                "negotiated_render_range": {
                    "minimum": {"width": minimum_width, "height": minimum_height},
                    "maximum": {"width": maximum_width, "height": maximum_height},
                },
                "output_dimensions": {"width": output_width, "height": output_height},
                "effective_factor": {
                    "width": output_width / input_width,
                    "height": output_height / input_height,
                },
                "nr_upscaling_requested": nr_upscaling_requested,
                "nr_upscaling_active": nr_upscaling_active,
                "nr_native_fallback": nr_native_fallback,
                "ngx_setup_result": f"0x{setup_result:08X}",
                "scene_resets": scene_resets,
                "pts_collisions_adjusted": int(writer_stats.get("pts_collisions", 0)),
                "pipeline": "renodx-dlssnr-feature18",
                "feature_id": 18,
                "feature_18_confirmed": True,
                "carrier_create_result": carrier_create_result,
                "successful_neural_rendering_frames": nr_count,
                "addon_release": runtime_bundle["addon"]["release"],
                "addon_sha256": runtime_bundle["addon"]["sha256"].lower(),
                "model_sha256": runtime_bundle["neural_runtime"]["sha256"].lower(),
                "worker_sha256": runtime_bundle["worker"]["sha256"].lower(),
                "loaded_module_inventory": [
                    "nvngx.dll (standalone worker image)",
                    "dxgi.dll (ReShade carrier)",
                    "renodx-dlss5.addon64",
                    "nvngx_dlss.dll",
                    "nvngx_dlssnr.dll",
                    "system D3D12/DXGI/NGX core",
                ],
                "native_settings": native,
                "elapsed_seconds": elapsed,
                "average_fps": delivered / elapsed,
                "timings": timings,
                "worker_log": session.worker_logs,
                "worker_log_dropped_lines": session.worker_log_dropped_lines,
                "encoder_log": encoder_logs,
                "dlssnr_evidence": feature_evidence["evidence"],
            }
            report_path = LOGS / f"{output.name}.report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if progress:
                progress(1.0, "Complete — feature 18 confirmed")
            return ConversionResult(
                str(output),
                str(report_path),
                delivered,
                nr_count,
                elapsed,
                gpu["display_name"],
                input_width,
                input_height,
                render_width,
                render_height,
                output_width,
                output_height,
                factor,
                str(mode["name"]),
                options.dlss_model_preset,
                session.applied_dlss_model_preset,
            )
        except Exception as exc:
            was_cancelled = controller.cancel.is_set()
            pipeline_stop.set()
            if controller.cancel.is_set():
                controller.stop()
            else:
                controller.terminate_processes()
            for target in (prepared_frames if "prepared_frames" in locals() else None,
                           rendered_frames if "rendered_frames" in locals() else None):
                if target is not None:
                    with suppress(queue.Full):
                        target.put_nowait(stop_marker)
            for thread in (
                encoder_setup_thread,
                producer_thread,
                sender_thread,
                writer_thread,
            ):
                if thread is not None:
                    thread.join(timeout=2)
            if session is not None and not session.closed:
                with suppress(Exception):
                    session.abort()
            if encoder is not None and encoder.stdin and not encoder.stdin.closed:
                with suppress(OSError):
                    encoder.stdin.close()
            if output and output.exists():
                output.unlink()
            if was_cancelled and not isinstance(exc, Cancelled):
                raise Cancelled("Render stopped by user.") from exc
            if isinstance(exc, Cancelled):
                raise
            worker_logs = session.worker_logs if session is not None else []
            reshade_lines = session.reshade_diagnostics() if session is not None else []
            worker_code = session.worker.poll() if session is not None else None
            report_path = write_failure_report(
                operation="video-render",
                source=str(source),
                error=exc,
                gpu=gpu,
                runtime_bundle=runtime_bundle,
                worker_code=worker_code,
                worker_logs=worker_logs,
                reshade_lines=reshade_lines,
                encoder_logs=list(encoder_failure_logs),
            )
            raise RuntimeError(f"{exc}\nDiagnostic report: {report_path}") from exc
        finally:
            pipeline_stop.set()
            if job_dir and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)


def _write_video_batch_manifest(
    stamp: str,
    options: ConversionOptions,
    successes: list[VideoConversionSuccess],
    failures: list[VideoConversionFailure],
    cancelled: bool,
) -> str:
    LOGS.mkdir(exist_ok=True)
    manifest = {
        "status": "cancelled" if cancelled else ("partial" if failures else "success"),
        "options": asdict(options),
        "successes": [asdict(item) for item in successes],
        "failures": [asdict(item) for item in failures],
    }
    manifest_path = LOGS / f"DLSS5_VIDEO_BATCH_{stamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(manifest_path)


def convert_videos(
    input_paths: Iterable[str | os.PathLike[str]],
    options: ConversionOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> VideoBatchResult:
    """Convert videos sequentially while holding one cancellable GPU batch slot."""
    from .prepare import prepare_runtime

    options = options or ConversionOptions()
    paths = [Path(path).resolve() for path in input_paths]
    if not paths:
        raise ValueError("Choose at least one video.")

    prepared_runtime = prepare_runtime()
    LOGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    successes: list[VideoConversionSuccess] = []
    failures: list[VideoConversionFailure] = []
    cancelled = False
    total = len(paths)

    with active_job() as controller:
        if getattr(_BATCH_CONTEXT, "controller", None) is not None:
            raise RuntimeError("A video batch is already active on this worker thread.")
        _BATCH_CONTEXT.controller = controller
        _BATCH_CONTEXT.prepared_runtime = prepared_runtime
        try:
            for index, path in enumerate(paths):
                position = index + 1
                prefix = f"[{position}/{total}] {path.name}"
                if controller.cancel.is_set():
                    cancelled = True
                    failures.extend(
                        VideoConversionFailure(
                            queued_index,
                            str(queued),
                            "Cancelled before rendering.",
                            cancelled=True,
                        )
                        for queued_index, queued in enumerate(paths[index:], start=index)
                    )
                    break

                def report_item(
                    value: float,
                    message: str,
                    *,
                    item_index: int = index,
                    item_prefix: str = prefix,
                ) -> None:
                    bounded = min(1.0, max(0.0, float(value)))
                    overall = (item_index + bounded) / total
                    if progress:
                        progress(overall, f"{item_prefix} — {message}")

                if progress:
                    progress(index / total, f"{prefix} — starting")
                try:
                    result = convert_video(path, options, progress=report_item)
                except Cancelled:
                    cancelled = True
                    failures.append(
                        VideoConversionFailure(
                            index,
                            str(path),
                            "Cancelled during rendering.",
                            cancelled=True,
                        )
                    )
                    failures.extend(
                        VideoConversionFailure(
                            queued_index,
                            str(queued),
                            "Cancelled before rendering.",
                            cancelled=True,
                        )
                        for queued_index, queued in enumerate(paths[position:], start=position)
                    )
                    break
                except Exception as exc:
                    failures.append(VideoConversionFailure(index, str(path), str(exc)))
                    if progress:
                        progress(position / total, f"{prefix} — failed")
                    continue

                successes.append(VideoConversionSuccess(index, str(path), result))
                if progress:
                    progress(position / total, f"{prefix} — complete")
        finally:
            del _BATCH_CONTEXT.controller
            del _BATCH_CONTEXT.prepared_runtime

    manifest_path = _write_video_batch_manifest(
        stamp, options, successes, failures, cancelled
    )
    if progress:
        progress(1.0, "Cancelled" if cancelled else "Complete")
    return VideoBatchResult(successes, failures, cancelled, manifest_path)
