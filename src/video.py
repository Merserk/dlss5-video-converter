from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np

from . import ffmpeg
from .runtime import (
    JOBS,
    LOGS,
    OUTPUTS,
    RUNTIME,
    WORKER,
    Cancelled,
    DLSSFrameSession,
    active_job,
    detect_gpu,
    inspect_runtime_bundle,
    list_gpus,
    resize_fit,
    rotate_frame,
    validate_gpu_runtime,
    validate_runtime_files,
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
    warmup_frames: int = 120
    preview_seconds: float | None = None
    preview_frames: int | None = None
    nr_preset: str = "Default"
    automatic_mask: bool = False
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
        pixels = self.width * self.height
        if self.previous_gray is None:
            motion = np.zeros((self.height, self.width, 2), dtype=np.float32)
            reset = True
            scene_score = 1.0
        else:
            scene_score = float(np.mean(cv2.absdiff(current, self.previous_gray))) / 255.0
            reset = scene_score > 0.24
            if reset:
                motion = np.zeros((self.height, self.width, 2), dtype=np.float32)
            else:
                cur_to_prev = self.dis.calc(current, self.previous_gray, None)
                prev_to_cur = self.dis.calc(self.previous_gray, current, None)
                yy, xx = np.mgrid[0 : self.flow_height, 0 : self.flow_width].astype(
                    np.float32
                )
                sample_x = xx + cur_to_prev[..., 0]
                sample_y = yy + cur_to_prev[..., 1]
                reverse = cv2.remap(
                    prev_to_cur,
                    sample_x,
                    sample_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                consistency = cv2.magnitude(
                    cur_to_prev[..., 0] + reverse[..., 0],
                    cur_to_prev[..., 1] + reverse[..., 1],
                )
                warped_previous = cv2.remap(
                    self.previous_gray,
                    sample_x,
                    sample_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                residual = (
                    cv2.absdiff(current, warped_previous).astype(np.float32) / 255.0
                )
                invalid = np.maximum(
                    np.clip(consistency / 2.5, 0.0, 1.0),
                    np.clip(residual * 4.0, 0.0, 1.0),
                )
                cv2.dilate(invalid, np.ones((3, 3), np.uint8), iterations=1)

                motion = cv2.resize(
                    cur_to_prev,
                    (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                motion[..., 0] *= self.width / self.flow_width
                motion[..., 1] *= self.height / self.flow_height
        self.previous_gray = current
        assert motion.size == pixels * 2
        return GuideFrame(
            motion=np.ascontiguousarray(motion.astype(np.float16)),
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
    options = options or ConversionOptions()
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    validate_codec_container(options.codec, options.container)
    preview_seconds, preview_frames = _validate_preview_options(options)
    is_preview = preview_seconds is not None or preview_frames is not None
    if options.preserve_hdr:
        raise RuntimeError(
            "HDR preservation is disabled in this build because the verified DLSSNR path is "
            "RGBA8. HDR input is converted to SDR instead of being mislabeled as HDR."
        )
    validate_runtime_files()

    with active_job() as controller:
        started = time.perf_counter()
        job_dir: Path | None = None
        output: Path | None = None
        session: DLSSFrameSession | None = None
        gpu: dict | None = None
        runtime_bundle: dict | None = None
        encoder = None
        nut = None
        input_container = None
        try:
            metadata = probe_video(source)
            if preview_frames is not None:
                frame_count = min(int(metadata["frames"]), preview_frames)
            elif preview_seconds is not None:
                frame_count = ffmpeg.preview_frame_count(source, preview_seconds)
            else:
                frame_count = int(metadata["frames"])
            gpu = detect_gpu()
            runtime_bundle = inspect_runtime_bundle()
            validate_gpu_runtime(gpu, runtime_bundle)
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
            output = OUTPUTS / f"{source.stem}_{output_kind}_{stamp}{extension}"
            temp_video = job_dir / "processed-video.mkv"
            native = resolve_native_settings(options)
            if progress:
                progress(0.01, "Starting feature 18 (the native worker picks the adapter)")

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

            encode_gpu = None
            if options.dual_gpu_encode:
                # The DLSS render owns whichever adapter the worker bound; give
                # the encode to any other detected card so both GPUs share the job.
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
                            0.03,
                            f"Dual GPU skipped: {encode_gpu['name']} has only "
                            f"{encode_gpu.get('memory_free_mb', 0)} MB VRAM free "
                            f"(needs {MIN_ENCODE_GPU_FREE_MB} MB); encoding on "
                            f"{gpu['display_name']}",
                        )
                    encode_gpu = None
                elif encode_gpu and progress:
                    progress(
                        0.03,
                        f"Dual GPU: rendering on {gpu['display_name']}, "
                        f"encoding on {encode_gpu['name']}",
                    )
            (
                encoder,
                encoder_thread,
                encoder_logs,
                selected_encoder,
                encoding_quality,
            ) = ffmpeg.start_encoder(
                temp_video,
                options.codec,
                options.quality,
                controller,
                output_width,
                output_height,
                float(metadata["fps"]),
                encode_gpu=encode_gpu,
            )
            assert encoder.stdin is not None
            nut = av.open(encoder.stdin, mode="w", format="nut")
            input_container = av.open(str(source))
            input_stream = input_container.streams.video[0]
            input_stream.thread_type = "AUTO"
            rate = input_stream.average_rate or metadata["rate"]
            raw_stream = nut.add_stream("rawvideo", rate=rate)
            raw_stream.width = output_width
            raw_stream.height = output_height
            raw_stream.pix_fmt = "rgba"
            raw_stream.time_base = input_stream.time_base or metadata["time_base"]
            # Keep the source's fine time base in the encoder context too. The
            # default (1/rate) quantizes VFR timestamps to frame-number ticks,
            # and two close frames then collide into one tick, which the NUT
            # muxer rejects as non-monotonic dts.
            raw_stream.codec_context.time_base = raw_stream.time_base
            encoded_count = 0

            def mux_packet(packet) -> None:
                """Surface encoder death as its own log instead of a mux EINVAL."""
                try:
                    nut.mux(packet)
                except Exception as mux_exc:
                    encoder_code = encoder.poll()
                    if encoder_code is None:
                        raise
                    encoder_thread.join(timeout=5)
                    raise RuntimeError(
                        f"The video encoder ({selected_encoder}) exited with code "
                        f"{encoder_code} during frame {encoded_count + 1}/{frame_count}:\n"
                        + ("\n".join(encoder_logs[-40:]) or "It produced no output.")
                    ) from mux_exc

            # Three-stage pipeline: decode+guide generation (CPU) and the final
            # encode handoff each run in their own thread so the DLSS worker is
            # never left waiting on them; only the worker round-trip is serial.
            guides = TemporalGuideGenerator(render_width, render_height)
            delivered = 0
            scene_resets = 0
            preview_pts_origin: int | None = None
            last_mux_pts: int | None = None
            pts_collisions = 0
            max_in_flight = 2
            prepared_frames: queue.Queue = queue.Queue(maxsize=3)
            processed_frames: queue.Queue = queue.Queue(maxsize=3)
            stop_pipeline = threading.Event()
            stage_errors: list[BaseException] = []

            def _pipe_put(target: queue.Queue, item) -> bool:
                while not stop_pipeline.is_set():
                    try:
                        target.put(item, timeout=0.2)
                        return True
                    except queue.Full:
                        continue
                return False

            def _pipe_get(source: queue.Queue):
                while not stop_pipeline.is_set():
                    try:
                        return source.get(timeout=0.2)
                    except queue.Empty:
                        continue
                raise Cancelled("Render stopped.")

            def _stage(target) -> threading.Thread:
                def runner() -> None:
                    try:
                        target()
                    except BaseException as exc:
                        stage_errors.append(exc)
                        stop_pipeline.set()
                thread = threading.Thread(target=runner, daemon=True)
                thread.start()
                return thread

            def prepare_stage() -> None:
                for index, frame in enumerate(input_container.decode(input_stream)):
                    if index >= frame_count or stop_pipeline.is_set():
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
                    if not _pipe_put(prepared_frames, (index, rgba, guide, pts)):
                        return
                _pipe_put(prepared_frames, None)

            def encode_stage() -> None:
                nonlocal encoded_count, preview_pts_origin, last_mux_pts, pts_collisions
                while True:
                    item = _pipe_get(processed_frames)
                    if item is None:
                        return
                    processed, out_pts = item
                    out_frame = av.VideoFrame.from_ndarray(processed, format="rgba")
                    if is_preview:
                        if preview_pts_origin is None:
                            preview_pts_origin = out_pts
                        out_frame.pts = out_pts - preview_pts_origin
                    else:
                        out_frame.pts = out_pts
                    out_frame.time_base = input_stream.time_base or metadata["time_base"]
                    if last_mux_pts is not None and out_frame.pts <= last_mux_pts:
                        out_frame.pts = last_mux_pts + 1
                        pts_collisions += 1
                    last_mux_pts = out_frame.pts
                    for packet in raw_stream.encode(out_frame):
                        mux_packet(packet)
                    encoded_count += 1

            def send_stage() -> None:
                # A dedicated sender lets the main thread keep draining the
                # worker's stdout; sending and receiving from one thread can
                # deadlock on full pipes once a frame is in flight.
                while True:
                    item = _pipe_get(prepared_frames)
                    if item is None:
                        _pipe_put(sent_frames, None)
                        return
                    index, rgba, guide, pts = item
                    session.send_frame(
                        index=index,
                        rgba=rgba,
                        motion=guide.motion,
                        reset=guide.reset,
                        pts=pts,
                    )
                    if not _pipe_put(sent_frames, (index, guide.reset)):
                        return

            sent_frames: queue.Queue = queue.Queue(maxsize=max_in_flight)
            prepare_thread = _stage(prepare_stage)
            send_thread = _stage(send_stage)
            encode_thread_out = _stage(encode_stage)
            try:
                while True:
                    if stage_errors:
                        raise stage_errors[0]
                    try:
                        entry = _pipe_get(sent_frames)
                    except Cancelled:
                        if stage_errors:
                            raise stage_errors[0] from None
                        raise
                    if entry is None:
                        break
                    index, was_reset = entry
                    result = session.receive_frame(index)
                    if not _pipe_put(processed_frames, result):
                        raise stage_errors[0] if stage_errors else Cancelled(
                            "Render stopped."
                        )
                    scene_resets += int(was_reset and index != 0)
                    delivered += 1
                    if progress:
                        progress(
                            0.04 + 0.84 * delivered / frame_count,
                            f"DLSS 5 frame {delivered}/{frame_count}",
                        )
                _pipe_put(processed_frames, None)
                encode_thread_out.join(timeout=120)
                if stage_errors:
                    raise stage_errors[0]
                if encode_thread_out.is_alive():
                    raise RuntimeError("The encode stage did not finish in time.")
            finally:
                stop_pipeline.set()
                prepare_thread.join(timeout=10)
                send_thread.join(timeout=10)
                encode_thread_out.join(timeout=10)

            if delivered != frame_count:
                raise RuntimeError(
                    f"Decoded {delivered} frames instead of the expected {frame_count}; "
                    "refusing an incomplete render."
                )
            for packet in raw_stream.encode():
                mux_packet(packet)
            nut.close()
            nut = None
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            input_container.close()
            input_container = None
            session.close()
            encoder_code = encoder.wait(timeout=120)
            encoder_thread.join(timeout=2)
            controller.unregister(encoder)
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
            ffmpeg.final_mux(temp_video, source, output, options.container)
            verified = probe_video(output)
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
                "pts_collisions_adjusted": pts_collisions,
                "pipeline": "renodx-dlssnr-feature18",
                "feature_id": 18,
                "feature_18_confirmed": True,
                "carrier_create_result": carrier_create_result,
                "successful_neural_rendering_frames": nr_count,
                "model_sha256": hashlib.sha256(
                    (RUNTIME / "nvngx_dlssnr.dll").read_bytes()
                ).hexdigest(),
                "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
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
                "worker_log": session.worker_logs,
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
            )
        except Exception as exc:
            was_cancelled = controller.cancel.is_set()
            controller.stop()
            if session is not None and not session.closed:
                with suppress(Exception):
                    session.abort()
            if nut is not None:
                with suppress(Exception):
                    nut.close()
            if input_container is not None:
                with suppress(Exception):
                    input_container.close()
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
            failure_encoder_logs = encoder_logs if encoder is not None else []
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
                encoder_logs=failure_encoder_logs,
            )
            raise RuntimeError(f"{exc}\nDiagnostic report: {report_path}") from exc
        finally:
            if job_dir and job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
