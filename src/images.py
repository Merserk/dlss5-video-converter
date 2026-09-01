from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pillow_heif
import rawpy
import resvg_py
from PIL import Image, ImageCms, ImageOps, TiffImagePlugin

from .naming import output_filename, require_available_output, validate_rename
from .runtime import (
    LOGS,
    OUTPUTS,
    Cancelled,
    DLSSFrameSession,
    active_job,
    resize_fit,
    validate_gpu_runtime,
    verify_feature_18,
    write_failure_report,
)
from .video import (
    ConversionOptions,
    DLSS_MODEL_PRESETS,
    resolve_native_settings,
    resolve_output_size,
    resolve_upscaling_mode,
)


pillow_heif.register_heif_opener()
Image.MAX_IMAGE_PIXELS = 100_000_000

IMAGE_FORMATS = ("PNG", "JPEG", "WebP", "AVIF", "TIFF")
IMAGE_EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp", "AVIF": ".avif", "TIFF": ".tiff"}
_PREVIEW_CACHE_LIMIT = 32
_preview_cache: OrderedDict[str, Image.Image] = OrderedDict()
_preview_cache_lock = threading.Lock()
RAW_EXTENSIONS = {
    ".3fr", ".arw", ".bay", ".cap", ".cr2", ".cr3", ".dcr", ".dcs", ".dng",
    ".drf", ".eip", ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc",
    ".mef", ".mos", ".mrw", ".nef", ".nrw", ".obm", ".orf", ".pef", ".ptx",
    ".pxn", ".r3d", ".raf", ".raw", ".rw2", ".rwl", ".rwz", ".sr2", ".srf",
    ".srw", ".x3f",
}


@dataclass(slots=True)
class ImageConversionOptions:
    nr_style: str = "Default"
    nr_intensity: float = 1.0
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    upscaling_factor: float = 1.0
    output_format: str = "PNG"
    quality: int = 95
    preserve_metadata: bool = True
    warmup_frames: int = 0
    nr_preset: str = "Default"
    automatic_mask: bool = False
    rename_mode: str = "Auto"
    custom_suffix: str = "_DLSS5"
    dlss_model_preset: str = "Default"

    def neural_options(self) -> ConversionOptions:
        return ConversionOptions(
            nr_preset=self.nr_preset,
            nr_style=self.nr_style,
            nr_intensity=self.nr_intensity,
            local_tone_strength=self.local_tone_strength,
            local_structure_strength=self.local_structure_strength,
            skin_structure_strength=self.skin_structure_strength,
            automatic_mask=self.automatic_mask,
            upscaling_factor=self.upscaling_factor,
            warmup_frames=self.warmup_frames,
            dlss_model_preset=self.dlss_model_preset,
        )


@dataclass(slots=True)
class ImageConversionResult:
    input_path: str
    output_path: str
    report_path: str
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
    output_format: str
    dlss_model_preset: str = "Default"
    applied_dlss_model_preset: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImageConversionFailure:
    input_path: str
    error: str


@dataclass(slots=True)
class ImageBatchResult:
    successes: list[ImageConversionResult]
    failures: list[ImageConversionFailure]
    cancelled: bool
    manifest_path: str
    zip_path: str | None


@dataclass(slots=True)
class _ImageProbe:
    index: int
    path: Path
    width: int
    height: int
    output_width: int
    output_height: int
    decoded: _DecodedImage | None = None


@dataclass(slots=True)
class _DecodedImage:
    rgba: np.ndarray
    alpha: np.ndarray
    decoder: str
    metadata: dict[str, object]
    warnings: list[str]


@dataclass(slots=True)
class _ImageReportData:
    decoder: str
    metadata: dict[str, object]
    warnings: list[str]


def _report_data(decoded: _DecodedImage) -> _ImageReportData:
    """Keep diagnostics without retaining the decoded pixel and alpha arrays."""
    return _ImageReportData(decoded.decoder, decoded.metadata, decoded.warnings)


def _json_safe_metadata(value: object) -> object:
    """Create a report-only JSON-safe copy without changing save metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, TiffImagePlugin.IFDRational):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    return str(value)


def _validate_options(options: ImageConversionOptions) -> ConversionOptions:
    if options.output_format not in IMAGE_FORMATS:
        raise ValueError(f"Unknown image output format: {options.output_format!r}.")
    if isinstance(options.quality, bool):
        raise ValueError("Image quality must be an integer from 1 to 100.")
    try:
        quality = int(options.quality)
    except (TypeError, ValueError) as exc:
        raise ValueError("Image quality must be an integer from 1 to 100.") from exc
    if quality != options.quality or not 1 <= quality <= 100:
        raise ValueError("Image quality must be an integer from 1 to 100.")
    validate_rename(options.rename_mode, options.custom_suffix)
    neural = options.neural_options()
    resolve_native_settings(neural)
    resolve_upscaling_mode(neural.upscaling_factor)
    return neural


def _orientation_swaps_dimensions(image: Image.Image) -> bool:
    try:
        return int(image.getexif().get(274, 1)) in {5, 6, 7, 8}
    except (TypeError, ValueError):
        return False


def _open_pillow_source(path: Path) -> tuple[Image.Image, str]:
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            array = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
            )
        return Image.fromarray(array, mode="RGB"), "LibRaw"
    if suffix == ".svg":
        png = resvg_py.svg_to_bytes(svg_path=str(path), resources_dir=str(path.parent))
        return Image.open(io.BytesIO(png)), "resvg"
    return Image.open(path), "Pillow"


def probe_image(path: str | os.PathLike[str]) -> tuple[int, int]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    image, _decoder = _open_pillow_source(source)
    try:
        width, height = image.size
        if _orientation_swaps_dimensions(image):
            width, height = height, width
    finally:
        image.close()
    if width < 64 or height < 64:
        raise ValueError(
            f"{source.name} is {width}×{height}; DLSS requires both input dimensions to be at least 64 pixels."
        )
    return width, height


@lru_cache(maxsize=1)
def _srgb_profile_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def initialize_image_runtime() -> None:
    """Initialize reusable image color-management state during app preparation."""
    _srgb_profile_bytes()


def take_image_preview(output_path: str | os.PathLike[str]) -> Image.Image | None:
    """Return and remove a UI thumbnail created from the rendered frame in memory."""
    key = str(Path(output_path).resolve())
    with _preview_cache_lock:
        return _preview_cache.pop(key, None)


def _remember_image_preview(
    output_path: Path, rgba: np.ndarray, output_format: str
) -> None:
    image = Image.fromarray(rgba, mode="RGBA")
    try:
        if output_format == "JPEG":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image)
            preview = background
        else:
            preview = image.copy()
        preview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
        key = str(output_path.resolve())
        with _preview_cache_lock:
            previous = _preview_cache.pop(key, None)
            if previous is not None:
                previous.close()
            _preview_cache[key] = preview
            while len(_preview_cache) > _PREVIEW_CACHE_LIMIT:
                _unused_key, unused = _preview_cache.popitem(last=False)
                unused.close()
    finally:
        image.close()


def _encode_image(
    output: Path,
    rgba: np.ndarray,
    options: ImageConversionOptions,
    metadata: dict[str, object],
) -> list[str]:
    warnings = save_image(output, rgba, options, metadata)
    try:
        _remember_image_preview(output, rgba, options.output_format)
    except Exception:
        # A UI convenience must never invalidate an otherwise correct output.
        pass
    return warnings


def decode_image(path: str | os.PathLike[str]) -> _DecodedImage:
    source = Path(path).resolve()
    image, decoder = _open_pillow_source(source)
    opened_image = image
    warnings: list[str] = []
    try:
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        if frame_count > 1:
            warnings.append(f"Used the first frame/page of {frame_count}.")
            image.seek(0)
        original_mode = image.mode
        info = dict(image.info)
        image = ImageOps.exif_transpose(image)
        image.load()
        if original_mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
            warnings.append(f"Converted {original_mode} source data to 8-bit SDR sRGB.")

        has_alpha = image.mode in {"LA", "RGBA"} or (
            image.mode == "P" and "transparency" in info
        )
        alpha_image = (
            image.convert("RGBA").getchannel("A")
            if has_alpha
            else Image.new("L", image.size, 255)
        )
        rgb_image = image.convert("RGB")
        icc_profile = info.get("icc_profile")
        if isinstance(icc_profile, bytes) and icc_profile:
            try:
                source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                rgb_image = ImageCms.profileToProfile(
                    rgb_image,
                    source_profile,
                    ImageCms.createProfile("sRGB"),
                    outputMode="RGB",
                )
            except (ImageCms.PyCMSError, OSError, ValueError) as exc:
                warnings.append(f"Embedded color profile could not be applied ({exc}); assumed sRGB.")

        rgb = np.asarray(rgb_image, dtype=np.uint8)
        alpha = np.asarray(alpha_image, dtype=np.uint8)
        rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha
        exif = image.getexif()
        if exif:
            exif[274] = 1
        metadata: dict[str, object] = {
            "icc_profile": _srgb_profile_bytes(),
            "exif": exif.tobytes() if exif else None,
            "dpi": info.get("dpi"),
            "xmp": info.get("xmp"),
            "source_mode": original_mode,
            "source_format": image.format or source.suffix.lstrip(".").upper(),
            "frame_count": frame_count,
        }
        return _DecodedImage(
            np.ascontiguousarray(rgba),
            np.ascontiguousarray(alpha),
            decoder,
            metadata,
            warnings,
        )
    finally:
        image.close()
        if opened_image is not image:
            opened_image.close()


def _metadata_save_args(metadata: dict[str, object], output_format: str) -> dict[str, object]:
    args: dict[str, object] = {}
    if metadata.get("icc_profile"):
        args["icc_profile"] = metadata["icc_profile"]
    if metadata.get("dpi"):
        args["dpi"] = metadata["dpi"]
    if metadata.get("exif") and output_format in {"JPEG", "WebP", "AVIF", "TIFF", "PNG"}:
        args["exif"] = metadata["exif"]
    if metadata.get("xmp") and output_format in {"WebP", "AVIF"}:
        args["xmp"] = metadata["xmp"]
    return args


def save_image(
    output: Path,
    rgba: np.ndarray,
    options: ImageConversionOptions,
    metadata: dict[str, object],
) -> list[str]:
    output_format = options.output_format
    image = Image.fromarray(rgba, mode="RGBA")
    warnings: list[str] = []
    args = _metadata_save_args(metadata, output_format) if options.preserve_metadata else {}
    if output_format == "PNG":
        args.update(optimize=False, compress_level=6)
    elif output_format == "TIFF":
        args.update(compression="tiff_deflate")
    elif output_format == "JPEG":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")
        args.update(
            quality=int(options.quality),
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        if np.any(rgba[..., 3] != 255):
            warnings.append("Transparency was composited over white for JPEG output.")
    elif output_format == "WebP":
        args.update(quality=int(options.quality), method=6)
    elif output_format == "AVIF":
        args.update(quality=int(options.quality), speed=4)

    temporary = output.with_name(f".{output.stem}.{time.time_ns()}{output.suffix}")
    try:
        image.save(temporary, format=output_format, **args)
        os.replace(temporary, output)
    finally:
        image.close()
        if temporary.exists():
            temporary.unlink()
    return warnings


def _output_path(
    source: Path,
    output_format: str,
    stamp: str,
    index: int,
    rename_mode: str,
    custom_suffix: str,
) -> Path:
    suffix = IMAGE_EXTENSIONS[output_format]
    safe_stem = source.stem.strip().rstrip(".") or "image"
    return OUTPUTS / output_filename(
        source,
        suffix,
        rename_mode,
        custom_suffix,
        f"{safe_stem}_DLSS5_IMAGE_{stamp}-{index + 1:04d}",
    )


def _write_report(
    result: ImageConversionResult,
    options: ImageConversionOptions,
    decoded: _ImageReportData,
    gpu: dict,
    session: DLSSFrameSession,
    evidence: dict[str, object],
) -> str:
    report = {
        "status": "success",
        "input": result.input_path,
        "output": result.output_path,
        "options": asdict(options),
        "gpu": gpu,
        "decoder": decoded.decoder,
        "source_metadata": {
            key: _json_safe_metadata(value)
            for key, value in decoded.metadata.items()
            if key not in {"icc_profile", "exif", "xmp"}
        },
        "warnings": result.warnings,
        "input_dimensions": {"width": result.input_width, "height": result.input_height},
        "negotiated_render_dimensions": {"width": result.render_width, "height": result.render_height},
        "negotiated_render_range": {
            "minimum": {"width": session.minimum_width, "height": session.minimum_height},
            "maximum": {"width": session.maximum_width, "height": session.maximum_height},
        },
        "output_dimensions": {"width": result.output_width, "height": result.output_height},
        "dlss_mode": result.dlss_mode,
        "requested_upscaling_factor": result.upscaling_factor,
        "dlss_model_preset": result.dlss_model_preset,
        "requested_dlss_model_preset": result.dlss_model_preset,
        "requested_dlss_model_preset_code": DLSS_MODEL_PRESETS[
            result.dlss_model_preset
        ],
        "applied_dlss_model_preset": result.applied_dlss_model_preset,
        "applied_dlss_model_preset_name": next(
            name
            for name, code in DLSS_MODEL_PRESETS.items()
            if code == result.applied_dlss_model_preset
        ),
        "pipeline": "renodx-dlssnr-feature18-image",
        "feature_id": 18,
        "feature_18_confirmed": True,
        "ngx_setup_result": f"0x{session.setup_result:08X}",
        "carrier_create_result": evidence["carrier_create_result"],
        "native_settings": session.native_settings,
        "addon_release": session.runtime_bundle["addon"]["release"],
        "addon_sha256": session.runtime_bundle["addon"]["sha256"].lower(),
        "model_sha256": session.runtime_bundle["neural_runtime"]["sha256"].lower(),
        "worker_sha256": session.runtime_bundle["worker"]["sha256"].lower(),
        "worker_log": session.worker_logs,
        "worker_log_dropped_lines": session.worker_log_dropped_lines,
        "dlssnr_evidence": evidence["evidence"],
        "elapsed_seconds": result.elapsed_seconds,
    }
    report_path = LOGS / f"{Path(result.output_path).name}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(report_path)


def _build_manifest_and_zip(
    stamp: str,
    options: ImageConversionOptions,
    successes: list[ImageConversionResult],
    failures: list[ImageConversionFailure],
    cancelled: bool,
) -> tuple[str, str | None]:
    manifest = {
        "status": "cancelled" if cancelled else ("partial" if failures else "success"),
        "options": asdict(options),
        "successes": [asdict(item) for item in successes],
        "failures": [asdict(item) for item in failures],
    }
    manifest_path = LOGS / f"DLSS5_IMAGE_BATCH_{stamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not successes:
        return str(manifest_path), None
    zip_path = OUTPUTS / f"DLSS5_IMAGE_BATCH_{stamp}.zip"
    temporary = zip_path.with_name(f".{zip_path.stem}.{time.time_ns()}.zip")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for item in successes:
                archive.write(item.output_path, arcname=Path(item.output_path).name)
        os.replace(temporary, zip_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(manifest_path), str(zip_path)


def convert_images(
    input_paths: Iterable[str | os.PathLike[str]],
    options: ImageConversionOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> ImageBatchResult:
    from .prepare import prepare_runtime

    options = options or ImageConversionOptions()
    neural = _validate_options(options)
    paths = [Path(path).resolve() for path in input_paths]
    if not paths:
        raise ValueError("Choose at least one image.")
    prepared_runtime = prepare_runtime()
    OUTPUTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    failures_by_index: dict[int, ImageConversionFailure] = {}
    probes: list[_ImageProbe] = []
    for index, path in enumerate(paths):
        try:
            decoded: _DecodedImage | None = None
            if path.suffix.lower() in {*RAW_EXTENSIONS, ".svg"}:
                decoded = decode_image(path)
                height, width = decoded.rgba.shape[:2]
                if width < 64 or height < 64:
                    raise ValueError(
                        f"{path.name} is {width}×{height}; DLSS requires both input "
                        "dimensions to be at least 64 pixels."
                    )
            else:
                width, height = probe_image(path)
            output_width, output_height = resolve_output_size(
                width, height, options.upscaling_factor
            )
            probes.append(
                _ImageProbe(
                    index, path, width, height, output_width, output_height, decoded
                )
            )
        except Exception as exc:
            failures_by_index[index] = ImageConversionFailure(str(path), str(exc))

    planned_outputs: dict[int, Path] = {}
    reserved_outputs: set[str] = set()
    available_probes: list[_ImageProbe] = []
    for probe in probes:
        output = _output_path(
            probe.path,
            options.output_format,
            stamp,
            probe.index,
            options.rename_mode,
            options.custom_suffix,
        )
        output_key = str(output.resolve()).casefold()
        try:
            require_available_output(output)
            if output_key in reserved_outputs:
                raise FileExistsError(
                    f"More than one input would create {output.name}. "
                    "Choose Auto naming or use unique input names."
                )
        except Exception as exc:
            failures_by_index[probe.index] = ImageConversionFailure(
                str(probe.path), str(exc)
            )
            continue
        reserved_outputs.add(output_key)
        planned_outputs[probe.index] = output
        available_probes.append(probe)
    probes = available_probes

    if not probes:
        manifest_path, zip_path = _build_manifest_and_zip(
            stamp, options, [], list(failures_by_index.values()), False
        )
        return ImageBatchResult([], list(failures_by_index.values()), False, manifest_path, zip_path)

    groups: dict[tuple[int, int], list[_ImageProbe]] = {}
    for probe in probes:
        groups.setdefault((probe.output_width, probe.output_height), []).append(probe)

    successes_by_index: dict[int, ImageConversionResult] = {}
    cancelled = False
    gpu: dict | None = None
    processed_total = 0
    decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dlss5-image-decode")
    encode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dlss5-image-encode")
    with active_job() as controller:
        gpu = dict(prepared_runtime.gpu)
        runtime_bundle = prepared_runtime.runtime_bundle
        try:
            validate_gpu_runtime(gpu, runtime_bundle)
        except Exception as exc:
            report_path = write_failure_report(
                operation="image-batch-runtime-validation",
                source="; ".join(str(path) for path in paths),
                error=exc,
                gpu=gpu,
                runtime_bundle=runtime_bundle,
            )
            raise RuntimeError(f"{exc}\nDiagnostic report: {report_path}") from exc
        if progress:
            progress(0.0, "Starting feature 18 (the native worker picks the adapter)")
        total = len(probes)
        factor, mode = resolve_upscaling_mode(neural.upscaling_factor)
        native_settings = resolve_native_settings(neural)
        try:
            for (output_width, output_height), group in groups.items():
                cursor = 0
                while cursor < len(group):
                    if controller.cancel.is_set():
                        cancelled = True
                        break
                    remaining = group[cursor:]
                    first = remaining[0]
                    try:
                        session = DLSSFrameSession(
                            input_width=first.width,
                            input_height=first.height,
                            output_width=output_width,
                            output_height=output_height,
                            frame_count=len(remaining),
                            warmup_frames=neural.warmup_frames,
                            factor=factor,
                            mode=mode,
                            native_settings=native_settings,
                            gpu=gpu,
                            runtime_bundle=runtime_bundle,
                            controller=controller,
                        )
                    except Exception as exc:
                        controller.terminate_processes()
                        report_path = write_failure_report(
                            operation="image-batch-dlss-setup",
                            source="; ".join(str(item.path) for item in remaining),
                            error=exc,
                            gpu=gpu,
                            runtime_bundle=runtime_bundle,
                        )
                        failure = f"{exc}\nDiagnostic report: {report_path}"
                        for item in remaining:
                            failures_by_index[item.index] = ImageConversionFailure(str(item.path), failure)
                        break
    
                    segment: list[
                        tuple[
                            _ImageProbe,
                            ImageConversionResult,
                            _ImageReportData,
                            Future[list[str]],
                            float,
                        ]
                    ] = []
                    sent = 0
                    motion = np.zeros(
                        (session.render_height, session.render_width, 2), dtype=np.float16
                    )
                    interrupted = False
                    close_error: Exception | None = None
                    decode_future: Future[_DecodedImage] | None = decode_executor.submit(
                        lambda probe=group[cursor]: probe.decoded or decode_image(probe.path)
                    )
                    while cursor < len(group):
                        item = group[cursor]
                        started = time.perf_counter()
                        if controller.cancel.is_set():
                            cancelled = True
                            interrupted = True
                            session.abort()
                            break
                        if progress:
                            progress(processed_total / max(1, total), f"Preparing {item.path.name}")
                        try:
                            assert decode_future is not None
                            decoded = decode_future.result()
                            item.decoded = None
                            next_cursor = cursor + 1
                            decode_future = (
                                decode_executor.submit(
                                    lambda probe=group[next_cursor]:
                                    probe.decoded or decode_image(probe.path)
                                )
                                if next_cursor < len(group)
                                else None
                            )
                        except Exception as exc:
                            failures_by_index[item.index] = ImageConversionFailure(str(item.path), str(exc))
                            cursor += 1
                            processed_total += 1
                            session.abort()
                            interrupted = True
                            break
                        try:
                            render_rgba = resize_fit(
                                decoded.rgba, session.render_width, session.render_height
                            )
                            processed, _pts = session.process(
                                index=sent,
                                rgba=render_rgba,
                                motion=motion,
                                reset=True,
                                pts=sent,
                            )
                            alpha = (
                                decoded.alpha
                                if decoded.alpha.shape == (output_height, output_width)
                                else cv2.resize(
                                    decoded.alpha,
                                    (output_width, output_height),
                                    interpolation=cv2.INTER_LANCZOS4,
                                )
                            )
                            processed[..., 3] = alpha
                            output = planned_outputs[item.index]
                            result = ImageConversionResult(
                                str(item.path),
                                str(output),
                                "",
                                time.perf_counter() - started,
                                str(gpu["display_name"]),
                                item.width,
                                item.height,
                                session.render_width,
                                session.render_height,
                                output_width,
                                output_height,
                                float(options.upscaling_factor),
                                str(session.mode["name"]),
                                options.output_format,
                                options.dlss_model_preset,
                                session.applied_dlss_model_preset,
                                list(decoded.warnings),
                            )
                            encoding = encode_executor.submit(
                                _encode_image, output, processed, options, decoded.metadata
                            )
                            segment.append(
                                (item, result, _report_data(decoded), encoding, started)
                            )
                            # Keep at most two full rendered frames queued for encoding.
                            # Completed futures retain only their small warning list.
                            if len(segment) > 2:
                                segment[-3][3].result()
                            sent += 1
                        except Cancelled:
                            cancelled = True
                            session.abort()
                            interrupted = True
                            break
                        except Exception as exc:
                            session.abort()
                            report_path = write_failure_report(
                                operation="image-render",
                                source=str(item.path),
                                error=exc,
                                gpu=gpu,
                                runtime_bundle=runtime_bundle,
                                worker_code=session.worker.poll(),
                                worker_logs=session.worker_logs,
                                reshade_lines=session.reshade_diagnostics(),
                            )
                            failures_by_index[item.index] = ImageConversionFailure(
                                str(item.path), f"{exc}\nDiagnostic report: {report_path}"
                            )
                            interrupted = True
                            cursor += 1
                            processed_total += 1
                            break
                        cursor += 1
                        processed_total += 1
                        if progress:
                            progress(processed_total / total, f"Rendered {item.path.name}")
    
                    if not segment and interrupted:
                        if cancelled:
                            break
                        continue
                    if not interrupted:
                        try:
                            session.close()
                        except Exception as exc:
                            close_error = exc
                    try:
                        if close_error:
                            raise close_error
                        evidence = verify_feature_18(
                            session.worker_logs, session.reshade_log_text()
                        )
                        assert gpu is not None
                        for item, result, decoded, encoding, item_started in segment:
                            result.warnings.extend(encoding.result())
                            result.elapsed_seconds = time.perf_counter() - item_started
                        for item, result, decoded, _encoding, _item_started in segment:
                            result.report_path = _write_report(
                                result, options, decoded, gpu, session, evidence
                            )
                            successes_by_index[item.index] = result
                    except Exception as exc:
                        report_path = write_failure_report(
                            operation="image-batch-verification",
                            source="; ".join(str(item.path) for item, *_rest in segment),
                            error=exc,
                            gpu=gpu,
                            runtime_bundle=runtime_bundle,
                            worker_code=session.worker.poll(),
                            worker_logs=session.worker_logs,
                            reshade_lines=session.reshade_diagnostics(),
                        )
                        failure = f"{exc}\nDiagnostic report: {report_path}"
                        for item, result, _decoded, encoding, _item_started in segment:
                            if not encoding.done():
                                encoding.cancel()
                            output = Path(result.output_path)
                            if output.exists():
                                output.unlink()
                            failures_by_index[item.index] = ImageConversionFailure(str(item.path), failure)
                    if cancelled:
                        break
                if cancelled:
                    break
        finally:
            decode_executor.shutdown(wait=True, cancel_futures=True)
            encode_executor.shutdown(wait=True, cancel_futures=True)
    if cancelled:
        completed = set(successes_by_index) | set(failures_by_index)
        for probe in probes:
            if probe.index not in completed:
                failures_by_index[probe.index] = ImageConversionFailure(
                    str(probe.path), "Cancelled before rendering."
                )
    successes = [successes_by_index[index] for index in sorted(successes_by_index)]
    failures = [failures_by_index[index] for index in sorted(failures_by_index)]
    manifest_path, zip_path = _build_manifest_and_zip(
        stamp, options, successes, failures, cancelled
    )
    if progress:
        progress(1.0, "Cancelled" if cancelled else "Complete")
    return ImageBatchResult(successes, failures, cancelled, manifest_path, zip_path)


def convert_image(
    input_path: str | os.PathLike[str],
    options: ImageConversionOptions | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> ImageConversionResult:
    result = convert_images([input_path], options, progress)
    if result.successes:
        return result.successes[0]
    if result.failures:
        raise RuntimeError(result.failures[0].error)
    raise RuntimeError("Image conversion produced no result.")
