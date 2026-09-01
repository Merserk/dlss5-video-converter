from __future__ import annotations

import atexit
import mmap
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runtime import (
    ADDON,
    FFMPEG,
    FFPROBE,
    NEURAL_RUNTIME,
    RUNTIME,
    WORKER,
    detect_gpu,
    inspect_runtime_bundle,
    validate_gpu_runtime,
    validate_runtime_files,
)


@dataclass(slots=True)
class PreparedRuntime:
    """Reusable, source-independent state prepared once for this process."""

    gpu: dict[str, Any]
    runtime_bundle: dict[str, Any]
    encoder_inventory: dict[str, bool]
    warmed_files: tuple[str, ...]
    _mappings: list[mmap.mmap] = field(default_factory=list, repr=False)

    def close(self) -> None:
        while self._mappings:
            mapping = self._mappings.pop()
            try:
                mapping.close()
            except (BufferError, OSError):
                pass


_PREPARE_LOCK = threading.Lock()
_PREPARED: PreparedRuntime | None = None


def _warm_mapping(path: Path) -> mmap.mmap | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    # Touch regularly spaced pages. Sequential hashing already warms the largest
    # model; this also covers un-hashed loader and FFmpeg components.
    checksum = 0
    for offset in range(0, len(mapping), 64 * 1024):
        checksum ^= mapping[offset]
    checksum ^= mapping[-1]
    del checksum
    return mapping


def _encoder_inventory() -> dict[str, bool]:
    result = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg encoder inventory failed.")
    output = result.stdout
    return {
        "h264_nvenc": "h264_nvenc" in output,
        "hevc_nvenc": "hevc_nvenc" in output,
        "av1_nvenc": "av1_nvenc" in output,
        "prores_ks": "prores_ks" in output,
    }


def prepare_runtime() -> PreparedRuntime:
    """Prepare all reusable runtime state before UI launch or first conversion."""
    global _PREPARED
    if _PREPARED is not None:
        return _PREPARED
    with _PREPARE_LOCK:
        if _PREPARED is not None:
            return _PREPARED

        validate_runtime_files()
        gpu = detect_gpu()
        runtime_bundle = inspect_runtime_bundle()
        validate_gpu_runtime(gpu, runtime_bundle)
        inventory = _encoder_inventory()

        # Initialize Pillow's reusable sRGB profile and optional image backends.
        from .images import initialize_image_runtime

        initialize_image_runtime()

        paths = (
            WORKER,
            RUNTIME / "dxgi.dll",
            ADDON,
            RUNTIME / "nvngx_dlss.dll",
            NEURAL_RUNTIME,
            FFMPEG,
            FFPROBE,
        )
        mappings: list[mmap.mmap] = []
        try:
            for path in paths:
                mapping = _warm_mapping(path)
                if mapping is not None:
                    mappings.append(mapping)
        except Exception:
            for mapping in mappings:
                mapping.close()
            raise

        _PREPARED = PreparedRuntime(
            gpu=dict(gpu),
            runtime_bundle=runtime_bundle,
            encoder_inventory=inventory,
            warmed_files=tuple(str(path.resolve()) for path in paths),
            _mappings=mappings,
        )
        return _PREPARED


def close_prepared_runtime() -> None:
    global _PREPARED
    with _PREPARE_LOCK:
        prepared = _PREPARED
        _PREPARED = None
    if prepared is not None:
        prepared.close()


atexit.register(close_prepared_runtime)
