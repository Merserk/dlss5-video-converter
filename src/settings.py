from __future__ import annotations

import configparser
import math
import os
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg import ENCODING_QUALITIES
from .runtime import AUTO_GPU
from .naming import RENAME_MODES, validate_rename
from .video import (
    DLSS_MODEL_PRESETS,
    NR_PRESETS,
    NR_STYLES,
    ConversionOptions,
    resolve_native_settings,
    resolve_upscaling_mode,
)


QUALITY_CHOICES = ENCODING_QUALITIES
CODEC_CHOICES = ("H.264", "HEVC", "AV1", "ProRes Proxy")
CONTAINER_CHOICES = ("MP4", "MKV", "MOV")
IMAGE_FORMAT_CHOICES = ("PNG", "JPEG", "WebP", "AVIF", "TIFF")
CONFIG_SECTION = "Settings"


@dataclass(frozen=True, slots=True)
class UISettings:
    nr_style: str = "Default"
    nr_intensity: float = 1.0
    local_tone_strength: float = 1.0
    local_structure_strength: float = 1.0
    skin_structure_strength: float = -1.0
    upscaling_factor: float = 1.0
    codec: str = "H.264"
    container: str = "MP4"
    quality: str = "Auto (Default)"
    image_format: str = "PNG"
    image_quality: int = 95
    nr_preset: str = "Default"
    automatic_mask: bool = False
    image_rename_mode: str = "Auto"
    image_custom_suffix: str = "_DLSS5"
    video_rename_mode: str = "Auto"
    video_custom_suffix: str = "_DLSS5"
    dlss_model_preset: str = "Default"
    gpu: str = AUTO_GPU
    dual_gpu_encode: bool = True

    def component_values(
        self,
    ) -> tuple[str, str, float, float, float, float, float, bool, str, str, str, str]:
        return (
            self.nr_preset,
            self.nr_style,
            self.nr_intensity,
            self.local_tone_strength,
            self.local_structure_strength,
            self.skin_structure_strength,
            self.upscaling_factor,
            self.automatic_mask,
            self.codec,
            self.container,
            self.quality,
            self.dlss_model_preset,
        )


DEFAULT_SETTINGS = UISettings()


def _validate(settings: UISettings) -> UISettings:
    resolve_native_settings(
        ConversionOptions(
            nr_preset=settings.nr_preset,
            nr_style=settings.nr_style,
            nr_intensity=settings.nr_intensity,
            local_tone_strength=settings.local_tone_strength,
            local_structure_strength=settings.local_structure_strength,
            skin_structure_strength=settings.skin_structure_strength,
            automatic_mask=settings.automatic_mask,
            upscaling_factor=settings.upscaling_factor,
            dlss_model_preset=settings.dlss_model_preset,
        )
    )
    resolve_upscaling_mode(settings.upscaling_factor)
    if not isinstance(settings.automatic_mask, bool):
        raise ValueError("Automatic Mask must be a boolean value.")
    allowed = {
        "Video codec": (settings.codec, CODEC_CHOICES),
        "Container": (settings.container, CONTAINER_CHOICES),
        "Encoding quality": (settings.quality, QUALITY_CHOICES),
        "Image format": (settings.image_format, IMAGE_FORMAT_CHOICES),
    }
    for label, (value, choices) in allowed.items():
        if value not in choices:
            raise ValueError(f"Unknown {label}: {value!r}.")
    if isinstance(settings.image_quality, bool) or not 1 <= int(settings.image_quality) <= 100:
        raise ValueError("Image quality must be an integer from 1 to 100.")
    if int(settings.image_quality) != settings.image_quality:
        raise ValueError("Image quality must be an integer from 1 to 100.")
    validate_rename(settings.image_rename_mode, settings.image_custom_suffix)
    validate_rename(settings.video_rename_mode, settings.video_custom_suffix)
    if not isinstance(settings.gpu, str) or not settings.gpu.strip():
        raise ValueError("GPU must be a non-empty selection label.")
    if not isinstance(settings.dual_gpu_encode, bool):
        raise ValueError("Dual-GPU encode must be a boolean value.")
    return settings


def load_settings(path: str | os.PathLike[str]) -> UISettings:
    config_path = Path(path)
    parser = configparser.ConfigParser()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error):
        return DEFAULT_SETTINGS

    section = parser[CONFIG_SECTION] if parser.has_section(CONFIG_SECTION) else {}

    def choice(key: str, choices: tuple[str, ...], default: str) -> str:
        value = section.get(key, default)
        return value if value in choices else default

    def number(key: str, minimum: float, maximum: float, default: float) -> float:
        try:
            value = float(section.get(key, str(default)))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and minimum <= value <= maximum else default

    def upscaling_factor() -> float:
        try:
            return resolve_upscaling_mode(float(section.get("upscaling_factor", "1.0")))[0]
        except (TypeError, ValueError):
            return DEFAULT_SETTINGS.upscaling_factor

    def image_quality() -> int:
        value = number("image_quality", 1, 100, DEFAULT_SETTINGS.image_quality)
        return int(value) if float(value).is_integer() else DEFAULT_SETTINGS.image_quality

    def boolean(key: str, default: bool) -> bool:
        raw_value = section.get(key)
        if raw_value is None:
            return default
        parsed = configparser.ConfigParser.BOOLEAN_STATES.get(str(raw_value).casefold())
        return parsed if parsed is not None else default

    image_rename_mode = choice(
        "image_rename_mode", RENAME_MODES, DEFAULT_SETTINGS.image_rename_mode
    )
    image_custom_suffix = section.get(
        "image_custom_suffix", DEFAULT_SETTINGS.image_custom_suffix
    )
    video_rename_mode = choice(
        "video_rename_mode", RENAME_MODES, DEFAULT_SETTINGS.video_rename_mode
    )
    video_custom_suffix = section.get(
        "video_custom_suffix", DEFAULT_SETTINGS.video_custom_suffix
    )
    try:
        validate_rename(image_rename_mode, image_custom_suffix)
    except ValueError:
        image_rename_mode = DEFAULT_SETTINGS.image_rename_mode
        image_custom_suffix = DEFAULT_SETTINGS.image_custom_suffix
    try:
        validate_rename(video_rename_mode, video_custom_suffix)
    except ValueError:
        video_rename_mode = DEFAULT_SETTINGS.video_rename_mode
        video_custom_suffix = DEFAULT_SETTINGS.video_custom_suffix

    return UISettings(
        nr_preset=choice("nr_preset", tuple(NR_PRESETS), DEFAULT_SETTINGS.nr_preset),
        nr_style=choice("nr_style", tuple(NR_STYLES), DEFAULT_SETTINGS.nr_style),
        nr_intensity=number("nr_intensity", 0.0, 2.0, DEFAULT_SETTINGS.nr_intensity),
        local_tone_strength=number(
            "local_tone_strength", 0.0, 2.0, DEFAULT_SETTINGS.local_tone_strength
        ),
        local_structure_strength=number(
            "local_structure_strength", 0.0, 2.0, DEFAULT_SETTINGS.local_structure_strength
        ),
        skin_structure_strength=number(
            "skin_structure_strength", -1.0, 2.0, DEFAULT_SETTINGS.skin_structure_strength
        ),
        automatic_mask=boolean("automatic_mask", DEFAULT_SETTINGS.automatic_mask),
        upscaling_factor=upscaling_factor(),
        codec=choice("codec", CODEC_CHOICES, DEFAULT_SETTINGS.codec),
        container=choice("container", CONTAINER_CHOICES, DEFAULT_SETTINGS.container),
        quality=choice("quality", QUALITY_CHOICES, DEFAULT_SETTINGS.quality),
        image_format=choice(
            "image_format", IMAGE_FORMAT_CHOICES, DEFAULT_SETTINGS.image_format
        ),
        image_quality=image_quality(),
        gpu=section.get("gpu", DEFAULT_SETTINGS.gpu) or DEFAULT_SETTINGS.gpu,
        dual_gpu_encode=boolean("dual_gpu_encode", DEFAULT_SETTINGS.dual_gpu_encode),
        dlss_model_preset=choice(
            "dlss_model_preset",
            tuple(DLSS_MODEL_PRESETS),
            DEFAULT_SETTINGS.dlss_model_preset,
        ),
        image_rename_mode=image_rename_mode,
        image_custom_suffix=image_custom_suffix,
        video_rename_mode=video_rename_mode,
        video_custom_suffix=video_custom_suffix,
    )


def save_settings(path: str | os.PathLike[str], settings: UISettings) -> None:
    settings = _validate(settings)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    parser[CONFIG_SECTION] = {
        "nr_preset": settings.nr_preset,
        "nr_style": settings.nr_style,
        "nr_intensity": f"{settings.nr_intensity:.2f}",
        "local_tone_strength": f"{settings.local_tone_strength:.2f}",
        "local_structure_strength": f"{settings.local_structure_strength:.2f}",
        "skin_structure_strength": f"{settings.skin_structure_strength:.2f}",
        "automatic_mask": str(settings.automatic_mask).lower(),
        "upscaling_factor": f"{settings.upscaling_factor:g}",
        "codec": settings.codec,
        "container": settings.container,
        "quality": settings.quality,
        "image_format": settings.image_format,
        "image_quality": str(settings.image_quality),
        "image_rename_mode": settings.image_rename_mode,
        "image_custom_suffix": settings.image_custom_suffix,
        "video_rename_mode": settings.video_rename_mode,
        "video_custom_suffix": settings.video_custom_suffix,
        "dlss_model_preset": settings.dlss_model_preset,
        "gpu": settings.gpu,
        "dual_gpu_encode": str(settings.dual_gpu_encode).lower(),
    }

    temporary = config_path.with_name(f".{config_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            parser.write(stream)
        os.replace(temporary, config_path)
    finally:
        if temporary.exists():
            temporary.unlink()
