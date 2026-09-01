from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import av

from .runtime import FFMPEG, FFPROBE, JobController, drain_text


ENCODING_QUALITIES = ("Auto (Default)", "Max", "Best", "Good")
AUTO_BITRATE_DIVISORS = {
    "H.264": 165_888,
    "HEVC": 331_776,
    "AV1": 414_720,
}


def validate_codec_container(codec: str, container: str) -> None:
    if codec == "ProRes Proxy" and container == "MP4":
        raise ValueError("ProRes Proxy is not supported in MP4. Choose the MOV or MKV container.")


def calculate_auto_bitrate_kbps(
    width: int,
    height: int,
    fps: float,
    codec: str,
    bit_depth: int = 8,
) -> int:
    try:
        divisor = AUTO_BITRATE_DIVISORS[codec]
    except KeyError as exc:
        raise ValueError(f"Automatic bitrate is unavailable for codec {codec!r}.") from exc
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or bit_depth <= 0:
        raise ValueError("Automatic bitrate requires positive dimensions, frame rate, and bit depth.")
    value = width * height * fps * bit_depth * 2 / divisor
    return max(1, int(math.floor(value + 0.5)))


def resolve_encoding_quality(
    quality_name: str,
    codec: str,
    width: int,
    height: int,
    fps: float,
) -> dict:
    if quality_name not in ENCODING_QUALITIES:
        raise ValueError(f"Unknown encoding quality: {quality_name!r}.")
    if codec == "ProRes Proxy":
        return {
            "selection": quality_name,
            "mode": "fixed-prores-proxy-profile",
            "target_bitrate_kbps": None,
            "cq": None,
        }
    if quality_name == "Max":
        return {
            "selection": quality_name,
            "mode": "constant-quality",
            "target_bitrate_kbps": None,
            "cq": 0,
        }
    multiplier = {"Auto (Default)": 1, "Good": 2, "Best": 4}[quality_name]
    auto = calculate_auto_bitrate_kbps(width, height, fps, codec)
    return {
        "selection": quality_name,
        "mode": "target-bitrate",
        "auto_bitrate_kbps": auto,
        "multiplier": multiplier,
        "target_bitrate_kbps": auto * multiplier,
        "cq": None,
    }


def _run_json(command: list[str]) -> dict:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Media probe failed")
    return json.loads(result.stdout)


def probe_video(path: str | os.PathLike[str]) -> dict:
    data = _run_json(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=index,codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,duration,nb_frames,nb_read_frames,color_primaries,color_transfer,color_space:stream_tags=rotate:stream_side_data=rotation",
            "-show_entries",
            "format=duration,format_name",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("The selected file contains no decodable video stream.")
    stream = streams[0]
    rotation = int((stream.get("tags") or {}).get("rotate", 0) or 0)
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"] or 0)
    rotation %= 360
    width, height = int(stream["width"]), int(stream["height"])
    if rotation in (90, 270):
        width, height = height, width
    frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    if frames <= 0:
        raise ValueError("Could not determine an exact frame count for this video.")
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    rate = Fraction(rate_text) if rate_text != "0/0" else Fraction(30, 1)
    transfer = stream.get("color_transfer") or "unknown"
    return {
        "width": width,
        "height": height,
        "coded_width": int(stream["width"]),
        "coded_height": int(stream["height"]),
        "rotation": rotation,
        "frames": frames,
        "fps": float(rate),
        "rate": rate,
        "time_base": Fraction(stream.get("time_base") or "1/1000"),
        "duration": float(
            (data.get("format") or {}).get("duration") or stream.get("duration") or 0
        ),
        "codec": stream.get("codec_name") or "unknown",
        "format": (data.get("format") or {}).get("format_name") or "unknown",
        "color_transfer": transfer,
        "hdr": transfer in {"smpte2084", "arib-std-b67"},
    }


def _encoder_probe(
    codec: str, width: int, height: int, env: dict[str, str] | None = None
) -> bool:
    command = [
        str(FFMPEG),
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=size={width}x{height}:rate=1",
        "-frames:v",
        "1",
        "-c:v",
        codec,
        "-f",
        "null",
        "-",
    ]
    return (
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        ).returncode
        == 0
    )


def _codec_command(
    codec: str,
    quality_name: str,
    width: int,
    height: int,
    fps: float,
    env: dict[str, str] | None = None,
) -> tuple[list[str], str, dict]:
    quality = resolve_encoding_quality(quality_name, codec, width, height, fps)
    if codec == "ProRes Proxy":
        return (
            ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"],
            "prores_ks (Proxy)",
            quality,
        )
    if quality["mode"] == "constant-quality":
        nvenc_quality = ["-rc", "vbr", "-cq", "0", "-b:v", "0"]
        software_quality = ["-crf", "0"]
    else:
        bitrate = f"{quality['target_bitrate_kbps']}k"
        nvenc_quality = ["-rc", "vbr", "-b:v", bitrate]
        software_quality = ["-b:v", bitrate]
    if codec == "H.264":
        if _encoder_probe("h264_nvenc", width, height, env):
            return (
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p6",
                    "-tune",
                    "hq",
                    *nvenc_quality,
                    "-pix_fmt",
                    "yuv420p",
                ],
                "h264_nvenc",
                quality,
            )
        return (
            ["-c:v", "libx264", "-preset", "slow", *software_quality, "-pix_fmt", "yuv420p"],
            "libx264",
            quality,
        )
    if codec == "HEVC":
        if _encoder_probe("hevc_nvenc", width, height, env):
            return (
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    "p6",
                    "-tune",
                    "hq",
                    *nvenc_quality,
                    "-pix_fmt",
                    "yuv420p",
                ],
                "hevc_nvenc",
                quality,
            )
        return (
            ["-c:v", "libx265", "-preset", "slow", *software_quality, "-pix_fmt", "yuv420p"],
            "libx265",
            quality,
        )
    if codec != "AV1":
        raise ValueError(f"Unknown video codec: {codec!r}.")
    if not _encoder_probe("av1_nvenc", width, height, env):
        raise RuntimeError(
            f"AV1 NVENC cannot encode the requested {width}×{height} output on this GPU/driver. "
            "Choose H.264/HEVC or a lower upscaling factor."
        )
    return (
        ["-c:v", "av1_nvenc", "-preset", "p6", *nvenc_quality, "-pix_fmt", "yuv420p"],
        "av1_nvenc",
        quality,
    )


def start_encoder(
    temp_video: Path,
    codec: str,
    quality_name: str,
    controller: JobController,
    width: int,
    height: int,
    fps: float,
    encode_gpu: dict | None = None,
):
    env = None
    if encode_gpu and encode_gpu.get("uuid"):
        # Pin NVENC (a CUDA context underneath) to the chosen card; probing uses
        # the same environment so the fallback decision matches the real encode.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(encode_gpu["uuid"])
    codec_args, selected, quality = _codec_command(
        codec, quality_name, width, height, fps, env
    )
    if env is not None and selected.endswith("_nvenc"):
        selected = f"{selected} on {encode_gpu['name']}"
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "nut",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        *codec_args,
        "-fps_mode",
        "passthrough",
        str(temp_video),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    controller.register(process)
    logs: list[str] = []
    assert process.stderr is not None
    thread = threading.Thread(target=drain_text, args=(process.stderr, logs), daemon=True)
    thread.start()
    return process, thread, logs, selected, quality


def _probe_rendered_duration(path: Path) -> float:
    """Read the intermediate video's duration without decoding/counting its frames."""
    data = _run_json(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = data.get("streams") or []
    values = [
        (streams[0] if streams else {}).get("duration"),
        (data.get("format") or {}).get("duration"),
    ]
    for raw_value in values:
        try:
            duration = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    raise RuntimeError(
        "Could not determine the rendered video's duration for the final audio mux."
    )


def final_mux(temp_video: Path, source: Path, output: Path, container: str) -> None:
    duration = _probe_rendered_duration(temp_video)
    if container == "MKV":
        maps = ["-map", "0:v:0", "-map", "1:a?", "-map", "1:s?"]
        streams = ["-c:v", "copy", "-c:a", "copy", "-c:s", "copy"]
    else:
        maps = ["-map", "0:v:0", "-map", "1:a?"]
        streams = [
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ]
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        str(temp_video),
        "-t",
        f"{duration:.9f}",
        "-i",
        str(source),
        *maps,
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        *streams,
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError("Final audio/metadata mux failed:\n" + result.stderr[-4000:])


def preview_frame_count(source: Path, seconds: float) -> int:
    """Count frames whose presentation times fall within the opening interval."""
    container = av.open(str(source))
    try:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 30)
        first_time: float | None = None
        count = 0
        for frame in container.decode(stream):
            timestamp = (
                float(frame.pts * stream.time_base)
                if frame.pts is not None and stream.time_base is not None
                else count / rate
            )
            if first_time is None:
                first_time = timestamp
            if count and timestamp - first_time >= seconds:
                break
            count += 1
        if count == 0:
            raise RuntimeError("The input video contains no decodable preview frames.")
        return count
    finally:
        container.close()
