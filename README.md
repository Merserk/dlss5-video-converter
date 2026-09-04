
# DLSS 5 Visual Enhancer

[![Downloads](https://img.shields.io/github/downloads/Merserk/dlss5-visual-enhancer/total.svg?style=flat-square&label=Downloads)](https://github.com/Merserk/dlss5-visual-enhancer/releases) ![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows11&logoColor=white) ![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX-76B900?style=flat-square&logo=nvidia&logoColor=white) ![DLSS](https://img.shields.io/badge/DLSS-5-76B900?style=flat-square) ![DLSS Frame Generation](https://img.shields.io/badge/DLSS-Frame%20Generation-76B900?style=flat-square&logo=nvidia&logoColor=white) ![Type](https://img.shields.io/badge/Type-Portable-2EA44F?style=flat-square) [![License](https://img.shields.io/badge/License-MIT-007EC6?style=flat-square)](LICENSE)

Windows application for applying a DLSS 5 Neural Rendering feature-18 pipeline to images and video, and NVIDIA DLSS Frame Generation to video frame interpolation, through a local Gradio interface. It is an independent community project and is not affiliated with, sponsored by, or endorsed by NVIDIA, ReShade, RenoDX, FFmpeg, or their contributors.

<img width="1810" height="1000" alt="Screenshot 2026-09-03 005153" src="https://github.com/user-attachments/assets/ad15df03-2934-4a20-90ca-5a6514a5de32" />

### Original

https://github.com/user-attachments/assets/8df8bd4c-01b4-47dd-9705-3614a0b0ff75

### DLSS 5

https://github.com/user-attachments/assets/cff68783-4ee9-4c99-8b36-4eee2a6437ec

### Frame Generation (DLSSG)

https://github.com/user-attachments/assets/81c29005-e4f0-4acf-b9f7-d58850bb055f

## Installation

1. Download the [latest release](https://github.com/Merserk/dlss5-visual-enhancer/releases/latest).
2. Unpack the downloaded ZIP archive.
3. Run `start.bat`.

## Main features

- **Images:** single-image and batch processing with per-file success/failure results, responsive input/output previews, individual downloads, a ZIP of successful outputs, batch manifests, and diagnostic JSON reports.
- **Image formats:** common Pillow formats plus HEIF/HEIC, SVG, and many camera RAW formats. Outputs are PNG, JPEG, WebP, AVIF, or TIFF.
- **Image handling:** EXIF orientation is applied; ICC input is converted to sRGB; supported EXIF, DPI, and XMP metadata is retained; alpha is preserved except when JPEG composites it over white. EXIF/TIFF rational values and other unusual metadata are converted safely for diagnostic JSON without modifying the metadata used to encode the output. Animated and multipage files use only the first frame/page.
- **Video:** single-video and reorderable batch-video Neural Rendering, plus one-frame and three-second previews for a single selected video. H.264, H.265, AV1, and ProRes Proxy are available in MP4, MKV, or MOV where compatible, each in a plain CPU variant or an `(NVIDIA NVENC)` GPU variant (except ProRes Proxy, which is CPU-only).
- **Frame Interpolation:** single-video and reorderable batch-video processing with NVIDIA DLSS Frame Generation, selectable output rates from 23.976 to 120 FPS, and a three-second preview for a single selected video.
- **Sequential video batches:** Neural Rendering and Frame Interpolation batches run once, one by one, in the displayed uploader order using one settings snapshot. A failed item is recorded without preventing later videos from rendering. Successful outputs remain individually downloadable and every batch receives an ordered JSON manifest.
- **Single-only video previews:** input and output players are shown when exactly one video is selected. They are hidden for multi-video batches; batch results are provided through downloadable files and the per-video results table. The final-render player follows the Preview Encoding setting: Auto may show an H.264 derivative when the result itself is not browser-playable, while Disabled always shows the actual file.
- **Media preservation:** frame timestamps and display rotation are handled; original metadata and chapters are copied. MKV copies compatible audio and subtitle streams, while MP4/MOV convert audio to 192 kbps AAC. Frame Interpolation also preserves supported text subtitles in MP4/MOV.
- **GPU selection:** AI Processing and Video Processing GPUs can be selected separately. The AI GPU is used for DLSS Neural Rendering and Frame Generation; the Video GPU is used only for codecs suffixed `(NVIDIA NVENC)`, while plain H.264/H.265/AV1 and ProRes Proxy remain CPU-based. Automatic selection is available for both.
- **Preview Encoding:** a Settings-tab control for how in-app video previews are produced on the Video and Frame Interpolation tabs (preview buttons and final-render player). Auto uses the result directly when the browser can play it (MP4 + H.264, verified by probe) and otherwise creates an H.264 preview; Always H.264 always generates a browser-compatible preview; Disabled never creates one and sends the actual file to the browser. Non-H.264 previews can be slower and larger.
- **HDR Mode:** an opt-in 10-bit output that copies the input colorspace and keeps HDR when the input is HDR. It is available only for H.265, AV1, and ProRes Proxy; H.264 stays 8-bit SDR. Frame Interpolation accepts SDR video only unless HDR Mode is enabled.
- **GPU compatibility:** supported RTX GPUs are detected by architecture and compute capability, covering Ampere, Ada, and Blackwell. RTX 30 uses the tested experimental Ampere Neural Rendering path and may be significantly slower than RTX 40/50.
- **Renaming:** Image, Video, and Frame Interpolation outputs support Auto timestamped naming, Copy naming with the original base filename, or a Custom suffix. Existing outputs are never overwritten.
- **Safety and diagnostics:** only one GPU render runs at a time. Stop cancels the active worker/encoder and removes only incomplete output/job data. In a batch, completed files are retained and queued items are marked skipped. Outputs are accepted only after the relevant render path and output properties are verified.
- **Persistent controls:** Image and Video tabs share neural settings and the DLSS Model Preset, including the experimental Automatic Mask toggle. Frame Interpolation settings, HDR Mode selections, Preview Encoding, and GPU selections are also saved in `config.ini`. Settings presets can be exported to or imported from JSON, and Reset restores the relevant controls to their defaults.

The application creates `outputs/`, `logs/`, and `jobs/` when needed. Successful media is written to `outputs/`, reports and manifests to `logs/`, and temporary active-render data to `jobs/`.

## Requirements

- 64-bit Windows 11 with Direct3D 12.
- NVIDIA GeForce RTX GPU based on a supported Ampere, Ada, or Blackwell architecture. RTX 40/50 are the primary Neural Rendering targets; RTX 30/20 is enabled as a slower experimental path using the tested compatible runtime pair.
- Frame Interpolation is supported only on NVIDIA Ada and Blackwell GPUs (GeForce RTX 40 and RTX 50 series). It also requires a compatible NVIDIA driver with DLSS Frame Generation support and Hardware-accelerated GPU scheduling (HAGS) enabled in Windows.

## Settings

| Neural control | Values | Default |
| --- | --- | --- |
| NR Preset | Default, Preset #1, Preset #2, Preset #3 | Default |
| NR Style | Default, Natural, Cinematic | Default |
| NR Intensity | 0.00–2.00 | 1.00 |
| Local Tone Strength | 0.00–2.00 | 1.00 |
| Local Structure Strength | 0.00–2.00 | 1.00 |
| Skin Structure Strength | -1.00–2.00 | -1.00 (native default) |
| Automatic Mask | Off, On | Off |

The preset numbers are experimental native model hints; their visual effect is content-dependent.

| DLSS Model Preset | SDK value | Behavior |
| --- | ---: | --- |
| Default | 0 | Applies no override and lets NVIDIA choose its normal mode-specific preset |
| J | 10 | Forces preset J for every supported DLSS scaling mode |
| K | 11 | Forces preset K for every supported DLSS scaling mode |
| L | 12 | Forces preset L for every supported DLSS scaling mode |
| M | 13 | Forces preset M for every supported DLSS scaling mode |

The DLSS Model Preset is independent of the experimental Neural Rendering **NR Preset** control.

| Upscaling mode | Factor | Behavior |
| --- | ---: | --- |
| DLAA / native | 1× | Keeps the source dimensions |
| Quality | 1.5× | Produces 1.5× output dimensions |
| Balanced | 1.724× | Produces approximately 1.724× output dimensions |
| Performance | 2× | Produces 2× output dimensions |
| Ultra Performance | 3× | Produces 3× output dimensions |

Output dimensions are rounded to even pixels and limited to a 7680×4320 boundary.

| Frame Interpolation setting | Choices and behavior | Default |
| --- | --- | --- |
| Output FPS | 23.976, 25, 29.97, 30, 50, 59.94, 60, 90, or 120 FPS | 60 |
| DLSS engine | Auto, Native DLSSG, or Cascade | Auto |
| Video codec | H.264, H.265, AV1, or ProRes Proxy, each in a plain CPU variant or an `(NVIDIA NVENC)` GPU variant (ProRes Proxy is CPU-only) | H.264 |
| Container | MP4, MKV, or MOV; ProRes Proxy requires MKV or MOV | MP4 |
| Encoding quality | Auto (Default), Good, Best, or Max | Auto (Default) |
| HDR Mode | Off, On; 10-bit output that copies the input colorspace; H.265 / AV1 / ProRes only | Off |
| Rename | Auto adds a DLSSFG timestamp; Copy keeps the original base name; Custom appends the entered suffix | Auto |

`Auto` uses a supported exact native DLSSG grid when possible and the cascade path when required. If the selected output FPS is equal to or below the source rate, source frames are resampled without generating additional frames.

| Output setting | Choices and behavior |
| --- | --- |
| Image format | PNG/TIFF are lossless; JPEG/WebP/AVIF use the 1–100 quality control (default 95) |
| Video codec | H.264, H.265, AV1, or ProRes Proxy, each in a plain CPU variant or an `(NVIDIA NVENC)` GPU variant (ProRes Proxy is CPU-only) |
| Container | MP4, MKV, or MOV; ProRes Proxy requires MKV or MOV |
| Encoding quality | Auto (Default) uses resolution/FPS/codec; Good = Auto×2; Best = Auto×4; Max uses CQ/CRF 0; ProRes uses its fixed Proxy profile |
| HDR Mode | Off, On; 10-bit output that copies the input colorspace; H.265 / AV1 / ProRes only |
| Rename | Auto adds the DLSS5 timestamp; Copy keeps the original base name; Custom appends the entered suffix before the output extension |

Plain H.264/H.265/AV1 are CPU-based (libx264 / libx265 / libsvtav1) and never require a GPU. Only codecs suffixed `(NVIDIA NVENC)` need working NVENC support on the selected or automatically chosen Video Processing GPU at the requested output size. ProRes Proxy uses CPU-based 10-bit 4:2:2 encoding, although the verified Neural Rendering path remains RGBA8.

| Application setting | Choices and behavior |
| --- | --- |
| AI Processing GPU | Automatic or a compatible Ampere, Ada, or Blackwell RTX GPU; used for Neural Rendering and Frame Generation |
| Video Processing GPU | Automatic or an available NVIDIA GPU; used only for codecs suffixed `(NVIDIA NVENC)`, while plain H.264/H.265/AV1 and ProRes stay on CPU |
| Preview Encoding | Auto (default), Always H.264, or Disabled; controls how in-app video previews are produced on the Video and Frame Interpolation tabs |
| Settings preset | Export all adjustable Image, Video, Frame Interpolation, GPU, HDR Mode, and Preview Encoding settings to JSON, or import a preset to apply and save them |

Saved GPU selections use stable GPU identity. If a previously saved GPU is no longer available, that selection returns to Automatic rather than silently switching to another saved device.

## License and third-party notices

Original application code is licensed under the MIT License, copyright © 2026 Merserk. That license covers only original project code; it does not relicense or grant rights to any third-party software, model, binary, trademark, media, or other asset.

- **NVIDIA DLSS/NGX:** NVIDIA and its suppliers retain their rights in genuine NVIDIA SDK and runtime files, including DLSS Neural Rendering and DLSS Frame Generation components. Use and distribution are governed by the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt). Their presence in a portable package does not imply a standalone redistribution right, and this project must not be represented as NVIDIA-sponsored or endorsed.
- **FFmpeg:** the referenced Gyan.dev `9.0.1-full_build` was configured with GPL and version-3 components and is distributed under GPLv3. Its build information, license, and exact [corresponding FFmpeg source commit](https://github.com/FFmpeg/FFmpeg/commit/bf1b838f2a) are retained under `bin/ffmpeg/`. Anyone redistributing that binary must satisfy the applicable GPLv3 and corresponding-source obligations. See [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md).
- **ReShade:** copyright belongs to Patrick Mours and contributors; ReShade is available under the [BSD 3-Clause License](https://github.com/crosire/reshade).
- **RenoDX:** RenoDX core is copyright its authors and available under [MIT](https://github.com/clshortfuse/renodx/blob/main/LICENSE). This does not establish the license of the separate `renodx-dlss5.addon64` file.
- **Python and packages:** Python is provided under the [PSF License](https://docs.python.org/3.13/license.html). Gradio, Pillow, pillow-heif, rawpy, resvg-py, PyAV, OpenCV, NumPy, their transitive dependencies, and bundled codecs retain their own copyright and license terms; preserve the notices shipped with each distribution.

NVIDIA, GeForce RTX, NGX, and DLSS are trademarks and/or registered trademarks of NVIDIA Corporation. FFmpeg, ReShade, RenoDX, Python, and other names belong to their respective owners. Codec patent or other permissions may also be required depending on jurisdiction and use. Review the controlling licenses before building or distributing a complete package.
