# DLSS 5 Visual Enhancer

[![Downloads](https://img.shields.io/github/downloads/Merserk/dlss5-visual-enhancer/total.svg?style=flat-square&label=Downloads)](https://github.com/Merserk/dlss5-visual-enhancer/releases) [![Patreon](https://img.shields.io/badge/Patreon-MM744-F96854?style=flat-square&logo=patreon&logoColor=white)](https://www.patreon.com/MM744) ![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&logo=windows11&logoColor=white) ![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX-76B900?style=flat-square&logo=nvidia&logoColor=white) ![DLSS](https://img.shields.io/badge/DLSS-5-76B900?style=flat-square) ![Type](https://img.shields.io/badge/Type-Portable-2EA44F?style=flat-square) [![License](https://img.shields.io/badge/License-MIT-007EC6?style=flat-square)](LICENSE)

Windows application for applying a DLSS 5 Neural Rendering feature-18 pipeline to images and video through a local Gradio interface. It is an independent community project and is not affiliated with, sponsored by, or endorsed by NVIDIA, ReShade, RenoDX, FFmpeg, or their contributors.

<img width="1403" height="630" alt="image" src="https://github.com/user-attachments/assets/4530410a-af0f-42b5-9ba7-72106e1f5517" />

### Original

https://github.com/user-attachments/assets/f27f61a3-cad3-4278-af66-eb11c54600fb

### DLSS 5

https://github.com/user-attachments/assets/d91591a9-2df1-4b4b-b18f-bd4dff73d5bc

## Changes in this fork

This fork ([speedyrulz/dlss5-visual-enhancer](https://github.com/speedyrulz/dlss5-visual-enhancer)) adds multi-GPU support, a pipelined render loop, and several fixes on top of the upstream project:

- **GPU selection:** a GPU dropdown on both tabs lists every detected RTX card (with a Refresh button and an Auto default), persisted in `config.ini` as `gpu`. The native worker always binds the first NVIDIA Direct3D adapter, so a selection it cannot honor fails fast before any frames render, with instructions instead of a wrong-GPU render. Reports and status lines name the adapter the worker actually bound rather than the first card `nvidia-smi` lists.
- **Dual-GPU encode:** on systems with two RTX cards, NVENC encoding runs on the card the DLSS render is not using ("Use both GPUs" checkbox, `dual_gpu_encode` in `config.ini`). The offload is skipped automatically when the second card has under 2 GB of free VRAM, since an NVENC session that cannot allocate takes the whole render down mid-encode.
- **Pipelined rendering (~2x faster):** decoding plus optical-flow guide generation, worker submission, and the encoder handoff now run in separate pipeline stages, so the CPU work and pipe transfers happen while the GPU evaluates the current frame. Measured on a 1836x3264 DLAA render: 2.86 to 5.9 fps, which is within 2% of the native worker's maximum throughput. Output is unchanged; the DLSS stream itself remains strictly sequential and in order.
- **Variable-frame-rate fix:** VFR videos (typical phone footage) previously crashed at a content-dependent frame with `Invalid argument ... returned 22`. The intermediate stream quantized timestamps to average-frame-duration ticks, so two close frames could collide into one tick and the muxer rejected the duplicate. The encoder context now keeps the source's fine time base, preserving exact VFR timing; a safety guard bumps genuinely duplicated timestamps and reports the count as `pts_collisions_adjusted`.
- **Better failure diagnostics:** if the video encoder process dies, the error names the encoder, exit code, and frame, and includes FFmpeg's stderr; failure reports gain an `encoder_log` section.

## Installation

1. Download the [latest release](https://github.com/Merserk/dlss5-visual-enhancer/releases/latest).
2. Unpack the downloaded ZIP archive.
3. Run `start.bat`.

## Main features

- **Images:** single-image and batch processing with per-file success/failure results, responsive input/output previews, individual downloads, a ZIP of successful outputs, batch manifests, and diagnostic JSON reports.
- **Image formats:** common Pillow formats plus HEIF/HEIC, SVG, and many camera RAW formats. Outputs are PNG, JPEG, WebP, AVIF, or TIFF.
- **Image handling:** EXIF orientation is applied; ICC input is converted to sRGB; supported EXIF, DPI, and XMP metadata is retained; alpha is preserved except when JPEG composites it over white. Animated and multipage files use only the first frame/page.
- **Video:** whole-video rendering, a one-frame preview, and a three-second preview. H.264, HEVC, AV1, and ProRes Proxy are available in MP4, MKV, or MOV where compatible.
- **Media preservation:** frame timestamps and display rotation are handled; original metadata and chapters are copied. MKV copies compatible audio and subtitle streams, while MP4/MOV convert audio to 192 kbps AAC. Final muxing preserves every rendered video frame when source audio is slightly shorter.
- **Safety and diagnostics:** only one GPU render runs at a time. Stop cancels the active worker/encoder, incomplete output and job data are removed, and outputs are accepted only after feature-18 execution and output dimensions/frame counts are verified.
- **Persistent controls:** Image and Video tabs share neural settings, including the experimental Automatic Mask toggle, saved in `config.ini`; Reset restores every setting to its default.

Outputs are written to `outputs/`, reports and manifests to `logs/`, and temporary data to `jobs/` while a render is active.

## Requirements

- 64-bit Windows 11 with Direct3D 12.
- NVIDIA GeForce RTX GPU. RTX 40/50 are the primary targets; RTX 30 is enabled as a slower beta path. RTX 20 and non-RTX GPUs are rejected.

## How processing works

1. The input is decoded to 8-bit SDR RGBA and its orientation and dimensions are normalized.
2. The selected fixed DLSS mode determines the output size; the native worker negotiates its required render size.
3. Images send zero motion with a history reset. Video uses OpenCV DIS optical flow for current-to-previous motion, with resets on the first frame and detected scene changes.
4. Frames are streamed to the project-specific D3D12 worker, which hosts the ReShade/RenoDX DLSS feature-18 path.
5. Alpha is restored for images. Video frames are sent to FFmpeg with source presentation timestamps, then audio, metadata, chapters, and compatible subtitles are muxed.
6. ReShade evidence and output properties are verified. Unverified or incomplete results are deleted; successful renders receive JSON reports containing settings, dimensions, GPU/encoder data, hashes, logs, and feature evidence.

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

| Upscaling mode | Factor | Behavior |
| --- | ---: | --- |
| DLAA / native | 1× | Keeps the source dimensions |
| Quality | 1.5× | Produces 1.5× output dimensions |
| Balanced | 1.724× | Produces approximately 1.724× output dimensions |
| Performance | 2× | Produces 2× output dimensions |
| Ultra Performance | 3× | Produces 3× output dimensions |

Output dimensions are rounded to even pixels and limited to a 7680×4320 boundary.

| Output setting | Choices and behavior |
| --- | --- |
| Image format | PNG/TIFF are lossless; JPEG/WebP/AVIF use the 1–100 quality control (default 95) |
| Video codec | H.264, HEVC, AV1, or ProRes Proxy |
| Container | MP4, MKV, or MOV; ProRes Proxy requires MKV or MOV |
| Encoding quality | Auto (Default) uses resolution/FPS/codec; Good = Auto×2; Best = Auto×4; Max uses CQ/CRF 0; ProRes uses its fixed Proxy profile |

H.264 and HEVC prefer NVENC and fall back to slow software encoding. AV1 requires working AV1 NVENC at the selected output size. ProRes Proxy uses 10-bit 4:2:2 encoding, although the verified neural-rendering path remains RGBA8.

## Required external files

The placeholder documents under `bin/` describe the omitted layout. Restore files only from sources you are authorized to use; filenames alone do not establish authenticity or redistribution rights.

| Expected path | Purpose and ownership |
| --- | --- |
| `bin/python-3.13.15-embed-amd64/` | Python 3.13 portable runtime and packages; omitted and replaced by [BINARIES.md](bin/python-3.13.15-embed-amd64/BINARIES.md) |
| `bin/ffmpeg/bin/ffmpeg.exe`, `ffprobe.exe` | FFmpeg processing and probing tools; omitted and documented in [BINARIES.md](bin/ffmpeg/BINARIES.md) |
| `bin/runtime/nvngx.dll` | Project-specific standalone D3D12 worker, named for its caller contract; it is **not** NVIDIA's NGX core DLL |
| `bin/runtime/dxgi.dll` | ReShade carrier with add-on support |
| `bin/runtime/renodx-dlss5.addon64` | Third-party DLSS 5 Neural Rendering add-on; its specific distribution license must be verified separately |
| `bin/runtime/nvngx_dlss.dll`, `nvngx_dlssnr.dll` | DLSS/NGX runtime and neural-rendering components; NVIDIA proprietary terms apply to genuine NVIDIA SDK files |
| `bin/runtime/ReShade.ini` | Local ReShade configuration used by the runtime layout |

See [the complete runtime inventory](bin/runtime/BINARIES.md). Do not obtain or redistribute proprietary or closed-source files through unauthorized mirrors.

## License and third-party notices

Original application code in this repository is licensed under the [MIT License](LICENSE), copyright © 2026 Merserk. That license covers only original project code; it does not relicense or grant rights to any third-party software, model, binary, trademark, media, or other asset.

- **NVIDIA DLSS/NGX:** NVIDIA and its suppliers retain their rights in genuine NVIDIA SDK files. Use and distribution are governed by the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt). The files are not included here, no standalone redistribution right is implied, and this project must not be represented as NVIDIA-sponsored or endorsed.
- **FFmpeg:** the referenced Gyan.dev `9.0.1-full_build` was configured with GPL and version-3 components and is distributed under GPLv3. Its build information, license, and exact [corresponding FFmpeg source commit](https://github.com/FFmpeg/FFmpeg/commit/bf1b838f2a) are retained under `bin/ffmpeg/`. Anyone redistributing that binary must satisfy the applicable GPLv3 and corresponding-source obligations. See [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md).
- **ReShade:** copyright belongs to Patrick Mours and contributors; ReShade is available under the [BSD 3-Clause License](https://github.com/crosire/reshade).
- **RenoDX:** RenoDX core is copyright its authors and available under [MIT](https://github.com/clshortfuse/renodx/blob/main/LICENSE). This does not establish the license of the separate `renodx-dlss5.addon64` file.
- **Python and packages:** Python is provided under the [PSF License](https://docs.python.org/3.13/license.html). Gradio, Pillow, pillow-heif, rawpy, resvg-py, PyAV, OpenCV, NumPy, their transitive dependencies, and bundled codecs retain their own copyright and license terms; preserve the notices shipped with each distribution.

NVIDIA, GeForce RTX, NGX, and DLSS are trademarks and/or registered trademarks of NVIDIA Corporation. FFmpeg, ReShade, RenoDX, Python, and other names belong to their respective owners. Codec patent or other permissions may also be required depending on jurisdiction and use. Review the controlling licenses before building or distributing a complete package.
