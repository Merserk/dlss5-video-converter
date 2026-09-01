from __future__ import annotations

from pathlib import Path


RENAME_MODES = ("Auto", "Copy", "Custom")
_INVALID_FILENAME_CHARACTERS = set('<>:"/\\|?*')


def validate_rename(mode: str, custom_suffix: str) -> str:
    if mode not in RENAME_MODES:
        choices = ", ".join(RENAME_MODES)
        raise ValueError(f"Rename must be one of: {choices}.")
    suffix = str(custom_suffix or "")
    if mode != "Custom":
        return suffix
    if not suffix:
        raise ValueError("Enter a suffix when Rename is set to Custom.")
    if any(character in _INVALID_FILENAME_CHARACTERS or ord(character) < 32 for character in suffix):
        raise ValueError('Custom suffix cannot contain < > : " / \\ | ? * or control characters.')
    if suffix.endswith((" ", ".")):
        raise ValueError("Custom suffix cannot end with a space or period.")
    return suffix


def output_filename(
    source: Path,
    extension: str,
    mode: str,
    custom_suffix: str,
    auto_stem: str,
) -> str:
    suffix = validate_rename(mode, custom_suffix)
    if not extension.startswith("."):
        raise ValueError("Output extension must start with a period.")
    if mode == "Auto":
        stem = auto_stem
    else:
        stem = source.stem.strip().rstrip(".") or "output"
        if mode == "Custom":
            stem += suffix
    return f"{stem}{extension}"


def require_available_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path.name}. Rename or remove the existing file first."
        )
