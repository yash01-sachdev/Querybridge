"""Helpers for safe temporary runtime directories in restricted environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def iter_temp_root_candidates() -> tuple[Path, ...]:
    """Return temp-root candidates without triggering tempfile's cwd probe fallback."""

    raw_candidates: list[str] = []

    for value in (os.getenv("TMPDIR"), os.getenv("TEMP"), os.getenv("TMP")):
        if value:
            raw_candidates.append(value)

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        raw_candidates.append(str(Path(local_app_data) / "Temp"))

    user_profile = os.getenv("USERPROFILE")
    if user_profile:
        raw_candidates.append(str(Path(user_profile) / "AppData" / "Local" / "Temp"))

    windows_dir = os.getenv("WINDIR")
    if windows_dir:
        raw_candidates.append(str(Path(windows_dir) / "Temp"))

    raw_candidates.extend(["C:\\temp", "C:\\tmp"])

    deduped_candidates: list[Path] = []
    seen: set[str] = set()
    for raw_candidate in raw_candidates:
        normalized = os.path.normcase(raw_candidate)
        if normalized in seen:
            continue

        seen.add(normalized)
        deduped_candidates.append(Path(raw_candidate))

    return tuple(deduped_candidates)


def pick_first_writable_directory(candidates: Iterable[Path]) -> Path | None:
    """Return the first directory that can be created for runtime scratch files."""

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue

        return candidate

    return None
