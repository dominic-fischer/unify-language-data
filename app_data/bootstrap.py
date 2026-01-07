from __future__ import annotations

import zipfile
from pathlib import Path

import requests
import streamlit as st


@st.cache_resource
def ensure_zip_extracted(zip_url: str, target_dir: Path) -> Path:
    """
    Download zip_url into target_dir and extract it there once.
    Uses a '.ready' marker file so it won't repeat.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".ready"
    if marker.exists():
        return target_dir

    zip_path = target_dir / "data.zip"
    if not zip_path.exists():
        r = requests.get(zip_url, stream=True, timeout=300)
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    marker.write_text("ok", encoding="utf-8")
    return target_dir
