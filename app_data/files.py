from __future__ import annotations

from pathlib import Path
from typing import List


def iter_files(root: Path, recursive: bool, exts: set[str]) -> List[Path]:
    if not root.exists():
        return []
    globber = root.rglob("*") if recursive else root.glob("*")
    out: List[Path] = []
    for p in globber:
        if not p.is_file():
            continue
        # special-case .jsonl.gz
        if p.name.lower().endswith(".jsonl.gz") and ".jsonl.gz" in exts:
            out.append(p)
        elif p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)
