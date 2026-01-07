from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

# Adjust if needed
IN_DIR = Path("data/wiktionary/outfiles")
OUT_DIR = Path("data/wiktionary/outfiles_pos")

# Safety knobs
FLUSH_EVERY = 2000  # lines per pos to buffer before writing
UNKNOWN_POS = "_UNKNOWN"


def iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def safe_name(s: str) -> str:
    # filesystem-safe POS label
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-"})


def split_one_language_file(src: Path, out_dir: Path) -> None:
    lang = src.name.replace(".jsonl.gz", "").replace(".jsonl", "")
    lang_safe = safe_name(lang)

    out_dir.mkdir(parents=True, exist_ok=True)

    # buffers[pos] = list[str lines]
    buffers: Dict[str, list[str]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)

    def flush_pos(pos_key: str) -> None:
        if not buffers[pos_key]:
            return
        out_path = out_dir / f"{lang_safe}__{pos_key}.jsonl"
        with out_path.open("a", encoding="utf-8") as out:
            out.writelines(buffers[pos_key])
        buffers[pos_key].clear()

    # ensure clean reruns: remove existing split files for this language
    for existing in out_dir.glob(f"{lang_safe}__*.jsonl"):
        existing.unlink()

    for entry in iter_jsonl_gz(src):
        pos = entry.get("pos")
        pos_key = safe_name(pos) if isinstance(pos, str) and pos.strip() else UNKNOWN_POS

        line = json.dumps(entry, ensure_ascii=False) + "\n"
        buffers[pos_key].append(line)
        counts[pos_key] += 1

        if len(buffers[pos_key]) >= FLUSH_EVERY:
            flush_pos(pos_key)

    # final flush
    for pos_key in list(buffers.keys()):
        flush_pos(pos_key)

    # write meta (counts only; you can extend later with categories)
    meta_path = out_dir / f"{lang_safe}__meta.json"
    meta = {
        "language_file": src.name,
        "language": lang,
        "pos_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {src.name}: wrote {len(counts)} POS files into {out_dir}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = sorted(IN_DIR.glob("*.jsonl.gz"))
    if not inputs:
        raise SystemExit(f"No *.jsonl.gz files found in {IN_DIR.resolve()}")

    for src in inputs:
        split_one_language_file(src, OUT_DIR)


if __name__ == "__main__":
    main()
