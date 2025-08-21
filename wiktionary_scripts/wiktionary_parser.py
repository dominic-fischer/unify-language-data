import sys, os, gzip, bz2, lzma, re, json
from pathlib import Path
from collections import OrderedDict, defaultdict

# Try fast JSON
try:
    import orjson as _json
    def jloads(s): return _json.loads(s)
    def jdumps(o): return _json.dumps(o)  # -> bytes
    BYTES = True
except ImportError:
    import json as _json
    def jloads(s): return _json.loads(s)
    def jdumps(o): return _json.dumps(o, ensure_ascii=False).encode("utf-8")
    BYTES = True

def _open_any(path, mode="rt", encoding="utf-8"):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, mode, encoding=encoding) if "t" in mode else gzip.open(p, mode)
    if p.endswith(".bz2"):
        return bz2.open(p, mode, encoding=encoding) if "t" in mode else bz2.open(p, mode)
    if p.endswith(".xz"):
        return lzma.open(p, mode, encoding=encoding) if "t" in mode else lzma.open(p, mode)
    return open(p, mode, encoding=encoding) if "t" in mode else open(p, mode)

def _sanitize(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)
    return name[:120]

def _get_nested(obj, dotted_key: str):
    # supports "lang" or "meta.lang" etc.
    cur = obj
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

class LRUWriterCache:
    def __init__(self, capacity: int, out_dir: Path, suffix: str = ".jsonl", compress: bool = False):
        self.capacity = capacity
        self.out_dir = out_dir
        self.suffix = suffix if not compress else suffix + ".gz"
        self.compress = compress
        self.cache = OrderedDict()  # key -> file handle (binary)

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if len(self.cache) >= self.capacity:
            old_key, fh = self.cache.popitem(last=False)
            try: fh.close()
            except: pass
        filename = f"{_sanitize(key)}{self.suffix}"
        path = self.out_dir / filename
        if self.compress:
            fh = gzip.open(path, "ab", compresslevel=6)
        else:
            fh = open(path, "ab", buffering=1024*1024)
        self.cache[key] = fh
        return fh

    def close_all(self):
        for _, fh in self.cache.items():
            try: fh.close()
            except: pass
        self.cache.clear()

def split_jsonl_by_lang(
    in_path: str,
    out_dir: str,
    lang_key: str = "lang",          # supports dotted path, e.g. "meta.lang"
    targets: set | None = None,      # e.g., {"Chichewa","German","English"}; None = all langs
    case_insensitive: bool = False,
    max_open_files: int = 64,
    report_every: int = 1_000_000,
    gzip_outputs: bool = False,
    manifest_name: str = "_manifest.json",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    writer_cache = LRUWriterCache(max_open_files, Path(out_dir), suffix=".jsonl", compress=gzip_outputs)

    def norm(x): return x.casefold() if case_insensitive and isinstance(x, str) else x
    targets_norm = {norm(t) for t in targets} if targets else None

    seen = kept = bad = 0
    per_lang_counts = defaultdict(int)

    try:
        with _open_any(in_path, "rt", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                seen += 1
                try:
                    obj = jloads(line)
                except Exception:
                    bad += 1
                    continue

                lang_val = _get_nested(obj, lang_key)
                if lang_val is None:
                    continue

                lang_cmp = norm(lang_val)
                if targets_norm is not None and lang_cmp not in targets_norm:
                    continue

                fh = writer_cache.get(str(lang_val))
                fh.write(jdumps(obj) + b"\n")
                kept += 1
                per_lang_counts[str(lang_val)] += 1

                if report_every and seen % report_every == 0:
                    print(f"[progress] processed={seen:,} kept={kept:,} bad_lines={bad:,}", file=sys.stderr)
    finally:
        writer_cache.close_all()

    # Write a small manifest for auditing
    manifest = {
        "input": in_path,
        "out_dir": out_dir,
        "lang_key": lang_key,
        "targets": sorted(list(targets)) if targets else None,
        "case_insensitive": case_insensitive,
        "processed": seen,
        "kept": kept,
        "bad_lines": bad,
        "per_language_counts": dict(sorted(per_lang_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "compressed_outputs": gzip_outputs,
    }
    with open(Path(out_dir) / manifest_name, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print(f"Done. processed={seen:,} kept={kept:,} bad_lines={bad:,} -> '{out_dir}'")
    print(f"Per-language counts written to {Path(out_dir)/manifest_name}")

# Example
if __name__ == "__main__":
    split_jsonl_by_lang(
        in_path="wiktionary_data/raw-wiktextract-data.jsonl",
        out_dir="wiktionary_files/by_lang",
        lang_key="lang",                          # or "meta.lang" if nested
        targets={"Chichewa", "Shona", "Swahili", "Zulu", "French", "Romanian", "Italian", "Portuguese", "Spanish"},# or None to split ALL langs
        case_insensitive=False,
        max_open_files=64,
        gzip_outputs=True,                        # turn on to save space
    )
