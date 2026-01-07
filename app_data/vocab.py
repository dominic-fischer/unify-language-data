from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

from config import VOCAB_DIR, VOCAB_SUPPORTED
from app_data.files import iter_files

# ----------------------------
# Vocab: streaming JSONL/JSONL.GZ
# ----------------------------


def iter_jsonl_records(path: Path) -> Iterable[dict]:
    """Yield dict records from .jsonl or .jsonl.gz without loading whole file into memory."""
    if path.name.lower().endswith(".jsonl.gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = path.open("r", encoding="utf-8")

    with f:
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


def vocab_record_categories(entry: dict) -> List[str]:
    """Collect categories from entry.senses[].categories (if present)."""
    cats: List[str] = []
    senses = entry.get("senses") or []
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict):
                sc = s.get("categories") or []
                if isinstance(sc, list):
                    for c in sc:
                        if isinstance(c, str):
                            cats.append(c)

    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for c in cats:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def vocab_record_first_gloss(entry: dict) -> str:
    senses = entry.get("senses") or []
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict):
                glosses = s.get("glosses") or []
                if isinstance(glosses, list) and glosses:
                    g0 = glosses[0]
                    if isinstance(g0, str):
                        return g0
    return ""


def vocab_record_etymology(entry: dict) -> str:
    """Try common keys; best-effort summary (full record still stored in details_json)."""
    for k in ["etymology_text", "etymology", "etymologies"]:
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return " | ".join(v)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            txts = []
            for d in v:
                if isinstance(d, dict):
                    t = d.get("text") or d.get("etymology_text")
                    if isinstance(t, str) and t.strip():
                        txts.append(t.strip())
            if txts:
                return " | ".join(txts)
    return ""


def vocab_record_meanings(entry: dict, max_glosses: int = 1) -> list[str]:
    """Return up to N gloss strings from senses[].glosses[] (in order)."""
    out: list[str] = []
    senses = entry.get("senses") or []
    if not isinstance(senses, list):
        return out

    for s in senses:
        if not isinstance(s, dict):
            continue
        glosses = s.get("glosses") or []
        if not isinstance(glosses, list):
            continue
        for g in glosses:
            if isinstance(g, str) and g.strip():
                out.append(g.strip())
                if len(out) >= max_glosses:
                    return out
    return out


# ----------------------------
# Language -> file mapping (per-language files)
# ----------------------------

def _normalize_lang_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z]+", "", s)
    return s


# UI uses "chewa" but file is "Chichewa.jsonl.gz"
LANG_ALIASES = {
    "chewa": "chichewa",
}


@st.cache_data(show_spinner=False)
def list_language_files() -> Dict[str, Path]:
    """
    Build mapping from normalized language name -> file path,
    using filename stem. Example: 'french' -> VOCAB_DIR/French.jsonl.gz
    """
    files = iter_files(VOCAB_DIR, recursive=False, exts=VOCAB_SUPPORTED)
    m: Dict[str, Path] = {}
    for fp in files:
        name = fp.name
        if name.lower().endswith(".jsonl.gz"):
            stem = name[: -len(".jsonl.gz")]
        elif name.lower().endswith(".jsonl"):
            stem = name[: -len(".jsonl")]
        else:
            continue
        key = _normalize_lang_key(stem)
        if key:
            m[key] = fp
    return m


def resolve_lang_files(langs: Tuple[str, ...]) -> List[Path]:
    """Resolve user-selected languages to actual per-language files."""
    m = list_language_files()
    out: List[Path] = []
    for lang in langs:
        k = _normalize_lang_key(lang)
        k = LANG_ALIASES.get(k, k)
        fp = m.get(k)
        if fp is not None:
            out.append(fp)
    return out


# ----------------------------
# POS + Category options (scanned per language file, cached)
# ----------------------------

@st.cache_data(show_spinner=True)
def scan_pos_to_categories_for_file(
    file_path: str,
    max_records: int = 80_000,
    max_unique_categories_total: int = 6000,
    max_unique_categories_per_pos: int = 2500,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Scan one language file and return:
      - list of POS present
      - mapping POS -> list of categories present in that POS

    The caps keep this fast and prevent runaway memory.
    """
    fp = Path(file_path)

    pos_set: set[str] = set()
    pos_to_cats: Dict[str, set[str]] = {}

    # track category counts to cap per POS and overall
    total_cats: set[str] = set()

    n = 0
    for e in iter_jsonl_records(fp):
        n += 1
        if n > max_records:
            break

        pos = e.get("pos")
        if isinstance(pos, str) and pos:
            pos_set.add(pos)
        else:
            continue  # categories are POS-scoped for our UI, skip unknown POS

        if len(total_cats) >= max_unique_categories_total:
            continue

        cats = vocab_record_categories(e)
        if not cats:
            continue

        bucket = pos_to_cats.setdefault(pos, set())
        if len(bucket) >= max_unique_categories_per_pos:
            continue

        for c in cats:
            if not isinstance(c, str) or not c:
                continue
            if len(total_cats) >= max_unique_categories_total:
                break
            if len(bucket) >= max_unique_categories_per_pos:
                break
            bucket.add(c)
            total_cats.add(c)

    pos_list = sorted(pos_set)
    pos_to_cat_list = {p: sorted(cs) for p, cs in pos_to_cats.items()}
    return pos_list, pos_to_cat_list


def _combine_pos_lists(pos_lists: List[List[str]], intersect: bool) -> List[str]:
    sets = [set(xs) for xs in pos_lists if xs]
    if not sets:
        return []
    return sorted(set.intersection(*sets) if intersect else set.union(*sets))


def _combine_category_maps(
    maps: List[Dict[str, List[str]]],
    pos_filter: str | None,
    intersect: bool,
) -> List[str]:
    """
    Combine categories for a given POS across languages.
    - If pos_filter is None: return [] (UI should prompt user to choose POS first
      if they want POS-scoped categories).
    - If intersect=True: only categories present in ALL languages for that POS.
    """
    if not pos_filter:
        return []

    cat_sets: List[set[str]] = []
    for m in maps:
        cats = m.get(pos_filter) or []
        cat_sets.append(set(cats))

    if not cat_sets:
        return []

    if intersect:
        out = set.intersection(*cat_sets) if all(cat_sets) else set()
    else:
        out = set.union(*cat_sets)

    return sorted(out)


@st.cache_data(show_spinner=True)
def get_pos_options_for_langs(langs: Tuple[str, ...]) -> List[str]:
    """
    POS options for selected languages.
    If exactly 2 languages: intersection of POS sets.
    Else: union.
    """
    files = resolve_lang_files(langs)
    if not files:
        return []

    pos_lists: List[List[str]] = []
    for fp in files:
        pos_list, _ = scan_pos_to_categories_for_file(str(fp))
        pos_lists.append(pos_list)

    intersect = len(files) == 2
    return _combine_pos_lists(pos_lists, intersect=intersect)


@st.cache_data(show_spinner=True)
def get_category_options_for_langs_and_pos(langs: Tuple[str, ...], pos: str | None) -> List[str]:
    """
    Category options are POS-scoped:
      A) If a POS is selected, show only categories present in that POS.
      B) If exactly 2 languages, show the INTERSECTION of categories in that POS.
         (i.e., only categories present in both languages for that POS)
      Else: union.
    """
    if not pos:
        return []

    files = resolve_lang_files(langs)
    if not files:
        return []

    maps: List[Dict[str, List[str]]] = []
    for fp in files:
        _, pos_to_cats = scan_pos_to_categories_for_file(str(fp))
        maps.append(pos_to_cats)

    intersect = len(files) == 2
    return _combine_category_maps(maps, pos_filter=pos, intersect=intersect)


# ----------------------------
# User-driven loading: scan only selected language files, stop early at limit
# ----------------------------

@st.cache_data(show_spinner=True)
def load_vocab_entries_filtered_from_lang_files(
    langs: Tuple[str, ...],
    pos: str | None,
    cat: str | None,
    word_query: str | None,
    limit_rows: int,
) -> pd.DataFrame:
    """
    Scan ONLY the per-language files for langs, collect matching rows, stop at limit_rows.
    """
    files = resolve_lang_files(langs)
    q = (word_query or "").strip().lower()

    rows: List[dict] = []

    for fp in files:
        for e in iter_jsonl_records(fp):
            if pos and e.get("pos") != pos:
                continue
            if cat:
                cats = vocab_record_categories(e)
                if cat not in cats:
                    continue

            word = e.get("word") or e.get("lemma") or e.get("_key") or ""
            if q and isinstance(word, str):
                if q not in word.lower():
                    continue

            rows.append(
                {
                    "lang": e.get("lang"),
                    "pos": e.get("pos"),
                    "word": word,
                    "first_gloss": vocab_record_first_gloss(e),
                    "categories": " | ".join(vocab_record_categories(e)),
                    "etymology": vocab_record_etymology(e),
                    "n_senses": len(e.get("senses") or []) if isinstance(e.get("senses"), list) else 0,
                    "details_json": json.dumps(e, ensure_ascii=False),
                }
            )

            if len(rows) >= limit_rows:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)


# ----------------------------
# Meaning alignment (two languages)
# ----------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+")
_STOP = {
    "a", "an", "the", "to", "of", "and", "or", "in", "on", "for", "with", "from", "by", "at", "as",
    "be", "is", "are", "was", "were", "it", "this", "that", "these", "those",
}


def _norm_gloss(gloss: str) -> set[str]:
    toks = [t.lower() for t in _WORD_RE.findall(gloss)]
    return {t for t in toks if t and t not in _STOP}


def gloss_similarity(g1: str, g2: str) -> float:
    a = _norm_gloss(g1)
    b = _norm_gloss(g2)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def align_meanings_two_langs(
    rows_lang1: list[tuple[str, str]],  # (word, gloss)
    rows_lang2: list[tuple[str, str]],  # (word, gloss)
    threshold: float = 0.35,
) -> list[dict]:
    """
    Greedy alignment of glosses (meaning anchors) between two languages.
    Returns rows: {meaning, lang1, lang2, _sim}
    """
    used2 = set()
    out: list[dict] = []

    for w1, g1 in rows_lang1:
        best_j = -1
        best_sim = 0.0
        best_w2 = ""

        for j, (w2, g2) in enumerate(rows_lang2):
            if j in used2:
                continue
            sim = gloss_similarity(g1, g2)
            if sim > best_sim:
                best_sim = sim
                best_j = j
                best_w2 = w2

        if best_sim >= threshold and best_j >= 0:
            used2.add(best_j)
            out.append({"meaning": g1, "lang1": w1, "lang2": best_w2, "_sim": best_sim})
        else:
            out.append({"meaning": g1, "lang1": w1, "lang2": "", "_sim": 0.0})

    for j, (w2, g2) in enumerate(rows_lang2):
        if j in used2:
            continue
        out.append({"meaning": g2, "lang1": "", "lang2": w2, "_sim": 0.0})

    return out
