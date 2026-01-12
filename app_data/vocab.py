from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

import os
from config import VOCAB_SUPPORTED

# Cache dir on Streamlit Cloud (writable). Can be overridden by env var if you want.
APP_CACHE_DIR = Path(os.environ.get("APP_CACHE_DIR", str(Path.home() / ".cache" / "unify-language-data")))
VOCAB_DIR = APP_CACHE_DIR / "data" / "wiktionary" / "outfiles"
VOCAB_POS_DIR = APP_CACHE_DIR / "data" / "wiktionary" / "outfiles_pos"

from app_data.files import iter_files

# If you created: data/wiktionary/outfiles_pos/
# and VOCAB_DIR is: data/wiktionary/outfiles/
#VOCAB_POS_DIR = VOCAB_DIR.parent / "outfiles_pos"

# ----------------------------
# JSONL readers
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


# ----------------------------
# Record helpers
# ----------------------------

# fr:Sciences / It:Sciences / de:Foo
_LANGCODE_PREFIX = re.compile(r"^[A-Za-z]{2,3}:\s*")

# English language names you want stripped
_LANGUAGE_NAMES = [
    "French",
    "Italian",
    "Romanian",
    "Spanish",
    "Portuguese",
    "Shona",
    "Chichewa",
    "Swahili",
    "Zulu",
]

_LANGUAGE_RE = re.compile(
    r"^(?:" + "|".join(re.escape(l) for l in _LANGUAGE_NAMES) + r")\s+",
    re.IGNORECASE,
)

def normalize_category(cat: str) -> str | None:
    c = (cat or "").strip()
    if not c:
        return None

    # Drop "Category:" prefix
    if c.lower().startswith("category:"):
        c = c.split(":", 1)[1].strip()

    # A) remove unwanted buckets
    if c.startswith("Pages with") or c.startswith("entries with incorrect"):
        return None
    for lang in _LANGUAGE_NAMES:
        if c.startswith(f"{lang} with"):
            return None
        if c.startswith(f"{lang} entries with incorrect"):
            return None

    # B1) remove language code prefixes (fr:, it:, etc.)
    c = _LANGCODE_PREFIX.sub("", c).strip()

    # B2) remove leading language names ("Italian doublets")
    c = _LANGUAGE_RE.sub("", c).strip()

    return c or None

def vocab_record_categories(entry: dict) -> List[str]:
    """Collect *normalized* categories from entry.senses[].categories."""
    cats: List[str] = []

    senses = entry.get("senses") or []
    if isinstance(senses, list):
        for s in senses:
            if isinstance(s, dict):
                sc = s.get("categories") or []
                if isinstance(sc, list):
                    for c in sc:
                        if isinstance(c, str):
                            nc = normalize_category(c)
                            if nc:
                                cats.append(nc)

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
# Language <-> file mapping (per-language source files)
# ----------------------------


def _normalize_lang_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z]+", "", s)
    return s


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-"})


def _pos_key(pos: str) -> str:
    # Must match your splitter's POS sanitization (close enough for POS labels)
    return _safe_name(pos)


# UI uses "chewa" but file is "Chichewa.jsonl.gz"
LANG_ALIASES = {
    "chewa": "chichewa",
}


@st.cache_data(show_spinner=False)
def list_language_files() -> Dict[str, Path]:
    """
    Map normalized language -> a representative file.

    Priority:
      1) outfiles/ per-language source files (French.jsonl.gz)
      2) outfiles_pos/ split files (French__Adjective.jsonl), if outfiles/ not present
    """
    m: Dict[str, Path] = {}

    # 1) Prefer original per-language source files if available
    if VOCAB_DIR.exists():
        files = iter_files(VOCAB_DIR, recursive=False, exts=VOCAB_SUPPORTED)
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
        if m:
            return m

    # 2) Fallback: infer languages from split files in outfiles_pos/
    if VOCAB_POS_DIR.exists():
        for fp in VOCAB_POS_DIR.glob("*.jsonl"):
            # Expect: <LangSafe>__<POS>.jsonl
            name = fp.name
            if "__" not in name:
                continue
            lang_part = name.split("__", 1)[0]
            # lang_part is "LangSafe" (e.g., French)
            key = _normalize_lang_key(lang_part)
            if key and key not in m:
                m[key] = fp

    return m


def resolve_lang_files(langs: Tuple[str, ...]) -> List[Path]:
    """
    Resolve user-selected languages to:
      - original per-language files if available
      - otherwise, any representative split file for that language
    """
    m = list_language_files()
    out: List[Path] = []
    for lang in langs:
        k = _normalize_lang_key(lang)
        k = LANG_ALIASES.get(k, k)
        fp = m.get(k)
        if fp is not None:
            out.append(fp)
    return out


def _lang_safe_from_source_file(source_fp: Path) -> str:
    """
    Your POS split uses the language filename stem (e.g., 'French') turned into safe_name.
    """
    name = source_fp.name
    if name.lower().endswith(".jsonl.gz"):
        stem = name[: -len(".jsonl.gz")]
    elif name.lower().endswith(".jsonl"):
        stem = name[: -len(".jsonl")]
    else:
        stem = name
    return _safe_name(stem)


def resolve_lang_pos_files(langs: Tuple[str, ...], pos: str | None) -> List[Path]:
    """
    Prefer POS-split .jsonl files if:
      - pos is given
      - the split file exists for EVERY selected language
    Otherwise fall back to original per-language files.
    """
    if not pos:
        return resolve_lang_files(langs)

    if not VOCAB_POS_DIR.exists():
        return resolve_lang_files(langs)

    pk = _pos_key(pos)
    src_map = list_language_files()

    out: List[Path] = []
    for lang in langs:
        lk = _normalize_lang_key(lang)
        lk = LANG_ALIASES.get(lk, lk)
        src = src_map.get(lk)
        if src is None:
            return resolve_lang_files(langs)

        lang_safe = _lang_safe_from_source_file(src)
        split_fp = VOCAB_POS_DIR / f"{lang_safe}__{pk}.jsonl"
        if not split_fp.exists():
            return resolve_lang_files(langs)
        out.append(split_fp)

    return out


# ----------------------------
# POS + category options (fast with split files)
# ----------------------------

@st.cache_data(show_spinner=True)
def list_pos_keys_for_language(lang: str) -> List[str]:
    """
    Discover POS keys for one language by looking at split files:
      <LangSafe>__<POS>.jsonl

    Returns POS keys (already sanitized), sorted.
    Falls back to scanning original file if split dir missing.
    """
    if VOCAB_POS_DIR.exists():
        src_map = list_language_files()
        lk = _normalize_lang_key(lang)
        lk = LANG_ALIASES.get(lk, lk)
        src = src_map.get(lk)
        if src is None:
            return []
        lang_safe = _lang_safe_from_source_file(src)

        keys: List[str] = []
        for fp in VOCAB_POS_DIR.glob(f"{lang_safe}__*.jsonl"):
            # extract pos_key from filename
            # e.g. French__Adjective.jsonl -> Adjective
            name = fp.name
            prefix = f"{lang_safe}__"
            if not name.startswith(prefix):
                continue
            pos_part = name[len(prefix) :]
            if pos_part.lower().endswith(".jsonl"):
                pos_part = pos_part[: -len(".jsonl")]
            if pos_part:
                keys.append(pos_part)
        return sorted(set(keys))

    # Fallback: scan the source file (slower)
    files = resolve_lang_files((lang,))
    if not files:
        return []
    pos_set: set[str] = set()
    for e in iter_jsonl_records(files[0]):
        pos = e.get("pos")
        if isinstance(pos, str) and pos.strip():
            pos_set.add(_pos_key(pos))
    return sorted(pos_set)


@st.cache_data(show_spinner=True)
def get_pos_options_for_langs(langs: Tuple[str, ...]) -> List[str]:
    """
    POS options for selected languages.
    - If exactly 2 languages: intersection of POS keys
    - Else: union
    """
    pos_sets: List[set[str]] = []
    for lang in langs:
        pos_sets.append(set(list_pos_keys_for_language(lang)))

    if not pos_sets:
        return []

    if len(langs) == 2:
        out = set.intersection(*pos_sets) if all(pos_sets) else set()
    else:
        out = set.union(*pos_sets)
    return sorted(out)



@st.cache_data(show_spinner=True)
def scan_categories_for_lang_and_pos(
    lang: str,
    pos: str,
    max_records: int = 200_000,
    max_unique: int = 5000,
) -> List[str]:
    """
    Get categories present for (language, pos).

    Prefer split file <LangSafe>__<POS>.jsonl, which is fast.
    Fallback: scan original file and filter by pos (slower).
    """
    cats: set[str] = set()
    pk = _pos_key(pos)

    # prefer split file
    if VOCAB_POS_DIR.exists():
        src_map = list_language_files()
        lk = _normalize_lang_key(lang)
        lk = LANG_ALIASES.get(lk, lk)
        src = src_map.get(lk)
        if src is not None:
            lang_safe = _lang_safe_from_source_file(src)
            split_fp = VOCAB_POS_DIR / f"{lang_safe}__{pk}.jsonl"
            if split_fp.exists():
                n = 0
                for e in iter_jsonl_records(split_fp):
                    n += 1
                    if n > max_records:
                        break
                    for c in vocab_record_categories(e):
                        if c:
                            cats.add(c)
                            if len(cats) >= max_unique:
                                break
                    if len(cats) >= max_unique:
                        break
                return sorted(cats)

    # fallback: scan original & filter pos
    files = resolve_lang_files((lang,))
    if not files:
        return []
    n = 0
    for e in iter_jsonl_records(files[0]):
        n += 1
        if n > max_records:
            break
        if e.get("pos") != pos:
            continue
        for c in vocab_record_categories(e):
            if c:
                cats.add(c)
                if len(cats) >= max_unique:
                    break
        if len(cats) >= max_unique:
            break
    return sorted(cats)


@st.cache_data(show_spinner=True)
def get_category_options_for_langs_and_pos(langs: Tuple[str, ...], pos: str | None) -> List[str]:
    """
    Categories depend on POS:
      A) If POS selected, categories are only those present in that POS
      B) If exactly 2 languages, categories are intersection across the two languages (within that POS)
         else union.
    """
    if not pos:
        return []

    cat_sets: List[set[str]] = []
    for lang in langs:
        cat_sets.append(set(scan_categories_for_lang_and_pos(lang, pos)))

    if not cat_sets:
        return []

    if len(langs) == 2:
        out = set.intersection(*cat_sets) if all(cat_sets) else set()
    else:
        out = set.union(*cat_sets)
    return sorted(out)


@st.cache_data(show_spinner=True)
def scan_etymologies_for_lang_and_pos(
    lang: str,
    pos: str,
    max_records: int = 200_000,
    max_unique: int = 5000,
) -> List[str]:
    """
    Returns short etymology *sources* (e.g. Latin/French/Proto-Bantu), not raw etymology strings.
    """
    out: set[str] = set()
    pk = _pos_key(pos)

    # prefer split file
    if VOCAB_POS_DIR.exists():
        src_map = list_language_files()
        lk = _normalize_lang_key(lang)
        lk = LANG_ALIASES.get(lk, lk)
        src = src_map.get(lk)
        if src is not None:
            lang_safe = _lang_safe_from_source_file(src)
            split_fp = VOCAB_POS_DIR / f"{lang_safe}__{pk}.jsonl"
            if split_fp.exists():
                n = 0
                for e in iter_jsonl_records(split_fp):
                    n += 1
                    if n > max_records:
                        break
                    src_lbl = ety_source(vocab_record_etymology(e))
                    if src_lbl:
                        out.add(src_lbl)
                        if len(out) >= max_unique:
                            break
                return sorted(out)

    # fallback: scan original & filter pos
    files = resolve_lang_files((lang,))
    if not files:
        return []

    n = 0
    for e in iter_jsonl_records(files[0]):
        n += 1
        if n > max_records:
            break
        if e.get("pos") != pos:
            continue
        src_lbl = ety_source(vocab_record_etymology(e))
        if src_lbl:
            out.add(src_lbl)
            if len(out) >= max_unique:
                break

    return sorted(out)


@st.cache_data(show_spinner=True)
def get_etymology_options_for_langs_and_pos(langs: Tuple[str, ...], pos: str | None) -> List[str]:
    """
    Etymologies depend on POS:
      - If POS selected, etymologies are only those present in that POS
      - If exactly 2 languages: intersection across the two langs (within that POS), else union
    """
    if not pos:
        return []

    ety_sets: List[set[str]] = []
    for lang in langs:
        ety_sets.append(set(scan_etymologies_for_lang_and_pos(lang, pos)))

    if not ety_sets:
        return []

    if len(langs) == 2:
        out = set.intersection(*ety_sets) if all(ety_sets) else set()
    else:
        out = set.union(*ety_sets)
    return sorted(out)


# --- Etymology grouping helpers ---------------------------------------------

_ETY_TYPE_RULES = [
    ("borrowed", re.compile(r"\bborrowed from\b|\bloanword\b", re.I)),
    ("inherited", re.compile(r"\binherited from\b", re.I)),
    ("calque", re.compile(r"\bcalque\b|\bloan translation\b", re.I)),
    ("compound", re.compile(r"\bcompound\b|\bfrom\b.+\+\s*.+", re.I)),
    ("derived", re.compile(r"\bderived\b|\bfrom\b.+\b(via|ultimately)\b", re.I)),
]

def ety_type(ety: str | None) -> str:
    t = (ety or "").strip()
    if not t:
        return ""
    for name, rx in _ETY_TYPE_RULES:
        if rx.search(t):
            return name
    return "other"

# fr:Foo, it:Foo, sw:Foo etc. (sometimes appear)
_LANGCODE_PREFIX = re.compile(r"^[A-Za-z]{2,3}:\s*")

# A long-but-practical list of etymology “languages/stages” commonly mentioned in Wiktionary.
# Includes broad IE coverage + Semitic + common areal contact languages + African/Bantu relevant items.
_KNOWN_ETY_LANGS = [
    # --- Proto & reconstructed
    "Proto-Indo-European", "Proto-Afroasiatic", "Proto-Uralic", "Proto-Turkic",
    "Proto-Semitic", "Proto-Berber", "Proto-Cushitic", "Proto-Bantu", "Proto-Niger-Congo",
    "Proto-Atlantic-Congo", "Proto-Volta-Congo", "Proto-Quechuan", "Proto-Dravidian",

    # --- Ancient / Classical
    "Akkadian", "Sumerian", "Ancient Egyptian", "Coptic",
    "Classical Latin", "Late Latin", "Vulgar Latin", "Medieval Latin",
    "Ancient Greek", "Koine Greek", "Byzantine Greek", "Hellenistic Greek",
    "Sanskrit", "Pali", "Prakrit", "Avestan", "Old Persian", "Middle Persian",
    "Biblical Hebrew", "Mishnaic Hebrew", "Aramaic", "Syriac",
    "Classical Arabic", "Quranic Arabic",

    # --- Germanic (useful even if your app targets Romance/Bantu; etys often cite these)
    "Proto-Germanic", "Gothic",
    "Old English", "Middle English", "Early Modern English",
    "Old Norse", "Old Swedish", "Old Danish",
    "Old High German", "Middle High German",
    "Old Saxon", "Old Frisian",

    # --- Romance
    "Latin", "Italian", "French", "Old French", "Middle French",
    "Spanish", "Old Spanish", "Portuguese", "Old Portuguese",
    "Catalan", "Occitan", "Old Occitan", "Romanian", "Sardinian",
    "Venetian", "Neapolitan", "Sicilian", "Galician",
    "Lombard", "Piedmontese", "Friulian",

    # --- Slavic
    "Proto-Slavic", "Old Church Slavonic",
    "Russian", "Ukrainian", "Belarusian", "Polish", "Czech", "Slovak",
    "Bulgarian", "Serbo-Croatian", "Croatian", "Serbian", "Slovenian", "Macedonian",

    # --- Celtic
    "Proto-Celtic", "Old Irish", "Middle Irish", "Irish",
    "Scottish Gaelic", "Welsh", "Breton", "Cornish", "Manx",

    # --- Iranian & Indo-Aryan
    "Persian", "New Persian", "Dari", "Tajik", "Kurdish", "Pashto",
    "Hindi", "Urdu", "Bengali", "Punjabi", "Marathi", "Gujarati", "Nepali", "Sinhala",

    # --- Turkic / Mongolic / Tungusic
    "Turkish", "Ottoman Turkish", "Azerbaijani", "Kazakh", "Kyrgyz", "Uzbek", "Turkmen",
    "Mongolian", "Manchu",

    # --- East & SE Asia
    "Chinese", "Middle Chinese", "Old Chinese", "Mandarin", "Cantonese",
    "Japanese", "Korean", "Vietnamese", "Thai", "Lao", "Khmer", "Burmese",

    # --- Semitic (modern)
    "Hebrew", "Yiddish", "Arabic", "Egyptian Arabic", "Levantine Arabic", "Maghrebi Arabic",
    "Amharic", "Tigrinya", "Tigre", "Geʽez",
    "Somali", "Oromo", "Afar", "Beja", "Hausa", "Berber", "Tamazight",

    # --- South Asia (Dravidian)
    "Tamil", "Telugu", "Kannada", "Malayalam",

    # --- Africa (high-contact + relevant)
    "Swahili", "Arabic (Swahili)",  # sometimes appears in templates
    "English", "Afrikaans", "Dutch", "German",
    "Xitsonga", "Tshivenda", "Sesotho", "Sepedi", "Setswana",
    "isiZulu", "Zulu", "isiXhosa", "Xhosa", "siSwati", "Swati",
    "Ndebele", "Northern Ndebele", "Southern Ndebele",
    "Chichewa", "Chewa", "Nyanja", "Cinyanja",
    "Shona", "Xitsonga", "Kikuyu", "Gikuyu",
    "Luganda", "Ganda", "Kinyarwanda", "Kirundi", "Rundi",
    "Lingala", "Kongo", "Kikongo",
    "Chokwe", "Bemba", "Tonga", "Lozi", "Luba-Katanga", "Luba-Lulua",
    "Makhuwa", "Yao", "Sena", "Tsonga",
    "Xitsonga", "Ronga", "Changana",
    "Fula", "Fulfulde", "Wolof", "Yoruba", "Igbo", "Ewe", "Twi", "Akan",
    "Xitsonga",  # duplicates are fine; regex is deduped by set below
    "Xitsonga",  # (you can remove duplicates later if you want)
]

# Deduplicate while keeping the list readable
_KNOWN_ETY_LANGS = list(dict.fromkeys(_KNOWN_ETY_LANGS))

# Match longest names first (e.g., "Old French" before "French")
_LANG_MENTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(x) for x in sorted(_KNOWN_ETY_LANGS, key=len, reverse=True)) + r")\b",
    re.I
)

def ety_source(ety: str | None) -> str:
    """
    Return a short grouping label (first mentioned language/stage).
    This is intentionally heuristic.
    """
    t = (ety or "").strip()
    if not t:
        return ""
    t = _LANGCODE_PREFIX.sub("", t).strip()
    m = _LANG_MENTION_RE.search(t)
    return (m.group(1) if m else "")


# ----------------------------
# Loading entries (fast with split files)
# ----------------------------

@st.cache_data(show_spinner=True)
def load_vocab_entries_filtered(
    langs: Tuple[str, ...],
    pos: str | None,
    cat: str | None,
    ety: str | None = None,
    word_query: str | None = None,
    limit_rows: int = 5000,
) -> pd.DataFrame:
    """
    Load matching entries for selected languages.
    - If pos is set AND split files exist for all langs: scan only those small .jsonl files (fast)
    - Else: scan original per-language files (could be slower)
    Stops at limit_rows.
    """
    files = resolve_lang_pos_files(langs, pos)
    q = (word_query or "").strip().lower()

    rows: List[dict] = []

    for fp in files:
        for e in iter_jsonl_records(fp):
            # If we are using split files, pos filter is implicitly satisfied.
            if pos and fp.name.lower().endswith(".jsonl.gz"):
                # fallback case (not split): need to filter pos
                if e.get("pos") != pos:
                    continue
            elif pos and not fp.name.lower().endswith(".jsonl.gz"):
                # split file: no need to check pos
                pass

            if cat:
                cats = vocab_record_categories(e)
                if cat not in cats:
                    continue

            if ety:
                src_lbl = ety_source(vocab_record_etymology(e))
                if src_lbl != ety:
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
    Returns rows:
      {meaning, meaning_lang2, lang1, lang2, _sim}

    - meaning: anchor gloss (lang1 gloss when available, else lang2 gloss for leftovers)
    - meaning_lang2: ONLY filled when there is a matched pair AND glosses differ (debug)
    """
    used2 = set()
    out: list[dict] = []

    for w1, g1 in rows_lang1:
        best_sim = 0.0
        best_j = -1
        best_w2 = ""
        best_g2 = ""

        for j, (w2, g2) in enumerate(rows_lang2):
            if j in used2:
                continue
            sim = gloss_similarity(g1, g2)
            if sim > best_sim:
                best_sim = sim
                best_j = j
                best_w2 = w2
                best_g2 = g2

        if best_sim >= threshold and best_j >= 0:
            used2.add(best_j)
            out.append(
                {
                    "meaning": g1,
                    "meaning_lang2": (best_g2 if isinstance(best_g2, str) and best_g2 != g1 else ""),
                    "lang1": w1,
                    "lang2": best_w2,
                    "_sim": best_sim,
                }
            )
        else:
            out.append({"meaning": g1, "meaning_lang2": "", "lang1": w1, "lang2": "", "_sim": 0.0})

    # leftover lang2 meanings
    for j, (w2, g2) in enumerate(rows_lang2):
        if j in used2:
            continue
        out.append({"meaning": g2, "meaning_lang2": "", "lang1": "", "lang2": w2, "_sim": 0.0})

    return out

