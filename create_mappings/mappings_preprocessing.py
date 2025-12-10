#!/usr/bin/env python

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from Embedder import SectionEmbedder
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_toc_block(text: str) -> str:
    """
    Extract the Table of Contents block from the grammar file.

    The TOC in your Italian file looks like:

        Table of Contents
        # Italian grammar
        ## Articles
        ...
        ```</pre>

    We:
    - start at the line containing "Table of Contents" (case-insensitive)
    - continue until the next line starting with ``` or end-of-file.
    """
    pattern = re.compile(
        r"(?is)(table of contents.*?)(?=^```|\Z)",
        re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        return ""
    return m.group(1).strip()

def parse_toc(toc_text: str) -> List[Dict[str, Any]]:
    """
    Parse TOC text into a list of section dicts:

    {
        'raw':  original line including hashes and any " > ### ...",
        'path': ['Adjectives', 'Degrees of comparison'],
        'full': 'Adjectives > Degrees of comparison'
    }

    We treat ">" in the TOC as hierarchy separators, and strip off any
    leading '#' from each segment.
    """
    sections: List[Dict[str, Any]] = []

    for line in toc_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # We only care about lines that look like headings ("#" at start)
        if not line.startswith("#"):
            continue

        # Split on ">" hierarchy markers and clean each piece
        parts: List[str] = []
        for part in line.split('>'):
            part = part.strip()
            if part:
                parts.append(part)

        if not parts:
            continue

        full = " > ".join(parts)
        sections.append({
            "raw": line,
            "path": parts,
            "full": full,
        })

    return sections

def lexical_match_sections(
    term: str,
    sections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find sections whose 'full' path contains the term, with the constraint
    that there is no letter directly before the term.

    Examples (term='verb'):
      - matches 'Verbs'        ✅ (no letter before 'verb')
      - matches 'verb tense'   ✅
      - does NOT match 'Adverbs' ❌ (there is 'd' before 'verb')
    """
    pattern = build_term_pattern(term)
    matches: List[Dict[str, Any]] = []

    for s in sections:
        full = s["full"]
        if pattern.search(full):
            matches.append(s)

    return matches

def lexical_match_for_terms_in_order(
    terms: List[str],
    sections: List[Dict[str, Any]],
) -> Tuple[str | None, List[Dict[str, Any]]]:
    """
    Try lexical search for each term in order and return
    (term_used, matches) of the first term that yields any matches.

    If none of the terms match, returns (None, []).
    """
    for term in terms:
        matches = lexical_match_sections(term, sections)
        if matches:
            return term, matches
    return None, []

def split_camel_case_phrase(s: str) -> str:
    """
    Convert camelCase or PascalCase into a multi-word phrase, but keep it as 1 token.
    Example:
        "pastTense" → "past tense"
        "reportedIndirectSpeech" → "reported indirect speech"
    """
    # Insert space before uppercase letters (except at start)
    s_spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    return s_spaced.lower()  # one phrase, not a list

def build_term_pattern(term: str) -> re.Pattern:
    """
    Build a case-insensitive regex that matches `term` only when
    there is NOT a letter immediately before it.

    - 'verb' matches 'verb', 'verbs', 'verb.' etc.
    - 'verb' does NOT match 'adverb', 'myverb', etc.

    For multi-word terms (e.g. 'past tense') we also allow flexible
    whitespace between words.
    """
    # Escape and make internal spaces flexible (\s+)
    term_escaped = re.escape(term.strip())
    # turn escaped spaces into \s+ for robustness
    term_escaped = re.sub(r"\\\s+", r"\\s+", term_escaped)

    # (?i)        -> case-insensitive
    # (?<![A-Za-z]) -> negative lookbehind: char before is not a letter
    pattern = re.compile(rf"(?i)(?<![A-Za-z]){term_escaped}")
    return pattern

def extract_topic_terms_from_filename(path: str) -> List[str]:
    """
    Extract ordered topic terms, keeping camelCase as a single multi-word token.

    Examples:
        FORMAT_pastTense_tense_verb.txt
        → ["past tense", "tense", "verb"]

        FORMAT_determiners-articles_noun.txt
        → ["determiners", "articles", "noun"]

        FORMAT_reportedIndirectSpeech_verb.txt
        → ["reported indirect speech", "verb"]
    """
    filename = os.path.basename(path)
    m = re.match(r'^FORMAT_(.+?)\.txt$', filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract topic terms from filename: {path}")

    body = m.group(1)

    # Split on _ and -
    raw_parts = re.split(r'[_-]+', body)

    final_terms = []
    for part in raw_parts:
        if not part.strip():
            continue

        # If camelCase → produce multi-word phrase as a single token
        if re.search(r'[A-Z]', part):   # contains capitals → camelCase/PascalCase
            phrase = split_camel_case_phrase(part)  # e.g. "past tense"
            final_terms.append(phrase)
        else:
            final_terms.append(part.lower())

    return final_terms

def list_format_files(base_path: str = "schemas/lang-de") -> List[str]:
    """
    Return all FORMAT_*.txt files under schemas/lang-de/, recursively.
    """
    base = Path(base_path)
    files: List[str] = [
        str(p)
        for p in base.rglob("FORMAT_*.txt")
        if p.is_file() and p.suffix.lower() == ".txt"
    ]
    return sorted(files)

def find_sections_for_file(
    path: str,
    sections: List[Dict[str, Any]],
    embedder: SectionEmbedder,
    top_k: int = 5,
    min_sim: float = 0.40,
) -> Dict[str, Any]:
    """
    For a given FORMAT_*.txt file:

    1. Extract ordered terms from filename (split on "_" and "-").
    2. Try lexical matches term-by-term.
    3. If none match, use embeddings on the full topic string.

    Returns a dict of:
    {
        "strategy": "lexical" | "embedding",
        "term_used": str | None,
        "topic_terms": [...],
        "matches": [section_dict, ...]
    }
    """
    terms = extract_topic_terms_from_filename(path)
    full_topic = " ".join(terms)

    term_used, lex_matches = lexical_match_for_terms_in_order(terms, sections)

    if lex_matches:
        return {
            "strategy": "lexical",
            "term_used": term_used,
            "topic_terms": terms,
            "matches": lex_matches,
        }

    emb_matches = embedder.match_by_embedding(
        full_topic,
        top_k=top_k,
        min_sim=min_sim,
    )

    return {
        "strategy": "embedding",
        "term_used": None,
        "topic_terms": terms,
        "matches": emb_matches,
    }

def map_files_to_toc(
    base_path: str,
    sections: List[Dict[str, Any]],
    embedder: SectionEmbedder,
    top_k: int = 5,
    min_sim: float = 0.40,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience wrapper to compute a mapping for all FORMAT_*.txt files
    under base_path.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    for path in list_format_files(base_path):
        mapping[path] = find_sections_for_file(
            path=path,
            sections=sections,
            embedder=embedder,
            top_k=top_k,
            min_sim=min_sim,
        )
    return mapping

def normalize_heading_to_full(line: str) -> str:
    """
    Given a heading line (WITHOUT the leading #'s), normalize it in the same
    way as parse_toc() does:

        - split on '>'
        - strip spaces around pieces
        - join with ' > '
    """
    parts: List[str] = []
    for part in line.split(">"):
        part = part.strip()
        if part:
            parts.append(part)
    return " > ".join(parts)

def remove_toc_block(full_text: str) -> str:
    """
    Remove the Table of Contents block from the full grammar text
    (the same block that extract_toc_block() returns).
    """
    pattern = re.compile(
        r"(?is)(table of contents.*?)(?=^```|\Z)",
        re.MULTILINE
    )
    return pattern.sub("", full_text)

def build_body_section_index(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan the grammar body text and create an index:

        {
          "Adjectives": {
              "full": "Adjectives",
              "start": int,
              "end": int,
              "content": "..."
          },
          "Verbs > Regular conjugation > Conditional mood": { ... },
          ...
        }

    Rules:

    - Headings are lines starting with '#'.
    - Headings whose last component is 'Table ...' do NOT start a new section
      and do NOT end the current section. Their content is folded into the
      current non-table section.
    - A section for heading H ends at the first subsequent heading H2 such that:
        strip_table_suffix(H2) != strip_table_suffix(H)
      and H2 is not a table title.
    """
    index: Dict[str, Dict[str, Any]] = {}

    heading_pattern = re.compile(r"^#+\s+.+$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return index

    # Precompute normalized info for each heading
    infos = []
    for m in matches:
        heading_line = m.group(0)
        full = normalize_heading_to_full(heading_line)
        table_flag = is_table_title(full)
        base = strip_table_suffix(full)  # same as full for non-table headings

        infos.append({
            "match": m,
            "full": full,
            "base": base,
            "is_table": table_flag,
            "heading_start": m.start(),
            "heading_end": m.end(),
        })

    # For each NON-table heading, define a section that runs until
    # the next NON-table heading with a different base title.
    n = len(infos)
    for i, info in enumerate(infos):
        if info["is_table"]:
            # Tables are not separate sections
            continue

        full = info["full"]
        base = info["base"]
        content_start = info["heading_end"]

        # Find end: first later non-table heading with different base
        content_end = len(text)
        for j in range(i + 1, n):
            next_info = infos[j]
            if next_info["is_table"]:
                # Do not terminate the section on table headings
                continue

            # Only stop if the "title that comes" (base path) is different
            if next_info["base"] != base:
                content_end = next_info["heading_start"]
                break

        content = text[content_start:content_end].strip("\n")

        # Only add the first occurrence for a given full heading
        if full not in index:
            index[full] = {
                "full": full,
                "start": content_start,
                "end": content_end,
                "content": content,
            }

    return index

def is_table_title(full: str) -> bool:
    """
    Return True if the last component is a 'Table ...' heading.

    Works with hashes in the components, e.g.:
        '## Verbs > ### Conditional mood > ##### Table 30'
    """
    parts = [p.strip() for p in full.split(">")]
    if not parts:
        return False

    last_clean = parts[-1].lstrip("#").strip().lower()
    return last_clean.startswith("table")

def strip_table_suffix(full: str) -> str:
    """
    Remove a trailing 'Table ...' component if present.

    Examples:
        '## Verbs > ### Regular conjugation > #### Conditional mood > ##### Table 30'
            -> 'Verbs > Regular conjugation > Conditional mood'
        '## Verbs > ### Regular conjugation > #### Conditional mood'
            -> 'Verbs > Regular conjugation > Conditional mood'
    """
    parts = [p.strip() for p in full.split(">")]
    if parts:
        last_clean = parts[-1].lstrip("#").strip().lower()
        if last_clean.startswith("table"):
            parts = parts[:-1]

    # normalize remaining components for base comparison
    normalized_parts = [p.lstrip("#").strip() for p in parts if p.strip()]
    return " > ".join(normalized_parts)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(lang, source) -> None:
    # 1. Load the Italian grammar plaintext file
    if source == "docx":
        lang = f"BANTU_{lang}_NEW"

    grammar_path = f"data/{source}/outfiles/{lang}_plaintext_structured_w_tables.txt"
    with open(grammar_path, encoding="utf8") as f:
        full_text = f.read()

    # 2. Extract and parse TOC
    toc_text = extract_toc_block(full_text)
    if not toc_text:
        raise RuntimeError("Could not extract TOC block from grammar file.")

    sections = parse_toc(toc_text)
    if not sections:
        raise RuntimeError("Parsed TOC sections list is empty.")

    # NEW: build body index (for attaching actual section text)
    body_text = remove_toc_block(full_text)
    body_index = build_body_section_index(body_text)

    # 3. Prepare embedding model
    print(f"Initializing SectionEmbedder with {len(sections)} sections...")
    embedder = SectionEmbedder(sections)

    # 4. Build mapping for all German format files
    base_schema_path = "schemas/lang-de"
    mapping = map_files_to_toc(
        base_path=base_schema_path,
        sections=sections,
        embedder=embedder,
        top_k=5,
        min_sim=0.40,
    )

    # 5. Print results for inspection
    for fn, data in mapping.items():
        print("\nFile:", fn)
        print("  Strategy:", data["strategy"])
        print("  Topic terms:", data["topic_terms"])
        if data["term_used"]:
            print("  Lexical term used:", data["term_used"])
        for m in data["matches"]:
            sim = m.get("similarity")
            if sim is not None:
                print(f"    - {m['full']} (sim={sim:.3f})")
            else:
                print(f"    - {m['full']}")

    # 6. Optionally dump mapping to JSON (comment out if not needed)
    out_path = f"mappings_{source}/{source}_lang-de_to_{lang}_toc_mapping.json"
    serializable = {}

    for fn, data in mapping.items():
        matches_with_content = []
        for m in data["matches"]:
            full = m["full"]
            match_entry = {
                "full": full,
                "path": m["path"],
                "similarity": float(m.get("similarity", 1.0)),
            }

            section_info = body_index.get(full)
            if section_info is not None:
                match_entry["content"] = section_info["content"]
            else:
                # Optional: debug info if a TOC entry has no matching body section
                print(f"[WARN] No body section found for '{full}'")
                match_entry["content"] = None

            matches_with_content.append(match_entry)

        serializable[fn] = {
            "strategy": data["strategy"],
            "term_used": data["term_used"],
            "topic_terms": data["topic_terms"],
            "matches": matches_with_content,
        }

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\nWrote mapping to {out_path}")



if __name__ == "__main__":
    lang = [("Chewa_language", "wikipedia"),
        ("French_grammar", "wikipedia"),
        ("Italian_grammar", "wikipedia"),
        ("Portuguese_grammar", "wikipedia"),
        ("Romanian_grammar", "wikipedia"),
        ("Shona_language", "wikipedia"),
        ("Spanish_grammar", "wikipedia"),
        ("Swahili_grammar", "wikipedia"),
        ("Zulu_grammar", "wikipedia"),
        ("Chichewa", "docx"),
        ("ChiShona", "docx"),
        ("isiZulu", "docx"),
        ("KiSwahili", "docx")]
    for (l, source) in lang:
        main(l, source)
