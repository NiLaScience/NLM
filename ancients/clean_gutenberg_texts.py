#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Optional, Tuple


START_MARKERS = [
    r"^\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*$",
    r"^\*\*\*\s*START OF.*PROJECT GUTENBERG.*$",
]

END_MARKERS = [
    r"^\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*$",
    r"^\*\*\*\s*END OF.*PROJECT GUTENBERG.*$",
]


def strip_gutenberg_header_footer(text: str) -> Tuple[str, bool]:
    lines = text.splitlines()
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    # Find start marker
    for i, line in enumerate(lines):
        for pat in START_MARKERS:
            if re.search(pat, line, flags=re.IGNORECASE):
                start_idx = i + 1
                break
        if start_idx is not None:
            break

    # Find end marker
    for j in range(len(lines) - 1, -1, -1):
        line = lines[j]
        for pat in END_MARKERS:
            if re.search(pat, line, flags=re.IGNORECASE):
                end_idx = j
                break
        if end_idx is not None:
            break

    if start_idx is not None and end_idx is not None and end_idx > start_idx:
        return "\n".join(lines[start_idx:end_idx]), True

    # Fallback: remove license if present
    # Drop everything from "START: FULL LICENSE" to end
    m = re.search(r"(?im)^START: FULL LICENSE[\s\S]*$", text)
    if m:
        text = text[: m.start()] 

    # Also trim the initial boilerplate if present up to the first "Title:" line then a blank line
    # This is heuristic; only applied if START marker wasn't found
    if start_idx is None:
        m_title = re.search(r"(?im)^Title:\s*.+$", text)
        if m_title:
            # keep from title line onward
            text = text[m_title.start():]
    return text, False


def normalize_unicode_punctuation(text: str) -> str:
    replacements = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
        "\xa0": " ",  # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def remove_bracketed_notes_and_citations(text: str) -> str:
    # Remove [footnote ...] blocks and simple numeric bracket refs like [241]
    text = re.sub(r"(?is)\[\s*footnote[^\]]*?\]", "", text)
    text = re.sub(r"\[\s*\d{1,4}\s*\]", "", text)
    return text


def remove_underscore_italics_and_editorial_parens(text: str) -> str:
    # Remove underscores used for italics: _word phrase_ -> word phrase
    text = re.sub(r"(?<!\w)_(\S[^_]*?\S)_(?!\w)", r"\1", text)
    # Remove entire parenthetical segments that include underscore-marked tokens e.g., (_alex._, 13)
    text = re.sub(r"\([^)]*_[^)]*\)", "", text)
    return text


def remove_braces_keep_content(text: str) -> str:
    # Replace {word} -> word
    return text.replace("{", "").replace("}", "")


def remove_sentence_initial_line_numbers(text: str) -> str:
    # Remove Arabic line numbers at the start or after sentence terminators
    return re.sub(r"(?:(?<=^)|(?<=[.!?]\s))\d{1,4}\s+(?=[a-zA-Z])", "", text)


def remove_structural_headings(text: str) -> str:
    # Remove tokens like Chapter XV., Book 2, Scene ii., Act I, Canto 3, Section 4 when they appear at sentence starts
    return re.sub(
        r"(?i)(?:(?<=^)|(?<=[.!?]\s))(?:chapter|book|scene|act|canto|section)\s+[ivxlcdm0-9]+\.?\s*",
        "",
        text,
    )


def remove_public_domain_dedication_blocks(text: str) -> str:
    # Line-based removal: if a line contains 'public domain' and ('dedicate' or 'work of authorship' or 'cc0'),
    # drop that line and continue dropping until a blank line.
    lines = text.splitlines()
    out_lines = []
    skipping = False
    for line in lines:
        l = line.lower()
        if not skipping and (
            ("public domain" in l and ("dedicat" in l or "work of authorship" in l))
            or "cc0" in l
            or "creativecommons" in l
        ):
            skipping = True
            continue
        if skipping:
            if not line.strip():
                skipping = False
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def capitalize_pronoun_I(text: str) -> str:
    # Replace standalone i with I
    return re.sub(r"\bi\b", "I", text)


def fix_hyphenated_linebreaks(text: str) -> str:
    # Join words split across line breaks with hyphens: e.g., "exami-\n nation" -> "examination"
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def remove_toc_blocks(text: str) -> str:
    # Remove simple Table of Contents sections
    lines = text.splitlines()
    out = []
    skipping = False
    seen_header = False
    for line in lines:
        l = line.strip().lower()
        if not skipping and (l.startswith("contents") or l.startswith("table of contents")):
            skipping = True
            seen_header = True
            continue
        if skipping:
            if not line.strip() and seen_header:
                skipping = False
                seen_header = False
            continue
        out.append(line)
    return "\n".join(out)


def remove_speaker_labels(text: str) -> str:
    # Remove speaker labels at the beginning of lines like "Chorus." or "APOLLO:" etc.
    # Apply multiple passes for nested or repeated patterns
    pattern = re.compile(r"(?m)^(?:[A-Z][A-Za-z\-']{1,24}|[A-Z]{2,24})(?:\.|:)\s+")
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub("", text)
    return text


def remove_all_bracketed_content(text: str) -> str:
    # Remove any content inside (), [], {}, <> (including nested by iterative passes), then drop stray brackets
    patterns = [
        r"\([^()]*\)",
        r"\[[^\[\]]*\]",
        r"\{[^{}]*\}",
        r"<[^<>]*>",
    ]
    prev = None
    # Iteratively strip to handle nesting
    while prev != text:
        prev = text
        for pat in patterns:
            text = re.sub(pat, " ", text)
    # Remove any remaining stray bracket characters
    text = re.sub(r"[\[\]{}()<>]", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse whitespace within paragraphs but preserve paragraph breaks as double newlines."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on blank lines to detect paragraphs
    paragraphs = re.split(r"\n\s*\n+", text)
    normalized = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        # Remove space before punctuation
        p = re.sub(r"\s+([,;:!?])", r"\1", p)
        # Ensure one space after sentence terminators if next is alnum/opening quote/paren
        p = re.sub(r"([.?!])(?!\s|$)", r"\1 ", p)
        p = re.sub(r"\s+", " ", p).strip()
        normalized.append(p)
    return "\n\n".join(normalized)


def sentence_case(text: str) -> str:
    # Lowercase all, then capitalize first letter of each sentence (start or after .?! )
    lower = text.lower()

    def cap_after(match: re.Match) -> str:
        prefix = match.group(1)
        letter = match.group(2)
        return f"{prefix}{letter.upper()}"

    # Capitalize first letter in text if alpha
    lower = re.sub(r"^(\s*)([a-z])", cap_after, lower)
    # Capitalize after sentence terminators followed by space(s)
    lower = re.sub(r"([.?!]\s+)([a-z])", cap_after, lower)
    return lower


def clean_text(raw: str) -> Tuple[str, bool]:
    raw = fix_hyphenated_linebreaks(raw)
    body, had_markers = strip_gutenberg_header_footer(raw)
    body = remove_public_domain_dedication_blocks(body)
    body = remove_toc_blocks(body)
    body = normalize_unicode_punctuation(body)
    body = remove_bracketed_notes_and_citations(body)
    body = remove_underscore_italics_and_editorial_parens(body)
    body = remove_braces_keep_content(body)
    body = remove_speaker_labels(body)
    body = remove_structural_headings(body)
    body = remove_sentence_initial_line_numbers(body)
    body = remove_all_bracketed_content(body)
    body = collapse_whitespace(body)
    # Preserve original casing; optionally fix standalone lowercase 'i'
    body = capitalize_pronoun_I(body)
    return body, had_markers


def process_file(src_path: str, dst_path: str, dry_run: bool = False) -> Tuple[bool, bool]:
    try:
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception as e:
        print(f"Failed to read {src_path}: {e}", file=sys.stderr)
        return False, False

    cleaned, had_markers = clean_text(raw)

    if dry_run:
        return True, had_markers

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
    except Exception as e:
        print(f"Failed to write {dst_path}: {e}", file=sys.stderr)
        return False, had_markers

    return True, had_markers


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Project Gutenberg texts: strip headers/footers, normalize punctuation, whitespace, and sentence case.")
    parser.add_argument("--in_dir", required=True, help="Input directory with .txt files")
    parser.add_argument("--out_dir", required=True, help="Output directory for cleaned .txt files")
    parser.add_argument("--dry_run", action="store_true", help="Do not write files; just report counts")
    args = parser.parse_args()

    total = 0
    ok = 0
    with_markers = 0
    for root, _, files in os.walk(args.in_dir):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            total += 1
            src = os.path.join(root, name)
            rel = os.path.relpath(src, args.in_dir)
            dst = os.path.join(args.out_dir, rel)
            success, had_markers = process_file(src, dst, dry_run=args.dry_run)
            ok += 1 if success else 0
            with_markers += 1 if had_markers else 0

    print(f"Processed {ok}/{total} files; {with_markers} had explicit START/END markers")


if __name__ == "__main__":
    main()



