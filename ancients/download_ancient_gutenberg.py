#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests


GUTENDEX_BASE = "https://gutendex.com/books/"


@dataclass
class BookRecord:
    gutenberg_id: int
    title: str
    authors: List[Dict]
    translators: List[Dict]
    subjects: List[str]
    bookshelves: List[str]
    languages: List[str]
    download_count: int
    formats: Dict[str, str]


def normalize_title(title: str) -> str:
    text = title.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.split(r"\s+[-:\u2013]\s+", text)[0].strip()
    return text


def translator_key(translators: List[Dict]) -> str:
    if not translators:
        return ""
    names = sorted(t.get("name", "").strip() for t in translators if t.get("name"))
    return "|".join(names)


def author_key(authors: List[Dict]) -> str:
    if not authors:
        return ""
    names = sorted(a.get("name", "").strip() for a in authors if a.get("name"))
    return "|".join(names)


def pick_plaintext_url(formats: Dict[str, str]) -> Optional[str]:
    for key in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if key in formats:
            return formats[key]
    return None


def fetch_books(author_year_end: int = 500, languages: str = "en", request_delay_s: float = 0.5) -> Iterable[BookRecord]:
    url = GUTENDEX_BASE
    params = {"author_year_end": author_year_end, "languages": languages}
    session = requests.Session()
    while url:
        resp = session.get(url, params=params if url == GUTENDEX_BASE else None, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for b in data.get("results", []):
            yield BookRecord(
                gutenberg_id=b.get("id"),
                title=b.get("title", ""),
                authors=b.get("authors", []) or [],
                translators=b.get("translators", []) or [],
                subjects=b.get("subjects", []) or [],
                bookshelves=b.get("bookshelves", []) or [],
                languages=b.get("languages", []) or [],
                download_count=b.get("download_count", 0) or 0,
                formats=b.get("formats", {}) or {},
            )
        url = data.get("next")
        params = None
        time.sleep(request_delay_s)


def choose_top_translation_per_work(records: Iterable[BookRecord]) -> List[BookRecord]:
    groups: Dict[Tuple[str, str], List[BookRecord]] = {}
    for r in records:
        key = (author_key(r.authors), normalize_title(r.title))
        groups.setdefault(key, []).append(r)

    chosen: List[BookRecord] = []
    for _, items in groups.items():
        per_translator: Dict[str, BookRecord] = {}
        for it in items:
            tkey = translator_key(it.translators)
            best = per_translator.get(tkey)
            if best is None or it.download_count > best.download_count:
                per_translator[tkey] = it
        top = max(per_translator.values(), key=lambda x: x.download_count)
        chosen.append(top)
    return chosen


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def should_keep_record(authors: List[Dict], cutoff_year: int) -> bool:
    for a in authors or []:
        dy = a.get("death_year")
        by = a.get("birth_year")
        if dy is not None and dy > cutoff_year:
            return False
        if dy is None and by is not None and by > cutoff_year:
            return False
    return True


def write_manifest(path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "gutenberg_id",
        "title",
        "authors",
        "translators",
        "languages",
        "download_count",
        "subjects",
        "bookshelves",
        "text_url",
        "local_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def download_texts(records: List[BookRecord], out_dir: str, request_delay_s: float = 0.5) -> List[Dict[str, str]]:
    ensure_dir(out_dir)
    session = requests.Session()
    manifest_rows: List[Dict[str, str]] = []
    for r in sorted(records, key=lambda x: (-x.download_count, x.gutenberg_id)):
        text_url = pick_plaintext_url(r.formats)
        if not text_url:
            continue
        filename = f"{r.gutenberg_id}_" + re.sub(r"[^A-Za-z0-9._-]", "-", r.title)[:80] + ".txt"
        local_path = os.path.join(out_dir, filename)
        try:
            resp = session.get(text_url, timeout=60)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            print(f"Failed to download {r.gutenberg_id} from {text_url}: {e}", file=sys.stderr)
            time.sleep(request_delay_s)
            continue

        manifest_rows.append(
            {
                "gutenberg_id": str(r.gutenberg_id),
                "title": r.title,
                "authors": author_key(r.authors),
                "translators": translator_key(r.translators),
                "languages": ",".join(r.languages),
                "download_count": str(r.download_count),
                "subjects": "|".join(r.subjects),
                "bookshelves": "|".join(r.bookshelves),
                "text_url": text_url,
                "local_path": os.path.abspath(local_path),
            }
        )

        time.sleep(request_delay_s)

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download English translations of ancient works (authors died <= 500 CE) from Project Gutenberg via Gutendex.")
    parser.add_argument("--out_dir", default="NLM/ancients/raw", help="Directory to store downloaded .txt files")
    parser.add_argument("--manifest", default=None, help="Path to write CSV manifest (default: <out_dir>/manifest.csv)")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--author_year_end", type=int, default=500, help="Upper bound for author death year (cutoff)")
    parser.add_argument("--languages", default="en", help="Comma-separated Gutenberg language codes (default: en)")
    parser.add_argument("--count_only", action="store_true", help="Only print counts after filtering/selection; do not download")
    args = parser.parse_args()

    out_dir = args.out_dir
    manifest_path = args.manifest or os.path.join(out_dir, "manifest.csv")

    print("Fetching catalog...")
    all_records = list(fetch_books(author_year_end=args.author_year_end, languages=args.languages, request_delay_s=args.sleep))
    print(f"Fetched {len(all_records)} records")

    print("Filtering to primary ancient works...")
    filtered_records: List[BookRecord] = []
    for r in all_records:
        if should_keep_record(r.authors, args.author_year_end):
            filtered_records.append(r)
    print(f"Retained {len(filtered_records)} after filtering; dropped {len(all_records) - len(filtered_records)} modern/secondary works")

    print("Selecting most popular translation per work...")
    selected = choose_top_translation_per_work(filtered_records)
    print(f"Selected {len(selected)} unique works (one translation each)")

    if args.count_only:
        print(f"COUNT_ONLY: {len(selected)} works at cutoff year {args.author_year_end}")
        return

    print("Downloading plain text files...")
    manifest_rows = download_texts(selected, out_dir=out_dir, request_delay_s=args.sleep)
    print(f"Downloaded {len(manifest_rows)} files")

    print(f"Writing manifest to {manifest_path} ...")
    write_manifest(manifest_path, manifest_rows)
    print("Done.")


if __name__ == "__main__":
    main()



