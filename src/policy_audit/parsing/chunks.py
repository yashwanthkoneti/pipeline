"""Chunk parsing for the `**__KEY__:**VALUE` format.

Two things the obvious implementation gets wrong:

  The colon sits INSIDE the asterisks, and keys arrive decorated
  (`__FILE_NAME__`). Capturing the key without normalising it means
  `__FILE_NAME__` never matches a lookup for `file name`, and every
  chunk silently loses its title - which takes the document name out of
  every finding and out of every drill-down.

  document_chunks is a comma-joined list, but values contain commas, so
  splitting on commas corrupts them. Split on the repeated LEADING key
  instead.
"""

from __future__ import annotations

import ast
import json
import re
from typing import List, Sequence, Tuple

from ..domain.models import Chunk, Table
from .tables import (looks_like_unstructured_table,
                     render_table, split_segments)

KV_RE = re.compile(r"\*\*\s*([^*:\n]{1,80}?)\s*:\s*\*\*")

TITLE_KEYS = ("document title", "doc title", "file name", "filename",
              "title", "document", "source", "policy name")
SECTION_KEYS = ("section", "clause", "heading", "sub section", "subsection",
                "paragraph", "article")


def clean_key(key: str) -> str:
    """`__DOC_TITLE__` -> `doc title`."""
    k = re.sub(r"\*+", "", str(key)).strip()
    k = re.sub(r"^[_\W]+|[_\W]+$", "", k)
    return re.sub(r"[_\s]+", " ", k).strip()


def parse_fields(text: str) -> List[Tuple[str, str]]:
    """One chunk -> ordered (key, value), values spanning to the next key."""
    text = str(text)
    out: List[Tuple[str, str]] = []
    last_key, last_end = None, 0
    for m in KV_RE.finditer(text):
        seg = text[last_end:m.start()].strip()
        if last_key is not None:
            out.append((last_key, seg))
        elif seg:
            out.append(("(preamble)", seg))
        last_key, last_end = clean_key(m.group(1)), m.end()
    tail = text[last_end:].strip()
    if last_key is not None:
        out.append((last_key, tail))
    elif tail:
        out.append(("(body)", tail))
    return [(k, v.strip().strip(",").strip()) for k, v in out]


def split_chunks(raw: object, split_key: str | None = None) -> List[str]:
    """document_chunks -> a list of chunk strings."""
    if raw is None or str(raw).strip() in ("", "[]", "null", "None", "nan"):
        return []
    txt = str(raw).strip()
    # Only try the structured loaders if the payload could BE a literal.
    # ast.literal_eval on a 500KB string that is not one still runs the
    # Python parser over all of it before failing - about 2.5ms a row,
    # paid on every row, for nothing.
    loaders = ((json.loads, ast.literal_eval)
               if txt[:1] in "[{(\"'" else ())
    for loader in loaders:
        try:
            value = loader(txt)
        except Exception:
            continue
        if isinstance(value, list):
            return [x if isinstance(x, str) else json.dumps(x) for x in value]
        if isinstance(value, dict):
            return [json.dumps(value)]
        if isinstance(value, str):
            txt = value
            break
    key = split_key
    if not key:
        m = KV_RE.search(txt)
        key = m.group(1).strip() if m else None
    if not key:
        return [txt]
    marker = re.compile(r"\*\*\s*" + re.escape(key) + r"\s*:\s*\*\*")
    starts = [m.start() for m in marker.finditer(txt)]
    if len(starts) <= 1:
        return [txt]
    bounds = starts + [len(txt)]
    out = []
    for lo, hi in zip(bounds, bounds[1:]):
        piece = txt[lo:hi].strip().rstrip(",").strip()
        if piece:
            out.append(piece)
    return out


def _pick(fields: Sequence[Tuple[str, str]], wanted: Sequence[str]) -> str:
    low = [(k.lower(), v) for k, v in fields]
    for w in wanted:
        for k, v in low:
            if k == w:
                return v
    for w in wanted:
        for k, v in low:
            if w in k:
                return v
    return ""


def parse_chunks(raw: object, split_key: str | None = None) -> List[Chunk]:
    """document_chunks -> [Chunk], tables expanded row by row."""
    chunks: List[Chunk] = []
    for index, piece in enumerate(split_chunks(raw, split_key)):
        fields = parse_fields(piece)
        title = _pick(fields, TITLE_KEYS) or "?"
        lines: List[str] = []
        tables: List[Table] = []
        for key, value in fields:
            if not value:
                continue
            segs = split_segments(value)
            if not any(kind == "table" for kind, _ in segs):
                lines.append(f"{key}: {value}")
                continue
            lines.append(f"{key}:")
            for kind, payload in segs:
                if kind == "table":
                    tables.append(payload)          # type: ignore[arg-type]
                    lines.append(render_table(payload))  # type: ignore[arg-type]
                else:
                    lines.append(str(payload))
        # Tested per FIELD VALUE, not on the raw piece. A chunk's
        # metadata is one long dense line, and on the raw piece it sat
        # inside the window the detector was trying to align, so no
        # borderless table ever matched.
        unstructured = not tables and any(
            looks_like_unstructured_table(v) for _, v in fields if v)
        chunks.append(Chunk(
            index=index,
            has_unstructured_table=unstructured,
            doc_title=re.sub(r"\*+", "", str(title)).strip()[:120],
            fields=tuple(fields),
            tables=tuple(tables),
            text="\n".join(lines),
        ))
    return chunks


def section_of(chunk: Chunk) -> str:
    return _pick(chunk.fields, SECTION_KEYS)
