"""Optional semantic search over one row's passages, for LOCATING text.

There is a line here that matters, and crossing it would quietly break
the audit.

  LOCATING is "where in the retrieval should the adjudicator look?" The
  screener's pointer is often a paraphrase of a section heading, so
  semantic search is the right tool and a near miss is harmless - the
  worst case is a slightly wrong passage, and the adjudicator will say
  it cannot confirm anything from it.

  VERIFYING is "is this quote actually in the source it claims?" That
  must stay LEXICAL, and `evidence/grounding.py` does. A judge quote is
  meant to be copied verbatim; a quote only findable by meaning is one
  that was paraphrased or invented, which is exactly what the grounding
  gate exists to catch. Semantic matching there would mark it grounded
  and the check would silently stop working.

So this module is imported by span extraction and never by grounding.

It is also OFF by default. The cost here is not the index - a row has
perhaps thirty passages and a numpy dot product beats FAISS at that size
- it is one embedding call per row. That only earns its place if the
lexical locator is actually failing, so it runs as a FALLBACK: lexical
first, embeddings only when lexical cannot find the passage. The
embedding bill is then proportional to the failure rate rather than to
the row count.

FAISS is used when installed, numpy otherwise. At this scale they give
the same answer; FAISS starts to matter only if you later index the whole
corpus once and reuse it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:                                              # pragma: no cover
    import numpy as np
except ImportError:                               # pragma: no cover
    np = None                                     # type: ignore

try:                                              # pragma: no cover
    import faiss                                  # type: ignore
    _HAVE_FAISS = True
except ImportError:                               # pragma: no cover
    faiss = None                                  # type: ignore
    _HAVE_FAISS = False


def have_faiss() -> bool:
    return _HAVE_FAISS


class Embedder:
    """OpenAIEmbeddings, loaded lazily so the import is not required."""

    def __init__(self, model: str = "text-embedding-3-small", *,
                 base_url: Optional[str] = None) -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:                # pragma: no cover
            raise ImportError(
                "langchain-openai is needed for semantic span location: "
                "pip install 'policy-audit[openai]'") from exc
        kw = {"model": model}
        if base_url:
            kw["base_url"] = base_url
        self._client = OpenAIEmbeddings(**kw)

    def embed(self, texts: Sequence[str]) -> "np.ndarray":
        vectors = self._client.embed_documents(list(texts))
        arr = np.asarray(vectors, dtype="float32")
        # cosine via inner product, so normalise once here
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)

    def embed_one(self, text: str) -> "np.ndarray":
        return self.embed([text])[0]


@dataclass
class PassageIndex:
    """Searchable passages from a single row's chunks.

    Scoped to one row on purpose. An index built across the whole run
    would happily return a passage from another interaction's retrieval,
    which would be evidence the agent never saw.
    """

    passages: Tuple[str, ...]
    refs: Tuple[Tuple[int, str], ...]             # (chunk index, label)
    _matrix: Optional["np.ndarray"] = None
    _faiss: object = None

    @classmethod
    def build(cls, passages: Sequence[Tuple[int, str, str]],
              embedder: Embedder) -> "PassageIndex":
        """passages = [(chunk_index, label, text)]."""
        if np is None:                            # pragma: no cover
            raise ImportError("numpy is required for semantic location")
        texts = [t for _, _, t in passages]
        refs = tuple((i, label) for i, label, _ in passages)
        if not texts:
            return cls(passages=(), refs=())
        matrix = embedder.embed(texts)
        index = None
        if _HAVE_FAISS:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
        return cls(passages=tuple(texts), refs=refs, _matrix=matrix,
                   _faiss=index)

    def search(self, query_vector: "np.ndarray", k: int = 3
               ) -> List[Tuple[float, int, str, str]]:
        """-> [(score, chunk_index, label, text)], best first."""
        if not self.passages or self._matrix is None:
            return []
        k = min(k, len(self.passages))
        if self._faiss is not None:
            scores, idx = self._faiss.search(
                query_vector.reshape(1, -1).astype("float32"), k)
            pairs = list(zip(scores[0].tolist(), idx[0].tolist()))
        else:
            sims = self._matrix @ query_vector
            order = np.argsort(-sims)[:k]
            pairs = [(float(sims[i]), int(i)) for i in order]
        out = []
        for score, i in pairs:
            if i < 0:
                continue
            chunk_index, label = self.refs[i]
            out.append((round(float(score), 3), chunk_index, label,
                        self.passages[i]))
        return out


_SPLIT = re.compile(r"\n(?=[A-Za-z][\w ]{0,40}:)|\n{2,}")


def passages_from_chunks(chunks: Sequence[object]) -> List[Tuple[int, str, str]]:
    """Break chunks into passages worth indexing separately.

    Field boundaries and table rows, not fixed windows. A table row is
    its own passage because that is the unit a pointer refers to, and
    splitting a table mid-row is how a misread gets manufactured.
    """
    out: List[Tuple[int, str, str]] = []
    for chunk in chunks:
        index = getattr(chunk, "index", 0)
        title = getattr(chunk, "doc_title", "")
        for table in getattr(chunk, "tables", ()):
            header = " | ".join(table.header)
            for row in table.as_rows():
                out.append((index, f"{title} | table", f"{header}\n{row}"))
        for key, value in getattr(chunk, "fields", ()):
            value = str(value or "").strip()
            if not value:
                continue
            for part in _SPLIT.split(value):
                part = part.strip()
                if len(part) > 40 and "|" not in part[:4]:
                    out.append((index, f"{title} | {key}", part))
    return out
