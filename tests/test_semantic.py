"""Semantic span location. Off by default, so tested with a fake embedder.

The property that matters is the ORDER: lexical first, semantic only when
lexical found nothing. Reversing it would pay for an embedding call on
every row and would make a paraphrased pointer look as good as a verbatim
one.
"""

import numpy as np
import pytest

from policy_audit.domain.models import Interaction, Pointer
from policy_audit.domain.taxonomy import Route
from policy_audit.evidence.index import passages_from_chunks
from policy_audit.evidence.spans import extract_spans
from policy_audit.parsing.chunks import parse_chunks
from policy_audit.parsing.clarification import parse_clarification
from policy_audit.parsing.ontology import build_ontology

CHUNKS = ("**__DOCUMENT_TITLE__:**Leave Policy**__SECTION__:**4 Bereavement"
          "**__TEXT__:**Employees may take up to five days of paid leave on "
          "the death of an immediate family member. Requests go to the line "
          "manager.,"
          "**__DOCUMENT_TITLE__:**Expense Policy**__SECTION__:**2 Limits"
          "**__TEXT__:**|Band|Approver|\n|---|---|\n"
          "|up to 10,000|Manager|\n|above 10,000|Division Head|")


class FakeEmbedder:
    """Bag-of-words vectors. Enough to exercise the plumbing."""

    def __init__(self):
        self.vocab = {}
        self.calls = 0

    def _vec(self, text):
        words = [w for w in text.lower().split() if len(w) > 3]
        for w in words:
            self.vocab.setdefault(w, len(self.vocab))
        v = np.zeros(256, dtype="float32")
        for w in words:
            v[self.vocab[w] % 256] += 1.0
        n = np.linalg.norm(v)
        return v / (n if n else 1.0)

    def embed(self, texts):
        self.calls += 1
        return np.vstack([self._vec(t) for t in texts])

    def embed_one(self, text):
        return self._vec(text)


def make_ix():
    chunks = tuple(parse_chunks(CHUNKS))
    return Interaction(
        transaction_id="T", conversation_id="C",
        request="How much bereavement leave do I get?",
        response="Five days.", history=(), chunks=chunks,
        ontology=build_ontology(chunks), route=Route.ANSWERED,
        clarification=parse_clarification("Five days.", ("please clarify",)))


def test_passages_keep_table_rows_whole():
    passages = passages_from_chunks(make_ix().chunks)
    table_rows = [t for _, label, t in passages if "table" in label]
    assert table_rows
    # every table passage carries its header, so a quote of it means
    # something on its own
    assert all("Band" in t and "Approver" in t for t in table_rows)


def test_lexical_hit_does_not_call_the_embedder():
    embedder = FakeEmbedder()
    spans = extract_spans(make_ix(),
                          [Pointer("chunks", 0, "immediate family member")],
                          embedder=embedder)
    assert spans
    assert "immediate family member" in spans[0].text
    assert embedder.calls == 0


def test_semantic_fallback_runs_only_when_lexical_fails():
    embedder = FakeEmbedder()
    spans = extract_spans(
        make_ix(),
        [Pointer("chunks", 0, "compassionate bereavement entitlement days")],
        embedder=embedder, semantic_floor=0.95)
    assert embedder.calls == 1
    assert spans


def test_no_embedder_means_lexical_only():
    spans = extract_spans(make_ix(),
                          [Pointer("chunks", 0, "something unfindable xyzzy")])
    assert isinstance(spans, list)        # no crash, no embedding call


def test_grounding_never_imports_the_semantic_index():
    """Verifying by meaning would silently defeat the grounding gate."""
    import inspect
    from policy_audit.evidence import grounding
    source = inspect.getsource(grounding)
    assert "index" not in source.split("\n\n")[0]
    assert "Embedder" not in source
    assert "faiss" not in source
