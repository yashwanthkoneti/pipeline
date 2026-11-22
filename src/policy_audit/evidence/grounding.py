"""Does a confirmed finding rest on the source it claims to rest on?

The adjudicator is careful but it is still a model, and the one error it
makes that survives every prompt instruction is citing text that is not
there - usually the user's own words, restated as though the policy said
them. That failure is invisible to a reviewer, because the finding looks
fully evidenced.

So the quote is checked in code. Two asymmetric outcomes:

  not in the policy text, but IS in the request or history
      -> the finding cites the user as policy. Not weak, wrong. Dropped,
         and said plainly in the record.

  not locatable anywhere
      -> cannot be verified. Dropped as a defect, kept as unproven for
         calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..domain.models import Evidence, Interaction

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _flat(text: str) -> list[str]:
    return _PUNCT.sub(" ", str(text or "").lower()).split()


def locate(quote: str, haystack: str) -> float:
    """Verbatim containment, then content-word overlap."""
    q, h = _flat(quote), _flat(haystack)
    if not q:
        return 0.0
    if " ".join(q) in " ".join(h):
        return 1.0
    long_q = [w for w in q if len(w) > 3]
    if not long_q:
        return 0.0
    hs = set(h)
    return round(sum(1 for w in long_q if w in hs) / len(long_q), 3)


@dataclass(frozen=True)
class GroundingCheck:
    grounded: bool
    cites_user_text: bool
    best_match: float
    note: str = ""


def check_grounding(evidence: Evidence, ix: Interaction, *,
                    floor: float = 0.55,
                    needs_policy_text: bool = True) -> GroundingCheck:
    """Verify the quote against the policy text it claims to come from."""
    candidates = [c for c in (evidence.quote, evidence.source_states) if c]
    if not candidates:
        return GroundingCheck(False, False, 0.0,
                              "no quote or source statement to verify")

    if not needs_policy_text:
        return GroundingCheck(True, False, 1.0,
                              "this check does not rest on policy text")

    if not ix.chunks:
        return GroundingCheck(
            False, False, 0.0,
            "the finding cites policy text, but nothing was retrieved for "
            "this request")

    in_policy = max(locate(c, ix.chunk_text) for c in candidates)
    if in_policy >= floor:
        return GroundingCheck(True, False, in_policy)

    user_text = f"{ix.request}\n{ix.history_text}"
    in_user = max(locate(c, user_text) for c in candidates)
    if in_user >= floor:
        return GroundingCheck(
            False, True, in_policy,
            f"the evidence quotes the user's own words, not the retrieved "
            f"policy text, for a claim about the policy "
            f"(policy match {in_policy}, user match {in_user})")

    return GroundingCheck(
        False, False, in_policy,
        f"the quoted passage is not locatable in the retrieved policy text "
        f"(best match {in_policy})")
