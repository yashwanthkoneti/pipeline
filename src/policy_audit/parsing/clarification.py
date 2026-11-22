"""Clarification templates, askback detection, and selection resolution.

The live template is:

    <preamble containing a sentence that starts "please clarify">

    1. Question?

    Option a: ...
    Option b: ...
    Option c: ...

    2. Question?
    ...

Three things here are easy to get wrong and all three were:

  The marker ALONE means a clarification has started. The question list
  is useful detail, not a precondition - if the body parses badly the
  route is still `clarified`, and treating a parse failure as "not a
  clarification" silently moves rows into `answered`, where every
  answer-shaped check then runs against a response that answered
  nothing.

  `Option a:` is capitalised in the live template. The option pattern
  was case-sensitive, so it matched `option a:` and nothing else, and
  every real clarification came back with zero options.

  The user replies `1a, 2b, 3c` - a REFERENCE, not a statement. It
  carries no subject and no vocabulary, so it has to be resolved against
  the previous turn's option list before anything can reason about it.
  Resolved here, in code, so the judge is handed a request it can
  actually read.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..domain.models import Clarification, Turn

# A numbered line is a question in this template, never an option.
_Q_LINE = re.compile(
    r"^\s*(?:\(?(\d{1,2})\s*[\).:]|[-*•●])\s*(.+?)\s*$")
# re.I matters: the live template writes "Option a:".
_O_LINE = re.compile(
    r"^\s*(?:option\s*([0-9a-zA-Z])\s*[:.\)]|\(?([a-hA-H])\s*[\).]|"
    r"[-*•●])\s*(.*)$", re.I)

# "1a, 2b, 3c" and the shapes around it: "1-a", "1a 2b 3c", "Q1: a".
# re.I, for the same reason _O_LINE needs it: the live text capitalises.
_SELECTION = re.compile(r"\b(?:q\s*)?([1-9])\s*[-.:)]?\s*([a-h])\b", re.I)
# a bare "a, b, c" with no question numbers, taken positionally
_BARE_LETTERS = re.compile(r"^\s*([a-hA-H])(?:\s*[,/]\s*([a-hA-H]))*\s*$",
                           re.I)


def parse_clarification(response: str,
                        markers: Sequence[str]) -> Clarification:
    """Pull the template apart. An option's text may be empty."""
    text = str(response or "")
    low = text.lower()
    marker = next((m for m in markers if m.lower() in low), "")

    questions: List[Tuple[str, List[str]]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        om = _O_LINE.match(line)
        is_option = bool(om) and bool(
            om.group(1) or om.group(2) or line[:1] in "-*•●")
        qm = _Q_LINE.match(line)
        if qm and qm.group(1):
            questions.append((qm.group(2).strip(), []))
            continue
        if is_option:
            if not questions:
                questions.append(("", []))
            questions[-1][1].append((om.group(3) or "").strip())
            continue
        if line.endswith("?"):
            questions.append((line, []))

    return Clarification(
        marker=marker,
        questions=tuple((q, tuple(o)) for q, o in questions),
    )


def has_clarify_marker(response: str, markers: Sequence[str]) -> bool:
    """Did the agent open a clarification?

    The marker alone decides it. A sentence beginning "please clarify"
    means the clarification flow has started whatever else the response
    contains and however badly the body parses.
    """
    low = str(response or "").lower()
    return any(m.lower() in low for m in markers)


def previous_assistant_turn(history: Sequence[Turn]) -> str:
    for turn in reversed(history):
        if turn.role == "assistant":
            return turn.content
    return ""


def detect_askback(history: Sequence[Turn],
                   markers: Sequence[str]) -> Tuple[bool, str]:
    """Is this turn the user answering the agent's clarifying question?

    Deterministic - it reads the previous assistant turn for the marker.
    Returns (is_askback, the question text that was asked).
    """
    prev = previous_assistant_turn(history)
    if not prev or not has_clarify_marker(prev, markers):
        return False, ""
    parsed = parse_clarification(prev, markers)
    asked = " / ".join(q for q, _ in parsed.questions if q)
    return True, asked


def parse_selections(request: str) -> List[Tuple[int, str]]:
    """`1a, 2b, 3c` -> [(1, 'a'), (2, 'b'), (3, 'c')].

    Also handles `1-a`, `1a 2b 3c`, `Q1: a`, and a bare `a, b, c` taken
    positionally as question 1, 2, 3.
    """
    text = str(request or "").strip()
    if not text:
        return []

    pairs = [(int(n), letter.lower())
             for n, letter in _SELECTION.findall(text)]
    if pairs:
        seen, out = set(), []
        for number, letter in pairs:
            if number not in seen:
                seen.add(number)
                out.append((number, letter))
        return out

    # a bare "a, b, c": position IS the question number
    if _BARE_LETTERS.match(text):
        letters = [x.strip().lower()
                   for x in re.split(r"[,/\s]+", text) if x.strip()]
        return [(i, x) for i, x in enumerate(letters, 1)]
    return []


def resolve_selections(request: str, clarification: Clarification
                       ) -> Tuple[Dict[str, str], str]:
    """Turn a coded reply into the options it picked.

    Returns ({"1a": "Listed equity", ...}, a readable request).

    This is what makes an askback turn auditable. `1a, 2b, 3c` on its own
    has no subject, so a judge handed it cannot assess relevance,
    coverage or anything else; every check would be guessing. Resolved
    against the option list it refers to, the same turn becomes
    "instrument: Listed equity; region: APAC; covered: I don't know".
    """
    picks = parse_selections(request)
    if not picks or not clarification.questions:
        return {}, ""

    resolved: Dict[str, str] = {}
    lines: List[str] = []
    for number, letter in picks:
        index = number - 1
        if not (0 <= index < len(clarification.questions)):
            continue
        question, options = clarification.questions[index]
        slot = ord(letter) - ord("a")
        if not (0 <= slot < len(options)):
            continue
        chosen = options[slot]
        resolved[f"{number}{letter}"] = chosen
        label = question.rstrip("?").strip() or f"question {number}"
        lines.append(f"{label}: {chosen}")
    return resolved, "; ".join(lines)


def answers_an_option(request: str, history: Sequence[Turn],
                      markers: Sequence[str]) -> Optional[str]:
    """Which offered option this turn picks, by text rather than by code.

    For a user who types the option out instead of coding it. Tried only
    after resolve_selections finds no coded reply.
    """
    prev = previous_assistant_turn(history)
    if not prev:
        return None
    parsed = parse_clarification(prev, markers)
    if not parsed.questions:
        return None
    req_words = set(re.sub(r"[^a-z0-9 ]+", " ", str(request).lower()).split())
    best, best_score = None, 0.0
    for _, options in parsed.questions:
        for opt in options:
            words = {w for w in
                     re.sub(r"[^a-z0-9 ]+", " ", opt.lower()).split()
                     if len(w) > 2}
            if not words:
                continue
            score = len(words & req_words) / len(words)
            if score > best_score:
                best, best_score = opt, score
    return best if best_score >= 0.5 else None
