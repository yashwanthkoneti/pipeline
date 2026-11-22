"""One LLM client, with the failure modes we have actually hit.

Three of them cost real time before:

  Truncated JSON. A reply cut off mid-object is not a model failure, it
  is a budget failure, and 42% of one run died this way. The fix is to
  repair what arrived and retry with a larger budget rather than discard
  the row.

  Parameter naming. Some endpoints take max_tokens, some
  max_completion_tokens. Probing once at startup beats failing on row 1.

  Parameter surface drift. Newer models take max_completion_tokens
  rather than max_tokens and reject a non-default temperature, so the
  client probes once and caches what worked rather than pinning one
  shape in code.

The Protocol at the bottom is what the tests use: the pipeline depends on
the interface, not on any vendor, so an end-to-end test needs no endpoint.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class LLM(Protocol):
    """What the agents need. Anything satisfying this will do."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        ...


# --------------------------------------------------------------------------
# JSON recovery
# --------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """Recover an object from a reply that was cut off.

    Walks the text tracking brace depth while ignoring braces inside
    strings, truncates at the last balanced point and closes what is
    still open. A reply that stopped halfway through row 40 of 46 still
    carries 39 usable decisions, and throwing it away was costing more
    than the truncation.
    """
    if not text:
        return None
    s = _FENCE.sub("", str(text)).strip()
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]

    try:
        return json.loads(s)
    except Exception:
        pass

    depth, in_str, esc, last_good = 0, False, False, None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_good = i + 1

    if last_good:
        try:
            return json.loads(s[:last_good])
        except Exception:
            pass

    # close what is open: drop any partial trailing entry first
    head = s[:last_good] if last_good else s
    head = re.sub(r",\s*\"[^\"]*\"\s*:\s*(?:\{[^{}]*)?$", "", head)
    head = re.sub(r",\s*$", "", head)
    for closing in ("}", "}}", "}}}", "]}", "}]}"):
        try:
            return json.loads(head + closing)
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------
# throttling and retry
# --------------------------------------------------------------------------
@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    truncations: int = 0
    failures: int = 0


class JsonAgent:
    """Wraps an LLM to return parsed JSON, escalating budget on truncation."""

    def __init__(self, llm: LLM, max_tokens: int, *, max_retries: int = 3,
                 retry_base_delay: float = 2.0,
                 min_seconds_between_calls: float = 0.0,
                 name: str = "agent") -> None:
        self.llm = llm
        self.base_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.min_gap = min_seconds_between_calls
        self.name = name
        self.stats = CallStats()
        self._last_call = 0.0

    def _wait(self) -> None:
        if self.min_gap <= 0:
            return
        gap = time.monotonic() - self._last_call
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)

    def ask(self, system: str, user: str,
            expect: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return parsed JSON, or {} with the reason under "_error".

        expect lists keys that must be present; a reply missing them is
        treated as truncated and retried with a larger budget, which is
        what distinguishes "the model disagreed" from "the reply was cut".
        """
        budget = self.base_tokens
        last_error = "no attempt made"

        for attempt in range(self.max_retries):
            self._wait()
            try:
                raw = self.llm.complete(system, user, budget)
                self._last_call = time.monotonic()
                self.stats.calls += 1
            except Exception as exc:                 # transport, rate limit
                last_error = f"{type(exc).__name__}: {exc}"
                self.stats.retries += 1
                time.sleep(self.retry_base_delay * (2 ** attempt))
                continue

            parsed = repair_truncated_json(raw)
            if parsed is None:
                last_error = "reply contained no recoverable JSON"
                self.stats.truncations += 1
                budget = min(budget * 2, self.base_tokens * 4)
                continue

            if expect:
                missing = [k for k in expect if k not in parsed]
                if missing:
                    last_error = f"missing keys: {missing}"
                    self.stats.truncations += 1
                    budget = min(budget * 2, self.base_tokens * 4)
                    # keep what arrived: a partial decision set still beats
                    # discarding the row entirely
                    if attempt == self.max_retries - 1:
                        parsed["_partial"] = missing
                        return parsed
                    continue

            return parsed

        self.stats.failures += 1
        return {"_error": f"{self.name}: {last_error}"}


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------
class OpenAILLM:
    """ChatOpenAI, defensive about which kwargs the model accepts.

    The parameter surface differs across the model families: the newer
    reasoning models take max_completion_tokens rather than max_tokens and
    reject a non-default temperature outright. Rather than hard-code one
    shape, the first call probes and the working combination is cached, so
    a model swap in .env needs no code change.

    JSON mode is requested when the model supports it. Both agents must
    return a JSON object, and a model that guarantees valid JSON removes
    most of what repair_truncated_json exists to clean up - though the
    repair stays, because JSON mode does not prevent a reply being cut off
    at the token limit.
    """

    def __init__(self, model: str, temperature: float = 0.0, *,
                 json_mode: bool = True, timeout: int = 120,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:        # pragma: no cover
            raise ImportError(
                "langchain-openai is needed for the OpenAI client: "
                "pip install 'policy-audit[openai]'") from exc
        self._ChatOpenAI = ChatOpenAI
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        # None = not probed yet
        self._token_kw: Optional[str] = None
        self._send_temperature: Optional[bool] = None
        self._send_json: Optional[bool] = None

    def _kwargs(self, max_tokens: int, token_kw: str,
                temperature: bool, json_mode: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"model": self.model, "timeout": self.timeout,
                              token_kw: max_tokens}
        if temperature:
            kw["temperature"] = self.temperature
        if json_mode:
            kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.base_url:
            kw["base_url"] = self.base_url
        return kw

    def _candidates(self):
        """Parameter shapes to try, modern first.

        Three, not every permutation: the reasoning models take
        max_completion_tokens and reject a non-default temperature, the
        older chat models take max_tokens and accept one, and a gateway
        that rejects JSON mode is the remaining case.
        """
        if self._token_kw is not None:
            yield (self._token_kw, bool(self._send_temperature),
                   bool(self._send_json))
            return
        j = self.json_mode
        yield "max_completion_tokens", False, j
        yield "max_tokens", True, j
        if j:
            yield "max_tokens", True, False

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        last: Exception | None = None
        for token_kw, temperature, json_mode in self._candidates():
            try:
                model = self._ChatOpenAI(
                    **self._kwargs(max_tokens, token_kw, temperature,
                                   json_mode))
                reply = model.invoke([("system", system), ("user", user)])
            except Exception as exc:
                text = f"{type(exc).__name__}: {exc}".lower()
                # a rejected PARAMETER is worth retrying with a different
                # shape; anything else is a real failure and must surface
                # so the retry and backoff in JsonAgent can handle it
                rejected = any(k in text for k in (
                    "unsupported", "unrecognized", "not supported",
                    "unexpected keyword", "invalid_request_error",
                    "does not support", "unknown parameter"))
                if not rejected or self._token_kw is not None:
                    raise
                last = exc
                continue
            self._token_kw = token_kw
            self._send_temperature = temperature
            self._send_json = json_mode
            return getattr(reply, "content", str(reply))
        raise RuntimeError(
            f"no accepted parameter combination for model {self.model}: "
            f"{last}")

    def smoke_test(self) -> None:
        """Fail at startup, not on row 1."""
        out = self.complete("Reply with JSON only.",
                            'Return exactly {"ok": true}', 64)
        if repair_truncated_json(out) is None:
            raise RuntimeError(
                f"model {self.model} did not return parseable JSON: "
                f"{out[:200]}")
        print(f"  {self.model}: ok "
              f"(token param={self._token_kw}, "
              f"temperature={'sent' if self._send_temperature else 'omitted'}, "
              f"json mode={'on' if self._send_json else 'off'})")


# --------------------------------------------------------------------------
# testing
# --------------------------------------------------------------------------
@dataclass
class ScriptedLLM:
    """Returns canned replies in order. Lets the pipeline be tested whole."""

    replies: List[str] = field(default_factory=list)
    seen: List[str] = field(default_factory=list)
    default: str = "{}"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.seen.append(user)
        if self.replies:
            return self.replies.pop(0)
        return self.default
