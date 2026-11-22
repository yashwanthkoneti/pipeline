# policy-audit

Two-stage LLM audit of a policy-retrieval assistant.

```
prepare (deterministic)  →  screener (cheap, 4 gates, every row)
                              →  escalations only
                                →  adjudicator (costly, narrow spans)
                                  →  verdict: expected behaviour by default
```

## The one idea

The two agents are tuned in opposite directions, and everything else
follows from that.

The **screener** is tuned for recall. Its job is not to miss things. A
`suspect` it cannot justify costs one adjudication call; a miss ships a
defect. `clear` is the only verdict that can hide something, because
nothing marked clear is looked at again, so the prompt sets a high bar
for `clear` and a low one for `suspect`. A check it omits becomes
`unsure`, not `clear` — silence is not evidence of absence.

The **adjudicator** is tuned for precision. It reads only the passages
the screener pointed at, and confirms only what it can state as two
complete sentences plus a verbatim quote. It is told that rejecting the
screener is the expected outcome, because a model asked to adjudicate a
colleague's suspicion otherwise ratifies it.

**A defect exists only if the adjudicator confirms it.** Everything else
— nothing suspected, suspicion rejected, suspicion unproven, adjudicator
unreachable — comes out as expected behaviour, with the reason recorded
so the cases stay distinguishable.

That ordering is deliberate. The assistant being audited works from
whatever retrieval handed it and cannot choose its own corpus. A judge
that accuses it whenever the evidence is murky produces a queue nobody
trusts and burns the time of the people who would have fixed the real
defects.

## Labels

Two layers, kept apart:

| layer | what it is | examples |
|---|---|---|
| **theme** | what the **user** experienced. The primary label. An observation, defensible by quoting the request and the response. | Incorrect, Incomplete, Irrelevant, Inconsistent, Unsubstantiated |
| **factor** | what the **system** did. An inference about mechanism, which routes the fix to an owner. Never the headline. | `intent_misread`, `retrieval_coverage_gap`, `excessive_guardrail`, `single_branch_presented_as_rule` |

A theme asserts no mechanism, so it cannot outrun its evidence. "Intent
misinterpretation" as a headline asserted that the assistant
misunderstood the question while resting on evidence that could only
show the answer was thin — which is why it could not be defended in
review. The mechanism claim now lives in the factor, where it is clearly
an inference.

## The four gates, three calls

| gate | asks | sources it reads |
|---|---|---|
| `request` | what did the user ask for in **this** turn | request, **ontology only** |
| `context` | what did the assistant have to work with | history, request, response |
| `route` | was answering / clarifying / declining the right move | history, request, response |
| `answer` | does the answer hold up | everything, including policy text |

`context` and `route` share one call: they read the same sources and
neither needs the policy text, so splitting them bought a second system
prompt and a second copy of the response to ask two more questions. The
gates stay distinct in the taxonomy and in the output.

The `request` gate gets the **ontology, not the policy bodies**. It asks
what the user asked for, and whether a missing fact would change the
answer; for the second part it needs to know what the policy *varies by*,
which the ontology states in ~120 tokens. Sending the bodies duplicated
the largest block in the prompt for no decision it could affect.

Together: **4 calls and 10,383 input tokens per answered row became 3
calls and 6,159 — 41% fewer tokens, 25% fewer calls**, with no change to
what any gate can see that it needs.

The answer gate is skipped on any non-answer route. On a clarification
there is no answer to be incomplete, and asking anyway is how
`answer_incomplete` used to become the cause of a clarification.

## What is deterministic

Everything that can be is, because a model asked to infer a fact we can
compute will sometimes get it wrong, and a wrong fact is invisible.

- **route** — static reply, clarification template, refusal, or answered
- **askback** — whether this turn answers the assistant's own clarifying
  question, read off the previous assistant turn
- **tables** — pipe rows parsed positionally, one row per line, every
  cell written `column=value`, and **an assessment of whether that parse
  can be trusted** (below)
- **ontology** — which documents, sections, table columns, and the
  dimensions the retrieved policy varies by
- **grounding** — whether a confirmed quote is actually in the source it
  claims

### Askback

When a turn answers the assistant's own clarifying question, the user has
**narrowed** the request — they have just said which case they want. The
topics they did not pick are no longer in scope, and an answer confined
to their choice is correct and complete.

Both agents are told this explicitly when it applies, because the
alternative reading — the original broad question still being in scope —
produces a false "incomplete answer" on exactly the rows where the
assistant did the right thing.

More generally: **use the history to understand what the request means;
use the current turn to decide what the answer had to cover.** Those are
different jobs. History is fully in play for comprehension; it does not
add topics the answer owed the user.

### Tables, as they actually arrive

Real chunks come out of PDF extraction and the pipes are not clean. Five
shapes turn up, and they split two ways:

| shape | what the parser does |
|---|---|
| borderless, columns held apart by spaces | no table — detected and routed |
| only the left rule survived (`\|Band  Approver`) | no table — detected and routed |
| ragged pipe counts per row | **parses, and may be wrong** |
| header reprinted mid-table at a page break | **parses, header leaks in as data** |
| merged cell carried down as a blank | **parses, blank reads as "no value"** |

The last three are the dangerous ones. A grid looks exactly as
authoritative whether or not its columns line up, so a bad parse used to
reach the judge unflagged and get the *assistant* blamed for a misreading
that belonged to the parser. Every parse is now assessed, and the two
different failures get two different remedies:

- **The columns may be misaligned** — ragged counts, a headerless column,
  one column, too many blanks. Marked unreliable, which routes the chunk
  to the LLM table reader. If the model's layout verifies cell-for-cell
  against the source it *replaces* the bad grid; if it doesn't, the bad
  grid stays and carries a CAUTION plus the specific reason into the text
  the judge reads, telling it not to raise a table misreading there.
- **The columns are right, the content is ambiguous** — a merged cell.
  A note, and no model call: the model sees the same blank and is
  forbidden to fill it, so re-reading buys nothing. Only the judge can be
  told, so it is.

A repeated header row is dropped, which is unambiguous, and noted,
because it dates the table to a page break.

**Nothing is forward-filled.** A blank under a merged cell might inherit
the row above or be genuinely empty, and guessing manufactures the exact
misreading this audit exists to catch.

Borderless detection looks for a blank column repeating at the **same
offset** down consecutive lines, not for a wide space. Prose wraps, so
its double-spaces land in different places; a column layout doesn't.

#### The LLM table reader

`PA_LLM_TABLES` — `off`, `fallback` (default), or `all`.

The model is never asked what a table **means**, only to lay it out, and
every cell it emits must appear verbatim in the source or the whole read
is rejected. Invention is therefore impossible; misalignment is not,
which is why a model-structured table is marked as such on the chunk.

`compare_tables.py` answers whether you need it at all, on your data:

```bash
python compare_tables.py          # PA_COMPARE_LIMIT=25 caps the calls
```

It prints the parse-quality breakdown for free, then diffs parser against
model on tables the parser already handled. ≥90% identical means the
parser is fine and `fallback` is already spending calls only where
needed. Below that, read the disagreements before changing anything —
`differ` alone does not say the model won.

### No corpus matching

Dropped. Knowing which titles exist elsewhere never distinguished a
corpus gap from a retrieval gap, and its "missing document" suggestions
were noise. What replaces it is the ontology: the assistant cannot be
told what the corpus holds, but it can be told that the retrieved policy
sets out four approvers depending on amount band — which is what turns
"the answer looks fine" into "the policy varies by amount and the answer
named no amount".

## The review sample

A defect the audit cannot substantiate is recorded as **expected
behaviour**, deliberately — an unverifiable finding costs a reviewer more
than the defect it might have caught. The consequence is that every miss
leaves the system silently, so a slice of the clean rows is reviewed
anyway to measure how often that happens.

**Stratified, not random**, because "expected behaviour" is not one
thing. A row nothing was suspected on and a row where a suspicion could
not be settled are both clean in the output and carry completely
different amounts of doubt. The strata run from most doubtful to least:

| stratum | why the row was clean |
|---|---|
| `errored` | the audit errored, so it was never really judged |
| `unverifiable` | suspected, could not be substantiated from the text |
| `no_evidence` | clean, but nothing was retrieved to check against |
| `unsure_screen` | the screener could not tell on at least one check |
| `cleared` | a suspicion was examined on the evidence and rejected |
| `clean` | the screener found nothing to look at |

Sampling each stratum gives a miss rate **per stratum**, which is
actionable — *"of the rows we cleared because nothing could be
substantiated, a third were real defects"* tells you what to fix. A flat
sample spends its budget on the `clean` stratum, where misses are
rarest.

`sample_min_per_stratum` is not a detail: at 10% a stratum of six rows
draws nothing, and the doubtful strata are usually the small ones. On a
447-row clean set that floor produces 46 sampled rows — 10.3% overall,
with every stratum represented.

Selection is a hash of the transaction id, so the same rows are drawn on
every rerun. A queue that reshuffles when the pipeline is rerun cannot be
worked through.

`audit_output.csv` carries `in_review_queue`, `sample_stratum` and
`sample_reason`.

## Run it

```bash
pip install -e '.[openai]'            # add ,faiss only if you enable semantic spans
# .env ships in the repo - edit it, starting with OPENAI_API_KEY
python check_env.py                   # is .env being read, and from where
python demo.py                        # end to end, no endpoint needed
pytest                                # 100 tests, no API key needed
python run.py                         # the real run; or: policy-audit
```

`run.py` needs no `pip install` — it chdir's to the project root and puts
`src` on the path itself, so it works from an IDE.

| script | what it is for |
|---|---|
| `check_env.py` | which `.env` was found and what it resolved to |
| `check_screener.py` | one row through the screener, printing the prompt |
| `demo.py` | the whole cascade on six real cases, fake models |
| `compare_tables.py` | parser vs model on your own tables |
| `profile_prepare.py` | where preparation time goes |

## Configure

Everything comes from **`.env`** — see `.env.example`, which documents
every field. No command-line arguments. A value already in the real
environment wins over the file, so a one-off override still works:

```bash
PA_LIMIT=20 PA_DRY_RUN=true policy-audit
```

`.env` is read by **python-dotenv** at import.
`src/policy_audit/config.py` holds the defaults and is the only module
that touches the environment, so a notebook, a test and a job stay on one
settings object.

**Two you must set to your live wording:**

- `PA_CLARIFY_MARKERS` — the clarification template's own phrasing
- `PA_STATIC_RESPONSES` — the canned replies that end a conversation

**Two worth knowing before the first run:**

- `PA_PREFLIGHT_ONLY=true` parses everything and makes no model calls.
  Read its warnings — each has silently invalidated a run before. A
  `clarify_markers` list that matches nothing empties a whole category
  and makes every askback turn look like a fresh request; chunk parsing
  that loses every title takes the document name out of every finding.
- `PA_DRY_RUN=true` screens without adjudicating, which sizes the
  escalation rate before you commit the expensive model. **Nothing is
  checkpointed on a dry run** and the summary says `*** DRY RUN ***`,
  because a dry run that marked rows complete made the next real run skip
  them.

## Output

| file | shape | for |
|---|---|---|
| `findings.json` | nested, one object per row with a defect, both statements per finding, plus what was examined and cleared | reading |
| `findings.csv` | one row per **finding** | sorting, routing to owners |
| `audit_output.csv` | one row per interaction: your input columns carried through, plus the verdict | joining straight back to the source |
| `checkpoint.jsonl` | per-row, append-only | resuming an interrupted run |

Each finding carries two complete sentences — what the answer claims and
what the policy states — so it can be judged without reopening the
trace. A quote travels with it for verification, but the sentences are
what you read.

## Reading the summary

Two numbers matter more than the headline.

**Why rows came out clean.** "The screener found nothing" and "nothing
could be substantiated" both count as expected behaviour, and they are
very different facts. If the second is large, the expected-behaviour
share is softer than it looks and should be quoted with that caveat.

**Confirmation rate.** If the adjudicator confirms almost everything
escalated, either the screener is too conservative or the adjudicator is
ratifying rather than adjudicating. Check a sample by hand before
trusting the rate. The summary warns about both.

## Models and vectors

Both agents go through `ChatOpenAI`. The client probes the parameter
surface once and caches what worked — newer models take
`max_completion_tokens` rather than `max_tokens` and reject a non-default
temperature — so swapping models in `.env` needs no code change. JSON
mode is requested where supported, which removes most malformed replies
at the source; the truncation repair stays, because JSON mode does not
stop a reply being cut at the token limit.

**Vector search is off by default, and deliberately confined to one job.**

| job | question | method | why |
|---|---|---|---|
| **locating** | where should the adjudicator look? | lexical, then semantic as a fallback | the screener's pointer is often a paraphrase of a heading; a near miss is harmless |
| **verifying** | is this quote really in the source it cites? | **lexical only, always** | a quote findable only by meaning is one that was paraphrased or invented — which is what the grounding gate exists to catch |

Using embeddings for the second job would mark a paraphrased quote as
grounded and the check would silently stop working. `evidence/grounding.py`
never imports the index, and a test asserts it.

Semantic location runs only when the lexical locator finds nothing, so
the embedding bill tracks the *failure* rate rather than the row count.
FAISS is used when installed and numpy otherwise; at a few dozen
passages per row they give the same answer, and FAISS only starts to
matter if you later index the whole corpus once and reuse it. Turn it on
with `PA_SEMANTIC_SPANS=true`.

## Layout

```
src/policy_audit/
  config.py            every tunable
  domain/              taxonomy, checks, dataclasses — no I/O
  parsing/             chunks, tables, ontology, clarification, routing
  agents/              client, prompts, screener, adjudicator, table_reader
  evidence/            span extraction, grounding
  decision/            verdict and run summary
  pipeline/            prepare, enrich, runner, cli
  reporting/           export
tests/                 100 tests, no endpoint required
data/                  put your input CSV here
out/                   run output lands here
```

`ScriptedLLM` in `agents/client.py` satisfies the same `LLM` protocol as
the real client, which is why the whole cascade — screener, adjudicator
and table reader — is testable without an endpoint.
