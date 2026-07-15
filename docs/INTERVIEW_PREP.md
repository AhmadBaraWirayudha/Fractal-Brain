# Interview Prep: Hybrid Cognitive AI Pipeline

Everything in here is grounded in what's actually in this repo -- exact numbers,
exact bugs, exact file names -- not embellishment. If an interviewer digs into
any claim below, it holds up because you can point at the code, the test, or
the CHANGELOG entry that backs it.

A note on framing: parts of this doc describe a debugging/review pass done
with Claude's help (a full read-through and fix pass on the "unified pipeline"
layer). That's worth being upfront about if asked directly -- "I used
AI-assisted code review as part of my process" is a normal, positive thing to
say in 2026, not something to hide. The architecture, the multi-month
iterative build-out (per the CHANGELOG, dozens of rounds), the design
decisions (zero dependencies, the OCLE+fractal_brain merge, what to build
next), and driving/directing every one of those review passes are yours.
Own that distinction confidently: you're the engineer who designed the system
and drove its development; AI-assisted review was one tool in that process,
the same way a linter or a colleague's PR review would be.

---

## 30-second pitch

"I built a pure-Python, zero-external-dependency AI system with two halves I
designed separately and then merged: a from-scratch neural architecture
(custom attention, mixture-of-experts routing, retrieval-augmented generation,
Hebbian plasticity, a JEPA-style predictive loss) and a closed-loop agent
(retrieve -> decompose -> plan -> generate -> reflect -> learn from
correction). Zero dependencies was a constraint, not a limitation -- it meant
implementing things like SGD/Adam, a BPE tokenizer, and SVD by hand instead of
importing them, which is where most of the actual learning happened. It's
tested with 161 checks across a smoke suite and pytest, all currently green,
and I've been through several honest audit passes finding and fixing real
bugs rather than just adding features."

Adjust the last sentence's tone to the room -- confident but not oversold; see
"Honest limitations" below for what to volunteer if asked to go deeper.

---

## What it actually is (in plain language)

Two systems merged into one pipeline:

1. **`fractal_brain/`** -- the neural architecture. Tokens go in, a mixture of
   small transformer-style "experts" plus several custom mechanisms
   (a fractal Markov state machine, a Hebbian/BCM plasticity rule, a
   retrieval-fusion path, a JEPA-style auxiliary predictive loss, PID-based
   gain control) combine to produce logits and a loss. Trained with a
   hand-written Adam optimizer, mini-batching, gradient clipping, LR
   schedules, checkpointing to SQLite -- no PyTorch/NumPy/anything.
2. **The closed-loop engine** (`engine.py` + neighbors) -- a much simpler,
   classical agent loop: embed the query (hash-based, not a neural embedding),
   retrieve similar past examples from a small memory store, classify intent
   with a keyword-based decomposer, pick a next action via a small learned
   Markov planner, and generate an answer. The generator is honestly a
   rule-based fallback, not a model -- see limitations below.
3. **The unification layer** (`ai_pipeline.py` / `hybrid_cli.py`) -- ties the
   two together: `fractal_brain` scores/contributes a "cognitive context"
   signal, the closed-loop engine produces the actual answer, and everything
   is wrapped with session state, reflection, and a feedback/teaching loop.

If you only remember one distinction: **`fractal_brain` is the deep,
from-scratch ML work; the closed-loop engine is the simpler, classical-AI
agent scaffolding around it.** Don't imply the closed-loop half uses any
learned embeddings or a real LLM -- it doesn't, and that's fine to say
directly (see below).

---

## Numbers worth knowing

| Metric | Value |
| --- | --- |
| External dependencies | Zero (`requirements.txt` literally says so) |
| `fractal_brain` smoke suite | 137 checks, all passing |
| pytest suite | 24 tests, all passing (was 14 before the latest review pass) |
| Total automated checks | 161 |
| Lines of custom neural-net code | `fractal_brain/` is dozens of files implementing attention, MoE, optimizers, autograd-free backprop, tokenization, and storage from scratch |
| Bugs found+fixed in the latest full review | 5 correctness bugs, 10 completeness/testing/docs gaps |
| CHANGELOG entries | A long, detailed history -- crash bugs, silent-no-op bugs, and "what I deliberately did not change" are each their own section, which is itself a good thing to point to (see below) |

Use the 137/24/161 numbers specifically and precisely if asked "how do you
know it works" -- specific numbers read as credible; "lots of tests" doesn't.

---

## Three strong stories to tell (STAR-shaped)

### 1. The bug that made the whole "agentic" story fake -- and how you found it
**Situation:** After merging the two subsystems into one pipeline, everything
*looked* like it worked -- retrieval ran, planning ran, the system produced
an answer.
**Task:** Verify the merge actually worked end to end, not just that each
piece ran without crashing.
**Action:** Actually ran the pipeline on a known example (an integral with a
worked solution sitting in memory) and read the literal output, instead of
trusting that "no exceptions raised" meant "correct." Found that the answer
was a fixed template string, unrelated to the question, every single time --
traced it to the generator function only ever reading the `Task:` line back
out of the prompt and discarding retrieval/plan context entirely.
**Result:** Fixed by threading the actual retrieved documents (with their
similarity scores) and the plan into the generator, with a tunable confidence
threshold deciding when to trust a retrieved match. Added a regression test
that asserts the *correct number* appears in the output, not just that the
output is non-empty -- because the old test would have passed even with the
bug present.
**Why this is a good story:** it's about the difference between "it runs" and
"it's correct," and about writing a test that would have actually caught the
bug, not just a shape check.

### 2. The database that gave away a bug before you even opened the code
**Situation:** A small (4-record) bootstrap dataset is supposed to seed a
SQLite memory store on startup.
**Task/Action:** Noticed the shipped database file was implausibly large for
4 records, opened it directly with a SQL query instead of guessing, and found
38 rows -- each of the 4 records duplicated about 9 times. Traced it to the
bootstrap function having no idempotency check: every process start
re-inserted every record with a fresh random ID.
**Result:** Fixed with a content-derived stable ID (a hash of the record) and
`INSERT OR IGNORE`, verified by initializing the engine three times against
the same database file and confirming the row count didn't move. This is a
good "attention to detail / read the data, don't just read the code" story.

### 3. Taking external critique and actually fixing the substance
**Situation:** An earlier review of `fractal_brain` pointed out that two
subsystems -- retrieval-augmented fusion and the JEPA predictive loss --
were computed every step but their outputs were silently discarded: RAG's
fused representation never reached the gating decision, and JEPA's loss was
reported without ever updating JEPA's own weights.
**Task/Action:** Rather than just acknowledging the critique, wired RAG's
fusion output into the actual gate computation with a new trained weight
matrix, and derived the real gate gradient by hand for the whole four-source
additive gating scheme, verified against numerical (finite-difference)
gradient checking.
**Result:** Documented in the CHANGELOG with the derivation and how the first
verification attempt actually failed in a misleading way before the real bug
was found -- a good example of not just accepting the first "looks right"
result.
**Why this is a good story:** shows you can take critical feedback, not get
defensive, and follow through on the hard part (the math), not just the easy
part (acknowledging it in a comment).

---

## Likely questions and how to answer them

**"Walk me through this project."**
Use the 30-second pitch, then let them steer into whichever half (the neural
architecture or the agent loop) they want to go deeper on.

**"What's the hardest bug or technical challenge you dealt with?"**
Story #1 or #3 above. Pick #3 if they seem more ML-research-oriented (it's
about deriving a gradient by hand and verifying it numerically); pick #1 if
they seem more systems/product-oriented (it's about output correctness and
test quality).

**"Why zero external dependencies? Wasn't that just extra work?"**
A reasonable, honest answer: it forces you to actually understand what
libraries like PyTorch or a real tokenizer are doing under the hood, rather
than calling an API. Backprop through attention, an Adam optimizer, a BPE
tokenizer, and an SVD via power iteration are all things you had to derive
and implement, not just import. It's slower and less capable than using real
libraries -- that trade-off is explicit and intentional, not a limitation you
were unaware of.

**"What would you do differently, or what's still missing?"**
Answer honestly and specifically (interviewers notice vague hand-waving
here) -- see "Honest limitations" below. Framing it as "here's what I know is
incomplete and why I haven't done it yet" reads far better than pretending
everything is finished.

**"How do you know it actually works? How did you test it?"**
Two-tier testing: a 137-check smoke suite for the neural architecture, plus a
24-test pytest suite (engine behavior, CLI, regressions) that now also runs
in CI -- worth mentioning that CI previously *didn't* run the pytest half at
all, and fixing that was itself part of the review. If asked for something
concrete: "I have a specific regression test that asserts the correct answer
appears in the output, because a previous test only checked the output was
non-empty and would have passed even with a real bug present."

**"How would you add a real language model / real embeddings to this?"**
The seams are already there: `SharedMoEBackbone` in `moe_model.py` has a
single `initialize()`/`generate()` interface that's currently hard-coded to a
rule-based fallback -- swapping in a real model means implementing that same
interface against, say, a local model or an API call, without touching the
retrieval/planning/decomposition code around it. Similarly, `TextEmbedder`'s
hash-based embedding could be swapped for a real sentence embedding model
behind the same `embed_text()` interface. Being able to point at the exact
seam is a much stronger answer than "I'd rewrite it."

**"How would you scale this / what breaks at scale?"**
The honest answer: retrieval is currently a brute-force, in-memory linear
scan over a hash-based embedding -- fine for a handful of documents, would
need a real ANN index (and real embeddings) past a few thousand. The planner
is a small learned Markov chain over states derived from a tiny bootstrap
set; it would need a much larger, more diverse training set to generalize.

**"Tell me about a time you incorporated critical feedback."**
Story #3. It's real, specific, and shows technical follow-through rather
than just taking the note.

---

## Honest limitations (know these cold -- don't get caught flat-footed)

Say these plainly if asked; trying to talk around them reads worse than
just stating them.

- **The generator is rule-based, not a language model.** It can surface a
  retrieved answer or list plan steps; it can't reason beyond that.
- **Retrieval and embeddings are hash-based (bag-of-hashed-words), not
  semantic.** Matching is close to keyword overlap. Great for near-exact
  repeats of known content, unreliable for paraphrases.
- **Attention and feed-forward weights inside `fractal_brain`'s transformer
  experts are still frozen at random initialization** -- only the output
  projections, gate, PID gains, and a couple of other components have real
  gradients flowing to them today. This is documented and deliberately
  scoped as future work, not an oversight you're unaware of.
- **The planner and decomposer are small heuristics**, not learned in any
  deep sense -- a handful of keyword buckets and a Markov chain fit on a
  four-record bootstrap dataset.
- **The "unified pipeline" -- the part that merges the two subsystems -- is
  the least battle-tested part of the codebase**, and was where nearly every
  bug in the most recent review pass was found. The core neural architecture
  has a much longer, more rigorously verified history.

The throughline for all of these: you know exactly where the rough edges
are, why they exist, and roughly what it would take to fix each one. That's
a much stronger position than either overselling or being vague.

---

## Quick glossary (so you don't blank on your own terms)

- **Fractal Markov state**: a custom recursive/nested Markov-chain-like state
  representation feeding into the model's gating decision -- descriptive
  branding for a custom mechanism, not a standard published method (worth
  saying if asked "is this a known technique" -- no, and that's fine).
- **Mixture of (transformer) experts**: several small transformer-style
  sub-networks ("experts"), with a learned gate deciding how much each
  contributes per input.
- **BCM plasticity**: a Hebbian-style learning rule (Bienenstock-Cooper-Munro)
  used here to update some weights based on activity correlation, alongside
  (not instead of) ordinary gradient-based updates.
- **JEPA**: "Joint-Embedding Predictive Architecture" -- an auxiliary loss
  that predicts a target representation from a context representation,
  rather than predicting raw tokens. In this repo, the loss is computed but
  (as of the last check) doesn't yet update JEPA's own encoder weights --
  see limitations.
- **Wormhole / tentacles**: project-specific names for two of the custom
  gating/routing mechanisms feeding into expert selection -- again, branding
  for bespoke components, not standard terminology.
- **PID gains**: proportional-integral-derivative control (borrowed from
  classical control theory) used here to adaptively tune something in the
  training loop rather than a physical system.
- **RAG (retrieval-augmented generation/fusion)**: retrieving related stored
  information and folding it into a computation -- used in two different
  places in this repo (the neural architecture's internal RAG-fusion path,
  and the closed-loop engine's separate memory retrieval), which is worth
  distinguishing if asked, since they're not the same mechanism.

---

## Delivery tips

- Lead with the merge/architecture story, not the bug list -- the bug list
  is your evidence of rigor, brought up when asked "how do you know it
  works" or "what's hard about this," not the headline.
- If you don't know the answer to something specific about a file, it's
  fine to say "let me think through that" and reason from the architecture
  out loud -- that's normally more impressive than a memorized answer.
- Don't inflate the "AI system" framing -- it's an agent pipeline with a
  custom neural component, not a general intelligence. Interviewers with any
  ML background will respect the precise version of that claim far more
  than the inflated one.
