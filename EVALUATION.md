# Evaluation

How we measure whether the search is any good — and, just as importantly, how
we'd do it properly with more time.

## Methodology

**The hard part of IR evaluation is getting relevance judgements** (which
documents *should* a query return?). Hand-labelling is the gold standard but
slow. We use a cheap proxy that the dataset gives us for free:

> **Each article has a `keywords` field. We treat a keyword as a query, and the
> set of articles tagged with that keyword as the "relevant" set.**

Crucially, BM25 searches the article **text** (`headline + brief + content`),
*not* the keywords field. So the test is fair: can lexical retrieval over the
text recover the articles a human editor tagged with that topic?

- **Query set:** `eval/queries.txt` — 18 hand-picked topical keywords. We
  deliberately dropped source/platform tags (`reuters`, `twitter`, `aktuálně.cz`)
  which are noise rather than search topics.
- **Metrics** (`evaluation.py`, all from scratch): Precision@10, MRR, nDCG@10.
- **Run:** `uv run python scripts/evaluate.py`

### Why these three metrics
| Metric | Question it answers | When it matters |
|---|---|---|
| **P@10** | Of the top 10, how many are relevant? | overall result-page quality |
| **MRR** | How high is the *first* relevant hit? | user wants one good answer fast |
| **nDCG@10** | Are relevant docs ranked *as high as possible*? | comparable across queries; rank-aware |

## Results — baseline vs. lemmatization

We formed a hypothesis ("Czech is heavily inflected, so folding word forms with
lemmatization should improve recall") and *measured* it rather than assuming.
Same query set, raw tokens vs. lemmatized pipeline (`scripts/evaluate.py` runs both):

| Pipeline | P@10 | MRR | nDCG@10 |
|---|---|---|---|
| BM25 baseline (raw tokens) | 0.594 | 0.838 | 0.620 |
| **BM25 + lemmatization** | **0.639** | **0.880** | **0.662** |
| **delta** | **+0.044** | **+0.042** | **+0.042** |

Lemmatization helped on all three metrics (~7% relative on P@10) — the hypothesis
held. Honest nuance: it's not a free lunch everywhere — `brno` got slightly
*worse* (0.40 → 0.30), a reminder that lemmatization can occasionally over-fold.
The aggregate win justifies keeping it on by default.

Notable cases:
- **Strong:** `donald trump` (P@10 0.90), `čína` (0.80), `vakcína` (0.80).
- **Weak:** `covid-19` (P@10 0.20) — our tokenizer splits `covid-19` → `covid` +
  `19`, and drops the number. A clear motivation for smarter tokenization.
- `vláda` MRR 0.25 — first relevant only at rank 4, despite decent P@10; ranking,
  not recall, is the issue here.

## Limitations of this methodology (honest)
- **Keyword tags ≠ human relevance.** A topic can be covered without the exact
  keyword string in the text, and a keyword can be tagged loosely. Numbers are a
  *relative* signal for comparing variants, not absolute truth.
- **Single-keyword queries** are easier than real natural-language queries.
- **No graded relevance** — every relevant doc counts equally (binary), so nDCG's
  graded-gain power is underused.

## What we'd do with more time
- **Human-judged set, TREC-style:** pool top-k from several systems, judge
  relevance by hand (or with an LLM-as-judge), reuse across experiments.
- **Graded judgements** (0–3) to exploit nDCG fully.
- **Realistic queries** from real query logs / click models, not keywords.
- **A/B online** with click-through as implicit feedback once in production.
- **Statistical significance** across queries when comparing pipelines.
