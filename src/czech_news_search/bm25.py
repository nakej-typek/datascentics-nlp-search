"""Okapi BM25 scoring -- implemented from scratch (no ``rank_bm25``).

For a query, BM25 sums a per-term score over the documents containing that term:

    score(D, Q) = Σ_t  IDF(t) · ( tf · (k1 + 1) ) / ( tf + k1 · (1 - b + b·|D|/avgdl) )

The two ideas we reasoned about during EDA are baked into that formula:

  * length normalization -- the ``b · |D|/avgdl`` term. Compares a document's
    length to the corpus average so long articles don't win just by having more
    room for a word to appear. ``b`` tunes it: 0 = ignore length, 1 = full
    normalization, 0.75 = the usual compromise.

  * TF saturation -- the ``k1`` term. The 10th occurrence of a word adds less
    than the 2nd; relevance grows with diminishing returns, not linearly.

IDF down-weights terms that appear in many documents (little discriminative
power) and rewards rare ones.
"""

import math

from .index import InvertedIndex


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1  # TF-saturation knob
        self.b = b    # length-normalization knob (0 = off, 1 = full)

    def idf(self, term: str) -> float:
        n = self.index.df(term)
        if n == 0:
            return 0.0
        # +1 inside the log keeps IDF non-negative even for very common terms.
        return math.log(1 + (self.index.N - n + 0.5) / (n + 0.5))

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in query_tokens:
            postings = self.index.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                dl = self.index.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.index.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
