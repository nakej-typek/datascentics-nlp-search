"""Inverted index -- the core data structure behind fast full-text search.

Naive search would scan every document for every query (O(N) per query). An
inverted index flips that around: for each *term* we precompute which documents
contain it and how often. BM25 then only has to score the documents that
actually contain a query term -- usually a tiny fraction of the corpus.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class InvertedIndex:
    # term -> { doc_id -> term frequency of that term in that doc }
    postings: dict[str, dict[int, int]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    doc_len: dict[int, int] = field(default_factory=dict)  # doc_id -> #tokens
    N: int = 0          # number of documents in the corpus
    avgdl: float = 0.0  # average document length in tokens -- needed by BM25

    @classmethod
    def build(cls, tokenized_docs: dict[int, list[str]]) -> "InvertedIndex":
        idx = cls()
        idx.N = len(tokenized_docs)
        total_len = 0
        for doc_id, tokens in tokenized_docs.items():
            idx.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            for term, tf in Counter(tokens).items():
                idx.postings[term][doc_id] = tf
        idx.avgdl = total_len / idx.N if idx.N else 0.0
        return idx

    def df(self, term: str) -> int:
        """Document frequency: in how many documents the term appears."""
        return len(self.postings.get(term, {}))
