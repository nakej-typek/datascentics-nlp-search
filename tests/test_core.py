"""Sanity tests for the retrieval core (pure, no I/O)."""

import math

import pytest

from czech_news_search.bm25 import BM25
from czech_news_search.evaluation import (
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)
from czech_news_search.index import InvertedIndex
from czech_news_search.preprocessing import tokenize


# --- preprocessing ---------------------------------------------------------

def test_tokenize_drops_stopwords_and_numbers():
    toks = tokenize("Vláda a 2021 rozpočet", lemmatize=False)
    assert "a" not in toks          # stopword
    assert "2021" not in toks       # pure number
    assert "rozpočet" in toks


def test_lemmatization_folds_inflections():
    toks = tokenize("vlády vládě vládou", lemmatize=True)
    assert set(toks) == {"vláda"}   # all inflected forms collapse to one lemma


# --- inverted index --------------------------------------------------------

def test_index_stats():
    idx = InvertedIndex.build({0: ["pes", "kočka"], 1: ["pes", "pes", "auto"], 2: ["auto"]})
    assert idx.N == 3
    assert idx.avgdl == pytest.approx((2 + 3 + 1) / 3)
    assert idx.df("pes") == 2        # in docs 0 and 1
    assert idx.postings["pes"][1] == 2  # term frequency in doc 1


# --- BM25 ------------------------------------------------------------------

def test_bm25_ranks_term_frequency_and_excludes_non_matches():
    idx = InvertedIndex.build({0: ["pes", "kočka"], 1: ["pes", "pes", "auto"], 2: ["auto"]})
    hits = BM25(idx).search(["pes"])
    returned = [doc_id for doc_id, _ in hits]
    assert 2 not in returned         # doc 2 has no "pes"
    assert returned[0] == 1          # doc 1 (tf=2) outranks doc 0 (tf=1)


# --- metrics ---------------------------------------------------------------

def test_metrics_on_known_ranking():
    ranked, relevant = [0, 1, 2, 3], {1, 3}
    assert precision_at_k(ranked, relevant, 2) == 0.5      # 1 hit in top-2
    assert reciprocal_rank(ranked, relevant) == 0.5        # first hit at rank 2

    dcg = 1 / math.log2(3) + 1 / math.log2(5)              # hits at positions 2 and 4
    idcg = 1 / math.log2(2) + 1 / math.log2(3)             # ideal: both at top
    assert ndcg_at_k(ranked, relevant, 4) == pytest.approx(dcg / idcg)
