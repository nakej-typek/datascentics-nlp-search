"""Retrieval evaluation metrics -- implemented from scratch.

All three answer "how good is a ranked list of results?" but emphasise different
things. Each takes the ranked list of document ids the system returned and the
set of ids that are actually relevant.

  * **Precision@k** -- of the top k results, what fraction are relevant?
    Simple and intuitive; ignores *where* in the top-k the hits are.

  * **MRR** (Mean Reciprocal Rank) -- 1 / (rank of the first relevant result).
    Rewards getting *a* good answer high up. Great when the user mostly wants
    one correct hit near the top (think "I'm feeling lucky").

  * **nDCG@k** (normalised Discounted Cumulative Gain) -- sums relevance with a
    logarithmic positional discount, then normalises by the best possible
    ordering. Rewards putting relevant docs *as high as possible*, and is
    comparable across queries with different numbers of relevant docs.
"""

import math
from collections.abc import Sequence


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for doc_id in topk if doc_id in relevant)
    return hits / k


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    # Binary relevance gains (1 if relevant, else 0), discounted by log2(rank+1).
    # Position 1 -> log2(2)=1 (no discount), deeper positions count for less.
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        gain = 1.0 if doc_id in relevant else 0.0
        dcg += gain / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = dcg_at_k(ranked, relevant, k)
    # Ideal DCG: all relevant docs packed at the very top (capped at k).
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
