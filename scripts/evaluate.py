"""Evaluate BM25 retrieval against a keyword-seeded relevance set.

Ground truth (a *proxy*, see limitations): for a query like "fotbal", the
relevant documents are the articles tagged with that keyword in the dataset's
`keywords` field. BM25 searches the article *text* (headline+brief+content), not
the keywords -- so this measures whether lexical retrieval over the text can
recover the keyword-labelled articles.

Run:  uv run python scripts/evaluate.py

Honest limitation: keyword tags are a noisy stand-in for human relevance
judgements. A topic may be discussed without the exact keyword string appearing
in the text (e.g. "USA" vs. "spojené státy americké"), and vice-versa. Treat the
numbers as a *relative* signal for comparing pipeline variants, not absolute truth.
"""

from pathlib import Path

from czech_news_search.cli import build_index, load_corpus
from czech_news_search.evaluation import ndcg_at_k, precision_at_k, reciprocal_rank
from czech_news_search.preprocessing import tokenize

K = 10
QUERIES_FILE = Path(__file__).resolve().parent.parent / "eval" / "queries.txt"


def load_queries(path: Path = QUERIES_FILE) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def relevant_docs(df, keyword: str) -> set[int]:
    kw = keyword.lower()
    rel = set()
    for doc_id, tags in enumerate(df["keywords"]):
        if tags is not None and kw in [t.lower().strip() for t in tags]:
            rel.add(doc_id)
    return rel


def evaluate(df, bm25, queries, lemmatize: bool) -> tuple[float, float, float, int]:
    """Run the query set, print per-query rows, return mean metrics + #queries."""
    print(f"\n=== {'WITH lemmatization' if lemmatize else 'baseline (raw tokens)'} "
          f"-- BM25 k1={bm25.k1}, b={bm25.b} @ k={K} ===\n")
    print(f"{'query':<16}{'#rel':>5}{'P@k':>8}{'MRR':>8}{'nDCG':>8}")
    print("-" * 45)

    p_sum = rr_sum = ndcg_sum = 0.0
    n = 0
    for q in queries:
        rel = relevant_docs(df, q)
        if not rel:
            print(f"{q:<16}{0:>5}   (no relevant docs -- skipped)")
            continue
        ranked = [doc_id for doc_id, _ in bm25.search(tokenize(q, lemmatize=lemmatize), k=K)]
        p = precision_at_k(ranked, rel, K)
        rr = reciprocal_rank(ranked, rel)
        ndcg = ndcg_at_k(ranked, rel, K)
        p_sum += p
        rr_sum += rr
        ndcg_sum += ndcg
        n += 1
        print(f"{q:<16}{len(rel):>5}{p:>8.3f}{rr:>8.3f}{ndcg:>8.3f}")

    print("-" * 45)
    print(f"{'MEAN':<16}{'':>5}{p_sum/n:>8.3f}{rr_sum/n:>8.3f}{ndcg_sum/n:>8.3f}")
    return p_sum / n, rr_sum / n, ndcg_sum / n, n


def main() -> None:
    df = load_corpus()
    queries = load_queries()

    # Before/after: same query set, raw tokens vs. lemmatized pipeline.
    base = evaluate(df, build_index(df, lemmatize=False), queries, lemmatize=False)
    lemma = evaluate(df, build_index(df, lemmatize=True), queries, lemmatize=True)

    print("\n========== SUMMARY (lemmatization delta) ==========")
    print(f"{'':<10}{'P@10':>10}{'MRR':>10}{'nDCG@10':>10}")
    print(f"{'baseline':<10}{base[0]:>10.3f}{base[1]:>10.3f}{base[2]:>10.3f}")
    print(f"{'+lemma':<10}{lemma[0]:>10.3f}{lemma[1]:>10.3f}{lemma[2]:>10.3f}")
    print(f"{'delta':<10}{lemma[0]-base[0]:>+10.3f}{lemma[1]-base[1]:>+10.3f}{lemma[2]-base[2]:>+10.3f}")
    print(f"\n({base[3]} queries evaluated)")


if __name__ == "__main__":
    main()
