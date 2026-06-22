"""Tiny CLI: build the index from the local corpus and run a BM25 query.

    uv run czech-news-search "povodně na Moravě"
    uv run czech-news-search -k 3 "vláda rozpočet"
"""

import argparse
from pathlib import Path

import pandas as pd

from .bm25 import BM25
from .index import InvertedIndex
from .preprocessing import tokenize

DATA = Path(__file__).resolve().parents[2] / "data" / "czech_news.parquet"


def load_corpus(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # What we index: headline + brief + content. Concatenating gives recall on
    # both the title and the body. (Field weighting -- boosting headline -- is a
    # later refinement; for now every field counts equally.)
    df["search_text"] = (
        df["headline"].fillna("") + " "
        + df["brief"].fillna("") + " "
        + df["content"].fillna("")
    )
    return df


def build_index(df: pd.DataFrame, lemmatize: bool = True) -> BM25:
    tokenized = {
        i: tokenize(text, lemmatize=lemmatize)
        for i, text in enumerate(df["search_text"])
    }
    index = InvertedIndex.build(tokenized)
    return BM25(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 search over Czech news.")
    parser.add_argument("query", help="search query (in Czech)")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    args = parser.parse_args()

    df = load_corpus()
    bm25 = build_index(df)
    hits = bm25.search(tokenize(args.query), k=args.k)

    print(f'\nTop {len(hits)} for: "{args.query}"\n')
    if not hits:
        print("  (no documents matched any query term)")
        return
    for rank, (doc_id, score) in enumerate(hits, 1):
        row = df.iloc[doc_id]
        print(f"{rank}. [{score:5.2f}] {row['headline']}")
        print(f"          {row['url']}")


if __name__ == "__main__":
    main()
