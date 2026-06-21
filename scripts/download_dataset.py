"""Download the Czech news corpus and save a local copy under data/.

Why this exists: pulling the dataset is pure plumbing — we want a single, reproducible
local snapshot (parquet) so the rest of the pipeline (EDA, indexing, BM25) reads from disk
without re-hitting HuggingFace every run. Parquet keeps column types (timestamps, sequences)
intact and is compact (~3 MB corpus).

Run:  uv run python scripts/download_dataset.py
"""

from pathlib import Path

from datasets import load_dataset

DATASET = "simecek/czech_news"
SPLIT = "train"
OUT = Path(__file__).resolve().parent.parent / "data" / "czech_news.parquet"


def main() -> None:
    print(f"Loading {DATASET} [{SPLIT}] ...")
    ds = load_dataset(DATASET, split=SPLIT)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(OUT))

    print(f"Saved {ds.num_rows} rows -> {OUT}")
    print(f"Columns: {ds.column_names}")


if __name__ == "__main__":
    main()
