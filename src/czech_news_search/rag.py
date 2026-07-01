"""RAG finale: retrieve top-k articles, then let Claude answer *from them*.

This is the Retrieval-Augmented Generation layer on top of our BM25 retriever:

    query --> BM25 retrieves top-k articles --> Claude reads them --> grounded answer

The LLM call shells out to the **Claude CLI** (`claude -p`, headless print mode),
so it runs on the existing Claude subscription -- no API key, no per-token cost.
In production you'd swap this for a hosted model / the Anthropic API (see the
architecture diagram in the README); the retrieval half stays exactly the same.

    uv run czech-news-answer "Co se psalo o povodních?"
"""

import argparse
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
import textwrap

from .cli import build_index, load_corpus
from .preprocessing import tokenize

# Keep each article excerpt bounded so the prompt stays a sane size.
EXCERPT_CHARS = 1200

PROMPT_TEMPLATE = """\
Jsi asistent, který odpovídá na dotaz POUZE na základě níže uvedených novinových
článků. Odpovídej česky, stručně a věcně. U tvrzení cituj zdroj číslem, např. [1].
Pokud články odpověď neobsahují, napiš to a nevymýšlej si.

Dotaz: {query}

Články:
{context}

Odpověď:"""


def build_context(df, hits) -> str:
    blocks = []
    for rank, (doc_id, score) in enumerate(hits, 1):
        row = df.iloc[doc_id]
        excerpt = textwrap.shorten(
            str(row["content"]).replace("\n", " "), width=EXCERPT_CHARS, placeholder=" …"
        )
        blocks.append(
            f"[{rank}] {row['headline']}\n"
            f"    URL: {row['url']}\n"
            f"    {excerpt}"
        )
    return "\n\n".join(blocks)


def ask_claude(prompt: str) -> str:
    """Send the prompt to the Claude CLI (subscription) and return the answer."""
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        return ("[!] 'claude' CLI nenalezeno v PATH -- nainstaluj/přihlas Claude Code, "
                "nebo přepni RAG na lokální model.")
    result = subprocess.run(
        [claude_bin, "-p"],
        input=prompt,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=180,
    )
    if result.returncode != 0:
        return f"[!] claude CLI selhalo (exit {result.returncode}):\n{result.stderr.strip()}"
    return result.stdout.strip()


def answer(query: str, k: int = 3) -> str:
    df = load_corpus()
    bm25 = build_index(df)
    hits = bm25.search(tokenize(query), k=k)
    if not hits:
        return "Žádný článek neodpovídá dotazu, nemám z čeho čerpat."
    context = build_context(df, hits)
    return ask_claude(PROMPT_TEMPLATE.format(query=query, context=context))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG answer over Czech news (BM25 + Claude).")
    parser.add_argument("query", help="question in Czech")
    parser.add_argument("-k", type=int, default=3, help="how many articles to ground on")
    args = parser.parse_args()

    print(f'\nDotaz: "{args.query}"\n')
    print("Hledám relevantní články a ptám se Clauda…\n")
    print(answer(args.query, k=args.k))


if __name__ == "__main__":
    main()
