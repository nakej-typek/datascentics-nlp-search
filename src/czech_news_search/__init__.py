"""Czech news full-text search -- a from-scratch BM25 retrieval layer.

This is the *retrieval* stage of a RAG-style pipeline: given a query it finds
the most relevant news articles. A later step can hand those top results to an
LLM to synthesize an answer.
"""

from .cli import main

__all__ = ["main"]
