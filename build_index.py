#!/usr/bin/env python3
"""Wrapper to build index with TF-IDF mode enabled."""
import os
# MUST set before any imports that might trigger sentence-transformers
os.environ["USE_TFIDF"] = "1"
os.environ["DISABLE_FAISS"] = "1"

if __name__ == "__main__":
    # Import AFTER setting env vars
    from aor_context_index import AAVEContextIndex
    indexer = AAVEContextIndex()
    indexer.build_index()
