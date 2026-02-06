#!/usr/bin/env python3
"""
AoR Context Indexer - Semantic Search for AAVE Corpora
=======================================================
Creates vector embeddings for contextual search across:
- Lyrics corpus
- Literature corpus
- Leadership speeches
- Social media corpus

Uses TF-IDF (or sentence-transformers) for embeddings, FAISS/numpy for similarity search.
Integrates with AAVE lexicon for cultural context retrieval.

Usage:
    python3 aor_context_index.py build          # Build index from corpora
    python3 aor_context_index.py search "query" # Semantic search
    python3 aor_context_index.py context "term" # Get AAVE context for term
    python3 aor_context_index.py explain "term" # Rich cultural/historical explanation
    python3 aor_context_index.py highaave       # Find high AAVE-density passages
    python3 aor_context_index.py inversions     # Find semantic inversions

Environment Variables:
    USE_TFIDF=1       # Force TF-IDF mode (Python 3.14 compatibility)
    DISABLE_FAISS=1   # Use numpy fallback instead of FAISS

Authors: Jon + Claude - February 2026
"""

import os
import sys
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

# Check environment variables FIRST to avoid importing problematic libraries
USE_TFIDF_MODE = os.environ.get("USE_TFIDF", "").lower() in ("1", "true", "yes")
DISABLE_FAISS_MODE = os.environ.get("DISABLE_FAISS", "").lower() in ("1", "true", "yes")

# TF-IDF fallback (always available via sklearn)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TFIDF = True
except ImportError:
    HAS_TFIDF = False

# Optional imports with fallbacks - skip if USE_TFIDF is set
HAS_SBERT = False
if not USE_TFIDF_MODE:
    try:
        from sentence_transformers import SentenceTransformer
        HAS_SBERT = True
    except ImportError:
        HAS_SBERT = False
    except Exception as e:
        print(f"[!] sentence-transformers import failed: {e}")
        HAS_SBERT = False

HAS_FAISS = False
if not DISABLE_FAISS_MODE:
    try:
        import faiss
        HAS_FAISS = True
    except ImportError:
        HAS_FAISS = False

# Paths
SCRIPT_DIR = Path(__file__).parent
CORPORA_DIR = Path.home() / "Documents" / "aor docs" / "aave_corpora-main"
INDEX_DIR = SCRIPT_DIR / "context_index"
LEXICON_PATH = SCRIPT_DIR / "aave_lexicon.json"

# Index files
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "metadata.pkl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"


class AAVEContextIndex:
    """Semantic index for AAVE corpora with cultural context retrieval"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
        self.lexicon = None
        self.chunk_size = 512  # chars per chunk
        self.overlap = 64

        self._load_lexicon()

    def _load_lexicon(self):
        """Load AAVE lexicon for context enrichment"""
        if LEXICON_PATH.exists():
            with open(LEXICON_PATH) as f:
                self.lexicon = json.load(f)
            print(f"[+] Loaded AAVE lexicon: {LEXICON_PATH.name}")
        else:
            print(f"[!] Lexicon not found: {LEXICON_PATH}")

    def _init_model(self):
        """Initialize embedding model (sentence-transformers or TF-IDF fallback)"""
        if HAS_SBERT:
            if self.model is None:
                print(f"[*] Loading model: {self.model_name}")
                try:
                    self.model = SentenceTransformer(self.model_name)
                except Exception as e:
                    print(f"[!] sentence-transformers failed: {e}")
                    print("[*] Falling back to TF-IDF...")
                    return self._init_tfidf()
            return True
        elif HAS_TFIDF:
            return self._init_tfidf()
        else:
            print("[!] No embedding backend available")
            print("    Install: pip install sentence-transformers OR scikit-learn")
            return False

    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer as fallback"""
        if not HAS_TFIDF:
            return False
        self.use_tfidf = True
        self.tfidf_vectorizer = None  # Will be fit during build
        print("[*] Using TF-IDF vectorizer (sklearn)")
        return True

    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        text = text.strip()

        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Split into chunks
        pos = 0
        chunk_id = 0
        while pos < len(text):
            end = min(pos + self.chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                for delim in ['. ', '? ', '! ', '\n']:
                    last_delim = text.rfind(delim, pos, end)
                    if last_delim > pos + self.chunk_size // 2:
                        end = last_delim + 1
                        break

            chunk_text = text[pos:end].strip()
            if len(chunk_text) > 50:  # Skip tiny chunks
                # Detect AAVE features in chunk
                aave_features = self._detect_aave_features(chunk_text)

                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": chunk_id,
                    "char_start": pos,
                    "char_end": end,
                    "aave_density": aave_features["density"],
                    "aave_terms": aave_features["terms"][:10],  # Top 10
                    "has_inversion": aave_features["has_inversion"]
                })
                chunk_id += 1

            pos = end - self.overlap if end < len(text) else end

        return chunks

    def _detect_aave_features(self, text: str) -> Dict:
        """Detect AAVE features in text chunk"""
        if not self.lexicon:
            return {"density": 0, "terms": [], "has_inversion": False}

        text_lower = text.lower()
        words = re.findall(r"[a-z']+", text_lower)
        word_count = len(words) if words else 1

        found_terms = []
        has_inversion = False

        # Check semantic inversions
        inversions = self.lexicon.get("semantic_inversions", {})
        for term in inversions:
            if term in text_lower:
                found_terms.append(f"~{term}")  # Mark inversions
                has_inversion = True

        # Check slang terms
        for category in ["positive", "negative", "neutral"]:
            terms = self.lexicon.get("slang_terms", {}).get(category, [])
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)

        # Check contractions
        for term in self.lexicon.get("contractions", []):
            if term in text_lower:
                found_terms.append(term)

        density = len(found_terms) / word_count if word_count > 0 else 0

        return {
            "density": round(density, 4),
            "terms": list(set(found_terms)),
            "has_inversion": has_inversion
        }

    def build_index(self):
        """Build vector index from all corpora"""
        if not self._init_model():
            return False

        INDEX_DIR.mkdir(exist_ok=True)
        all_chunks = []

        # Process each corpus
        corpora_paths = [
            (CORPORA_DIR / "corpora" / "lyrics_corpora.txt", "lyrics"),
            (CORPORA_DIR / "corpora" / "literature_corpora.txt", "literature"),
            (CORPORA_DIR / "corpora" / "leadership_corpora.txt", "leadership"),
            (CORPORA_DIR / "corpora" / "social_media_corpora.txt", "social_media"),
        ]

        for corpus_path, corpus_name in corpora_paths:
            if corpus_path.exists():
                print(f"[*] Processing {corpus_name}...")
                with open(corpus_path, 'r', errors='ignore') as f:
                    text = f.read()
                chunks = self._chunk_text(text, corpus_name)
                all_chunks.extend(chunks)
                print(f"    → {len(chunks)} chunks")
            else:
                print(f"[!] Not found: {corpus_path}")

        # Process individual lyric JSON files
        lyrics_dir = CORPORA_DIR / "lyrics"
        if lyrics_dir.exists():
            print(f"[*] Processing individual lyrics...")
            for json_file in lyrics_dir.glob("Lyrics_*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    artist = json_file.stem.replace("Lyrics_", "")
                    for song in data.get("songs", [])[:50]:  # Limit per artist
                        lyrics = song.get("lyrics", "")
                        if lyrics:
                            chunks = self._chunk_text(
                                lyrics,
                                f"lyrics/{artist}/{song.get('title', 'unknown')}"
                            )
                            all_chunks.extend(chunks)
                except Exception as e:
                    continue
            print(f"    → Total lyrics chunks: {len([c for c in all_chunks if 'lyrics/' in c['source']])}")

        if not all_chunks:
            print("[!] No chunks to index")
            return False

        print(f"\n[*] Total chunks: {len(all_chunks)}")
        print(f"[*] Generating embeddings...")

        texts = [c["text"] for c in all_chunks]

        # Check if using TF-IDF or sentence-transformers
        if getattr(self, 'use_tfidf', False):
            # TF-IDF approach
            print("[*] Building TF-IDF matrix...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            embeddings = self.tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)

            # Save vectorizer
            vectorizer_file = INDEX_DIR / "tfidf_vectorizer.pkl"
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"[+] TF-IDF vectorizer saved: {vectorizer_file}")
        else:
            # sentence-transformers approach
            batch_size = 256
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                print(f"    Batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
                batch_emb = self.model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_emb)
            embeddings = np.vstack(all_embeddings)

        print(f"[*] Embedding shape: {embeddings.shape}")

        # Build FAISS index
        if HAS_FAISS:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
            faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
            self.index.add(embeddings)

            faiss.write_index(self.index, str(INDEX_FILE))
            print(f"[+] FAISS index saved: {INDEX_FILE}")
        else:
            print("[!] FAISS not available, using numpy fallback")
            np.save(EMBEDDINGS_FILE, embeddings)

        # Save metadata
        self.metadata = all_chunks
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[+] Metadata saved: {METADATA_FILE}")

        # Stats
        aave_chunks = [c for c in all_chunks if c["aave_density"] > 0.01]
        inversion_chunks = [c for c in all_chunks if c["has_inversion"]]
        print(f"\n[=] Index Statistics:")
        print(f"    Total chunks: {len(all_chunks)}")
        print(f"    AAVE-rich chunks (>1% density): {len(aave_chunks)}")
        print(f"    Chunks with semantic inversions: {len(inversion_chunks)}")

        return True

    def load_index(self) -> bool:
        """Load existing index"""
        vectorizer_file = INDEX_DIR / "tfidf_vectorizer.pkl"

        if not INDEX_FILE.exists() and not EMBEDDINGS_FILE.exists():
            print("[!] No index found. Run: python3 aor_context_index.py build")
            return False

        # Check if TF-IDF was used
        if vectorizer_file.exists():
            self.use_tfidf = True
            with open(vectorizer_file, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print("[+] Loaded TF-IDF vectorizer")
        else:
            if not self._init_model():
                return False

        # Load metadata
        with open(METADATA_FILE, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load index
        if HAS_FAISS and INDEX_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            print(f"[+] Loaded FAISS index: {len(self.metadata)} chunks")
        elif EMBEDDINGS_FILE.exists():
            self.embeddings = np.load(EMBEDDINGS_FILE)
            print(f"[+] Loaded numpy embeddings: {len(self.metadata)} chunks")

        return True

    def search(self, query: str, k: int = 10, min_aave: float = 0.0) -> List[Dict]:
        """Semantic search across corpora"""
        if self.index is None and not hasattr(self, 'embeddings'):
            if not self.load_index():
                return []

        # Encode query based on backend
        if getattr(self, 'use_tfidf', False) and self.tfidf_vectorizer:
            query_embedding = self.tfidf_vectorizer.transform([query]).toarray().astype(np.float32)
        else:
            query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Search
        if HAS_FAISS and self.index:
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, k * 3)  # Over-fetch for filtering
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    chunk = self.metadata[idx].copy()
                    chunk["score"] = float(score)
                    if chunk["aave_density"] >= min_aave:
                        results.append(chunk)
                        if len(results) >= k:
                            break
        else:
            # Numpy fallback (cosine similarity)
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

            emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            emb_norms[emb_norms == 0] = 1  # Avoid division by zero
            embeddings_norm = self.embeddings / emb_norms

            scores = np.dot(embeddings_norm, query_embedding.T).flatten()
            top_indices = np.argsort(scores)[::-1][:k * 3]
            results = []
            for idx in top_indices:
                chunk = self.metadata[idx].copy()
                chunk["score"] = float(scores[idx])
                if chunk["aave_density"] >= min_aave:
                    results.append(chunk)
                    if len(results) >= k:
                        break

        return results

    def get_context(self, term: str, k: int = 5) -> Dict:
        """Get cultural context for an AAVE term"""
        result = {
            "term": term,
            "lexicon_info": None,
            "usage_examples": [],
            "related_passages": []
        }

        # Check lexicon
        if self.lexicon:
            # Check semantic inversions
            if term.lower() in self.lexicon.get("semantic_inversions", {}):
                info = self.lexicon["semantic_inversions"][term.lower()]
                result["lexicon_info"] = {
                    "type": "semantic_inversion",
                    "standard_sentiment": info["standard_sentiment"],
                    "aave_sentiment": info["aave_sentiment"],
                    "correction": info["correction"],
                    "note": f"'{term}' has inverted meaning in AAVE (negative → positive)"
                }

            # Check slang categories
            for category in ["positive", "negative", "neutral"]:
                terms = self.lexicon.get("slang_terms", {}).get(category, [])
                if term.lower() in terms:
                    result["lexicon_info"] = {
                        "type": f"slang_{category}",
                        "category": category
                    }
                    break

        # Search for usage examples
        results = self.search(f'"{term}"', k=k * 2)
        for r in results:
            if term.lower() in r["text"].lower():
                result["usage_examples"].append({
                    "text": r["text"][:300],
                    "source": r["source"],
                    "aave_density": r["aave_density"]
                })
                if len(result["usage_examples"]) >= k:
                    break

        # Get semantically related passages
        result["related_passages"] = self.search(term, k=k)

        return result

    def find_high_aave(self, k: int = 20) -> List[Dict]:
        """Find passages with highest AAVE density"""
        if not self.metadata:
            if not self.load_index():
                return []

        sorted_chunks = sorted(
            self.metadata,
            key=lambda x: x["aave_density"],
            reverse=True
        )
        return sorted_chunks[:k]

    def query_explain(self, term: str) -> Dict:
        """
        Provide rich cultural and historical context for an AAVE term.
        This goes beyond simple definitions to explain:
        - Linguistic origins and evolution
        - Cultural significance
        - Regional variations
        - Semantic shifts
        - Usage in different contexts (hip-hop, literature, everyday speech)
        """
        result = {
            "term": term,
            "linguistic_info": {},
            "cultural_context": [],
            "semantic_analysis": {},
            "regional_info": [],
            "usage_evolution": [],
            "corpus_examples": [],
            "related_terms": []
        }

        term_lower = term.lower()

        if not self.lexicon:
            self._load_lexicon()

        # Check all lexicon categories for the term
        if self.lexicon:
            # Semantic inversions - most culturally significant
            if term_lower in self.lexicon.get("semantic_inversions", {}):
                inv = self.lexicon["semantic_inversions"][term_lower]
                result["linguistic_info"] = {
                    "type": "semantic_inversion",
                    "description": f"'{term}' undergoes semantic inversion in AAVE, where a traditionally negative word becomes positive.",
                    "standard_english_sentiment": inv["standard_sentiment"],
                    "aave_sentiment": inv["aave_sentiment"],
                    "sentiment_shift": inv["correction"],
                    "explanation": f"In Standard American English, '{term}' typically carries negative connotations ({inv['standard_sentiment']:+.1f}). "
                                   f"In AAVE contexts, particularly hip-hop, it often expresses strong approval or excellence ({inv['aave_sentiment']:+.1f}). "
                                   f"This inversion reflects a broader pattern in AAVE of reclaiming and subverting negative terminology."
                }
                result["cultural_context"].append({
                    "aspect": "Linguistic Resistance",
                    "description": "Semantic inversion in AAVE is often interpreted as a form of linguistic resistance - "
                                 "transforming words with historically negative associations into terms of empowerment and praise."
                })

            # Check grammatical patterns
            for pattern_name, pattern_info in self.lexicon.get("grammatical_patterns", {}).items():
                if term_lower in pattern_info.get("examples", []) or term_lower in pattern_name:
                    result["linguistic_info"]["grammatical_feature"] = {
                        "name": pattern_name,
                        "description": pattern_info["description"],
                        "examples": pattern_info["examples"],
                        "note": "This is a systematic grammatical feature of AAVE, not an error or slang."
                    }

            # Check regional variants
            regional_found = []
            for region, terms in self.lexicon.get("regional_variants", {}).items():
                if term_lower in terms:
                    regional_found.append(region)
            if regional_found:
                result["regional_info"] = [{
                    "regions": regional_found,
                    "description": f"'{term}' is particularly associated with {', '.join(regional_found)} AAVE varieties.",
                    "note": "Regional variants reflect the diverse geographical development of AAVE across different African American communities."
                }]

            # Check slang categories
            for category in ["positive", "negative", "intensifiers", "money_terms", "drugs_alcohol",
                            "violence_conflict", "locations", "people", "actions", "vehicles",
                            "clothing_jewelry", "music_performance", "general"]:
                terms = self.lexicon.get("slang_terms", {}).get(category, [])
                if term_lower in terms:
                    result["semantic_analysis"]["category"] = category
                    result["semantic_analysis"]["description"] = f"'{term}' functions as a {category.replace('_', ' ')} term in AAVE."

            # Check classic hip-hop eras
            for era, terms in self.lexicon.get("classic_hip_hop_terms", {}).items():
                if term_lower in terms:
                    result["usage_evolution"].append({
                        "era": era.replace("_", " ").title(),
                        "description": f"'{term}' was particularly prominent during the {era.replace('_', ' ')} era of hip-hop."
                    })

            # Check phonological patterns
            for pattern_name, pattern_info in self.lexicon.get("phonological_patterns", {}).items():
                mappings = pattern_info.get("mappings", {})
                if term_lower in mappings or term_lower in mappings.values():
                    result["linguistic_info"]["phonological_feature"] = {
                        "name": pattern_name,
                        "description": pattern_info["description"],
                        "examples": pattern_info["examples"],
                        "note": "This reflects systematic phonological patterns in AAVE, rooted in historical language development."
                    }

            # Find related terms (terms in same categories)
            related = set()
            for category, terms in self.lexicon.get("slang_terms", {}).items():
                if term_lower in terms:
                    related.update(terms[:20])  # Limit related terms
            related.discard(term_lower)
            result["related_terms"] = list(related)[:15]

        # Get corpus examples
        if not self.metadata:
            self.load_index()

        search_results = self.search(term, k=10)
        for r in search_results:
            if term_lower in r["text"].lower():
                result["corpus_examples"].append({
                    "text": r["text"][:250],
                    "source": r["source"],
                    "aave_density": r["aave_density"],
                    "context_terms": r.get("aave_terms", [])[:5]
                })
                if len(result["corpus_examples"]) >= 5:
                    break

        return result


def print_results(results: List[Dict], show_text: bool = True):
    """Pretty print search results"""
    for i, r in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] Score: {r.get('score', 'N/A'):.4f} | AAVE: {r['aave_density']:.2%}")
        print(f"    Source: {r['source']}")
        if r.get('aave_terms'):
            print(f"    Terms: {', '.join(r['aave_terms'][:5])}")
        if r.get('has_inversion'):
            print(f"    ⚡ Contains semantic inversion")
        if show_text:
            print(f"\n    {r['text'][:400]}...")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    indexer = AAVEContextIndex()

    if cmd == "build":
        indexer.build_index()

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: aor_context_index.py search 'query'")
            return
        query = " ".join(sys.argv[2:])
        print(f"\n[*] Searching: '{query}'")
        results = indexer.search(query, k=10)
        print_results(results)

    elif cmd == "context":
        if len(sys.argv) < 3:
            print("Usage: aor_context_index.py context 'term'")
            return
        term = sys.argv[2]
        print(f"\n[*] Getting context for: '{term}'")
        context = indexer.get_context(term)

        if context["lexicon_info"]:
            print(f"\n[Lexicon Entry]")
            for k, v in context["lexicon_info"].items():
                print(f"  {k}: {v}")

        if context["usage_examples"]:
            print(f"\n[Usage Examples]")
            for ex in context["usage_examples"][:3]:
                print(f"\n  Source: {ex['source']}")
                print(f"  AAVE: {ex['aave_density']:.2%}")
                print(f"  \"{ex['text'][:200]}...\"")

    elif cmd == "highaave":
        print("\n[*] Finding highest AAVE density passages...")
        results = indexer.find_high_aave(k=15)
        print_results(results)

    elif cmd == "inversions":
        print("\n[*] Finding passages with semantic inversions...")
        if indexer.load_index():
            results = [c for c in indexer.metadata if c["has_inversion"]]
            results = sorted(results, key=lambda x: x["aave_density"], reverse=True)[:15]
            print_results(results)

    elif cmd == "explain" or cmd == "query_explain":
        if len(sys.argv) < 3:
            print("Usage: aor_context_index.py explain 'term'")
            return
        term = sys.argv[2]
        print(f"\n{'='*70}")
        print(f"  AAVE CULTURAL/LINGUISTIC EXPLANATION: '{term}'")
        print(f"{'='*70}")

        explanation = indexer.query_explain(term)

        # Linguistic info
        if explanation["linguistic_info"]:
            print(f"\n[LINGUISTIC ANALYSIS]")
            info = explanation["linguistic_info"]
            if "type" in info:
                print(f"  Type: {info['type'].replace('_', ' ').title()}")
            if "description" in info:
                print(f"  {info['description']}")
            if "explanation" in info:
                print(f"\n  {info['explanation']}")
            if "grammatical_feature" in info:
                gf = info["grammatical_feature"]
                print(f"\n  Grammatical Feature: {gf['name']}")
                print(f"  {gf['description']}")
                print(f"  Examples: {', '.join(gf['examples'])}")
            if "phonological_feature" in info:
                pf = info["phonological_feature"]
                print(f"\n  Phonological Feature: {pf['name']}")
                print(f"  {pf['description']}")
                print(f"  Examples: {', '.join(pf['examples'])}")

        # Semantic analysis
        if explanation["semantic_analysis"]:
            print(f"\n[SEMANTIC CATEGORY]")
            sa = explanation["semantic_analysis"]
            if "category" in sa:
                print(f"  Category: {sa['category'].replace('_', ' ').title()}")
            if "description" in sa:
                print(f"  {sa['description']}")

        # Cultural context
        if explanation["cultural_context"]:
            print(f"\n[CULTURAL CONTEXT]")
            for ctx in explanation["cultural_context"]:
                print(f"  {ctx['aspect']}:")
                print(f"    {ctx['description']}")

        # Regional info
        if explanation["regional_info"]:
            print(f"\n[REGIONAL VARIATIONS]")
            for ri in explanation["regional_info"]:
                print(f"  Regions: {', '.join(ri['regions']).replace('_', ' ').title()}")
                print(f"  {ri['description']}")

        # Usage evolution
        if explanation["usage_evolution"]:
            print(f"\n[HISTORICAL USAGE]")
            for ue in explanation["usage_evolution"]:
                print(f"  {ue['era']}: {ue['description']}")

        # Related terms
        if explanation["related_terms"]:
            print(f"\n[RELATED TERMS]")
            print(f"  {', '.join(explanation['related_terms'][:10])}")

        # Corpus examples
        if explanation["corpus_examples"]:
            print(f"\n[CORPUS EXAMPLES]")
            for i, ex in enumerate(explanation["corpus_examples"][:3], 1):
                print(f"\n  [{i}] Source: {ex['source']}")
                print(f"      AAVE Density: {ex['aave_density']:.2%}")
                print(f"      \"{ex['text'][:180]}...\"")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
