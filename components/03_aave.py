#!/usr/bin/env python3
"""
AoR-MIR Component 3: AAVE ANALYSIS
===================================
African American Vernacular English linguistic analysis.
Outputs: {stem}_aave.json

Requires: (pure Python - no external deps)
"""

import sys
import json
import re
from pathlib import Path

# AAVE Knowledge Base
AAVE_LEXICON = {
    "contractions": {
        "ain't", "gon'", "gonna", "gotta", "wanna", "tryna", "finna", "boutta",
        "bout", "cuz", "cause", "y'all", "imma", "lemme", "gimme", "kinda",
        "sorta", "dunno", "wassup", "whatchu", "aint", "yall", "ima", "ion"
    },
    "intensifiers": {
        "hella", "mad", "crazy", "straight", "real", "true", "dead", "deadass",
        "lowkey", "highkey", "super", "extra", "heavy", "deep", "hard", "raw",
        "wild", "sick", "dope", "fire", "lit", "turnt", "hype", "tight",
        "valid", "bussin", "slaps", "hits", "goated", "based", "no cap"
    },
    "markers": {
        "bruh", "bro", "fam", "cuh", "dawg", "homie", "shorty", "playa", "g",
        "og", "blood", "loc", "foo", "mane", "son", "kid", "yo", "aye"
    }
}

GRAMMAR_PATTERNS = [
    (r"\bi\s+be\s+\w+ing\b", "habitual_be"),
    (r"\bhe\s+be\s+\w+ing\b", "habitual_be"),
    (r"\bshe\s+be\s+\w+ing\b", "habitual_be"),
    (r"\bthey\s+be\s+\w+ing\b", "habitual_be"),
    (r"\bi\s+been\s+\w+", "remote_past_been"),
    (r"\bdone\s+\w+ed\b", "completive_done"),
    (r"\bain't\s+no\b", "negative_concord"),
    (r"\bdon't\s+got\s+no\b", "negative_concord"),
]

def analyze_aave(transcript_path: str, output_dir: str, lexicon_path: str = None):
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)

    # Load transcript
    with open(transcript_path, 'r') as f:
        data = json.load(f)

    text = data.get("text", "")
    source_file = data.get("source_file", transcript_path.stem)

    print(f"📝 Analyzing AAVE: {source_file}")

    # Load custom lexicon if provided
    lexicon = AAVE_LEXICON.copy()
    if lexicon_path and Path(lexicon_path).exists():
        with open(lexicon_path, 'r') as f:
            custom = json.load(f)
            for k, v in custom.items():
                if k in lexicon:
                    lexicon[k].update(set(v) if isinstance(v, list) else v)

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_set = set(words)

    # Combine all terms
    all_terms = set()
    for cat in lexicon.values():
        if isinstance(cat, set):
            all_terms.update(cat)

    # Find matches
    matched = [w for w in word_set if w in all_terms]

    # Grammar patterns
    grammar_hits = []
    for pattern, name in GRAMMAR_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            grammar_hits.extend([(m, name) for m in matches])

    # Density
    total_markers = len(matched) + len(grammar_hits)
    density = total_markers / max(len(words), 1)

    output = {
        "source_file": source_file,
        "word_count": len(words),
        "unique_terms_found": len(matched),
        "grammar_patterns_found": len(grammar_hits),
        "aave_density": round(density, 4),
        "matched_terms": sorted(matched),
        "grammar_examples": grammar_hits[:10],
        "corpus_source": "CUSTOM" if lexicon_path else "DEFAULT"
    }

    stem = transcript_path.stem.replace("_transcript", "")
    out_file = output_dir / f"{stem}_aave.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved: {out_file}")
    return str(out_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 03_aave.py <transcript.json> <output_dir> [lexicon.json]")
        sys.exit(1)

    lexicon = sys.argv[3] if len(sys.argv) > 3 else None
    analyze_aave(sys.argv[1], sys.argv[2], lexicon)
