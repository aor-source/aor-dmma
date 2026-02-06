#!/usr/bin/env python3
"""
AoR-MIR Component 4: FINAL ASSEMBLY
====================================
Combines all component outputs into final analysis JSON.
Calculates God Equation / MIR Score.

Requires: (pure Python - no external deps)
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

def calculate_god_equation(features: dict, aave: dict) -> dict:
    """
    THE GOD EQUATION - Maximum Intelligence Resonance

    MIR = (R * 0.3) + (C * 0.5) + (H * 0.2)

    R = Spectral Rigidity (structural intelligence)
    C = Cultural Density (AAVE markers * 10)
    H = Harmonic Ratio (musical coherence)
    """

    R = min(features.get("spectral_rigidity", 0), 10.0)
    C = min(aave.get("aave_density", 0) * 10, 10.0)
    H = min(features.get("harmonic_ratio", 0.5) * 10, 10.0)

    mir_score = (R * 0.3) + (C * 0.5) + (H * 0.2)

    # Provenance tag
    file_hash = features.get("file_hash", "unknown")[:8]
    tag = f"AOR-MIR-{file_hash}-{int(mir_score*100):04d}"

    return {
        "mir_score": round(mir_score, 4),
        "rigidity_component": round(R * 0.3, 4),
        "cultural_component": round(C * 0.5, 4),
        "harmonic_component": round(H * 0.2, 4),
        "provenance_tag": tag
    }

def assemble(stem: str, work_dir: str, output_dir: str):
    work_dir = Path(work_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Assembling final output: {stem}")

    # Load component outputs
    transcript_file = work_dir / f"{stem}_transcript.json"
    features_file = work_dir / f"{stem}_features.json"
    aave_file = work_dir / f"{stem}_aave.json"

    transcript = {}
    features = {}
    aave = {}

    if transcript_file.exists():
        with open(transcript_file) as f:
            transcript = json.load(f)

    if features_file.exists():
        with open(features_file) as f:
            features = json.load(f)

    if aave_file.exists():
        with open(aave_file) as f:
            aave = json.load(f)

    # Calculate God Equation
    god_eq = calculate_god_equation(features, aave)

    # Build analysis hash
    analysis_str = f"{features.get('spectral_rigidity', 0)}{aave.get('aave_density', 0)}"
    analysis_hash = hashlib.sha256(analysis_str.encode()).hexdigest()[:16]

    # Final assembly
    final = {
        "aor_mir_version": "7.0.0",
        "generated": datetime.now().isoformat(),
        "metadata": {
            "filename": features.get("source_file", stem),
            "duration_seconds": features.get("duration_seconds", 0),
            "sample_rate": features.get("sample_rate", 22050),
            "file_hash": features.get("file_hash", "unknown")
        },
        "fingerprint": {
            "file_hash": features.get("file_hash", "unknown"),
            "analysis_hash": analysis_hash,
            "provenance_tag": god_eq["provenance_tag"]
        },
        "quantum_metrics": {
            "spectral_rigidity": features.get("spectral_rigidity", 0),
            "harmonic_ratio": features.get("harmonic_ratio", 0),
            "tempo_bpm": features.get("tempo_bpm", 0),
            "rms_energy": features.get("rms_mean", 0)
        },
        "aave_analysis": {
            "word_count": aave.get("word_count", 0),
            "unique_terms": aave.get("unique_terms_found", 0),
            "grammar_patterns": aave.get("grammar_patterns_found", 0),
            "density_score": aave.get("aave_density", 0),
            "matched_terms": aave.get("matched_terms", []),
            "corpus_source": aave.get("corpus_source", "DEFAULT")
        },
        "god_equation": god_eq,
        "transcript": {
            "text": transcript.get("text", ""),
            "segments": transcript.get("segments", [])
        }
    }

    # Save
    out_file = output_dir / f"{stem}_aor-mir.json"
    with open(out_file, 'w') as f:
        json.dump(final, f, indent=2)

    print(f"✅ Final output: {out_file}")
    print(f"   MIR Score: {god_eq['mir_score']:.2f}")
    print(f"   Provenance: {god_eq['provenance_tag']}")

    return str(out_file)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python 04_assemble.py <file_stem> <work_dir> <output_dir>")
        sys.exit(1)

    assemble(sys.argv[1], sys.argv[2], sys.argv[3])
