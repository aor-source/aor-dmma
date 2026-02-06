#!/usr/bin/env python3
"""
AoR-MIR Component 2: AUDIO FEATURES
====================================
Librosa-based DSP feature extraction.
Outputs: {stem}_features.json

Requires: librosa, numpy, scipy
"""

import sys
import json
import hashlib
from pathlib import Path

def extract_features(audio_path: str, output_dir: str):
    import numpy as np
    import librosa

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    print(f"🔊 Extracting features: {audio_path.name}")

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # File hash for provenance
    with open(audio_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Harmonic-percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_ratio = float(y_harm.var() / (y.var() + 1e-10))

    # Tempo
    tempo, beats = librosa.beat.beat_track(y=y_perc, sr=sr)
    tempo_val = float(tempo) if hasattr(tempo, '__float__') else float(tempo[0])

    # Spectral rigidity (eigenvalue analysis)
    try:
        corr = np.corrcoef(chroma)
        eigs = np.linalg.eigvalsh(corr)
        eigs_sorted = np.sort(eigs)
        spacings = np.diff(eigs_sorted)
        rigidity = float(np.std(spacings) / (np.mean(spacings) + 1e-10))
    except:
        rigidity = 0.0

    # RMS energy stats
    rms = librosa.feature.rms(y=y)

    output = {
        "source_file": audio_path.name,
        "file_hash": file_hash,
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "spectral_rigidity": round(min(rigidity, 10.0), 4),
        "harmonic_ratio": round(harmonic_ratio, 4),
        "tempo_bpm": round(tempo_val, 1),
        "rms_mean": round(float(rms.mean()), 6),
        "rms_std": round(float(rms.std()), 6),
        "chroma_mean": [round(float(x), 4) for x in chroma.mean(axis=1)],
        "mfcc_mean": [round(float(x), 4) for x in mfcc.mean(axis=1)]
    }

    out_file = output_dir / f"{audio_path.stem}_features.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved: {out_file}")
    return str(out_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python 02_features.py <audio_file> <output_dir>")
        sys.exit(1)

    extract_features(sys.argv[1], sys.argv[2])
