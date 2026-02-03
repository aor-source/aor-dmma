#!/usr/bin/env python3
"""
AoR v7.0 - SOVEREIGN EDITION (The Unkillable Build)
===================================================
Target: Apple Silicon (M5) | Features: 300+ | Status: HARDENED

INTEGRATED MODULES:
1. THE CULTURAL CORE: AAVE Corpus Loader + Grammar Detection.
2. THE REINMAN LOGIC: SDS (Irony), TVT (Topology), Volatility.
3. THE M5 PIPELINE: Single-Load Whisper, Multi-Core Math.
4. THE VISUALIZER: Roughness/Intensity + TVT UMAP Trajectories.

Usage:
    python3 aor_mir.py /folder --batch --lexicon /path/to/corpus.json --visualize
"""

import os
import sys
import json
import time
import warnings
import re
import numpy as np
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# --- LIBRARIES ---
try: import torch; HAS_TORCH = True
except ImportError: HAS_TORCH = False
try: import librosa; HAS_LIBROSA = True
except ImportError: HAS_LIBROSA = False
try: import whisper; HAS_WHISPER = True
except ImportError: HAS_WHISPER = False
try: import essentia.standard as es; HAS_ESSENTIA = True
except ImportError: HAS_ESSENTIA = False
try: from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; HAS_VADER = True
except ImportError: HAS_VADER = False
try: import gudhi; HAS_GUDHI = True
except ImportError: HAS_GUDHI = False
try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; HAS_MATPLOTLIB = True
except ImportError: HAS_MATPLOTLIB = False
try: import umap; HAS_UMAP = True
except ImportError: HAS_UMAP = False

warnings.filterwarnings('ignore')

# --- HARDENED AAVE FALLBACKS ---
# (In case the external file "disappears")
DEFAULT_AAVE = {
    "ain't", "gon'", "gonna", "gotta", "finna", "boutta", "tryna", "cuz",
    "y'all", "imma", "lemme", "whatchu", "ion", "iono", "hella", "mad",
    "deadass", "lowkey", "highkey", "lit", "turnt", "hype", "bet", "cap",
    "no cap", "on god", "slime", "thot", "simp", "drip", "guap"
}

AAVE_GRAMMAR_PATTERNS = [
    (r"\bhe\s+\w+s\b", "3rd_person_s_absence"),
    (r"\bshe\s+\w+s\b", "3rd_person_s_absence"),
    (r"\bi\s+be\s+\w+ing\b", "habitual_be"),
    (r"\bthey\s+be\s+\w+ing\b", "habitual_be"),
    (r"\bi\s+been\s+\w+", "been_perfect"),
    (r"\bdone\s+\w+ed\b", "done_perfective"),
    (r"\bain't\s+no\b", "negative_concord"),
    (r"\bdon't\s+got\s+no\b", "negative_concord"),
]

# --- REINMAN METRICS (THE "MISSING" LOGIC) ---

def calculate_sds(lyric_valence: float, audio_arousal: float) -> float:
    """
    Semantic Dissonance Score (SDS) v3
    Measures 'Irony': The distance between what is said (Lyrics) and how it sounds (Audio).
    Range: 0.0 (aligned) to 1.0 (ironic/dissonant).
    """
    # Normalize inputs to 0-1
    v_norm = (lyric_valence + 1) / 2  # VADER is -1 to 1 -> 0 to 1
    a_norm = min(audio_arousal, 1.0)  # RMS is usually 0-1

    # The Reinman Formula: Divergence
    return round(abs(v_norm - a_norm), 4)

def calculate_tvt_score(tonnetz, mfcc_std):
    """
    Topological Valence Trajectory (TVT) Proxy
    Since we can't plot UMAP in a headless script easily, we calculate the
    'Trajectory Complexity' score.
    """
    if tonnetz is None: return 0.0
    # Variance of the harmonic path
    harmonic_flux = np.std(tonnetz)
    # Texture variance
    texture_flux = np.mean(mfcc_std)
    return round(harmonic_flux * texture_flux * 100, 4)

def load_lexicon(path):
    if not path or not os.path.exists(path):
        print("⚠️ No external lexicon found. Using HARDENED FALLBACK.")
        return None
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return None

# --- VISUALIZATION FUNCTIONS ---

def generate_roughness_plot(spectral_centroid, sr, hop_length, output_path, title="Roughness / Intensity Profile"):
    """
    Generate Roughness/Intensity Profile visualization.
    Plots normalized spectral centroid over time.
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available - skipping visualization")
        return None

    # Convert frames to time
    times = librosa.frames_to_time(np.arange(len(spectral_centroid)), sr=sr, hop_length=hop_length)

    # Normalize to 0-1 range
    sc_norm = (spectral_centroid - spectral_centroid.min()) / (spectral_centroid.max() - spectral_centroid.min() + 1e-10)

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='black')
    ax.set_facecolor('black')

    # Plot
    ax.plot(times, sc_norm, color='#00ff00', linewidth=0.8)

    # Styling
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Normalized Spectral Centroid', color='white')
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='black', edgecolor='none')
    plt.close()

    return output_path

def generate_tvt_umap_plot(features, output_path, title="Topological Valence Trajectory"):
    """
    Generate TVT UMAP trajectory visualization.
    Projects high-dimensional audio features to 2D using UMAP.
    """
    if not HAS_MATPLOTLIB or not HAS_UMAP:
        print("⚠️ matplotlib/umap not available - skipping TVT visualization")
        return None

    # Transpose features to (n_samples, n_features)
    if features.shape[0] < features.shape[1]:
        features = features.T

    # Need at least 15 samples for UMAP
    if features.shape[0] < 15:
        print("⚠️ Not enough samples for UMAP - skipping TVT visualization")
        return None

    # Run UMAP
    reducer = umap.UMAP(n_neighbors=min(15, features.shape[0]-1), min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    # Time progression for coloring
    time_progression = np.arange(len(embedding))

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='black')
    ax.set_facecolor('black')

    # Scatter plot with time-based coloring
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                         c=time_progression, cmap='plasma',
                         s=3, alpha=0.8)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Progression', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Styling
    ax.set_xlabel('UMAP Dim 1', color='white')
    ax.set_ylabel('UMAP Dim 2', color='white')
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='black', edgecolor='none')
    plt.close()

    return output_path

def generate_feature_correlation_heatmap(tonnetz, chroma, output_path, title="Feature Correlation"):
    """
    Generate Feature Correlation Heatmap.
    Shows correlation between Tonnetz and Chroma features.
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available - skipping heatmap")
        return None

    # Combine features
    feature_names = [f"Tonnetz_{i}" for i in range(tonnetz.shape[0])] + \
                    [f"Chroma_{i}" for i in range(chroma.shape[0])]
    combined = np.vstack([tonnetz, chroma])

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(combined)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=10)

    # Labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
    plt.close()

    return output_path

def generate_visualizations(y, sr, output_dir, filename_base):
    """
    Generate all visualizations for a track.
    Returns dict of output paths.
    """
    viz_paths = {}
    hop_length = 512

    # 1. Roughness/Intensity Profile (Spectral Centroid)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    roughness_path = output_dir / f"{filename_base}_roughness.png"
    result = generate_roughness_plot(
        spectral_centroid, sr, hop_length,
        str(roughness_path),
        title=f"Roughness / Intensity Profile: {filename_base}"
    )
    if result:
        viz_paths['roughness_plot'] = str(roughness_path)
        print(f"📊 Generated: {roughness_path.name}")

    # 2. TVT UMAP Trajectory (using MFCCs + Chroma)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # Combine features for UMAP
    combined_features = np.vstack([mfcc, chroma]).T  # Shape: (n_frames, n_features)

    tvt_path = output_dir / f"{filename_base}_tvt_umap.png"
    result = generate_tvt_umap_plot(
        combined_features,
        str(tvt_path),
        title=f"Topological Valence Trajectory: {filename_base}"
    )
    if result:
        viz_paths['tvt_umap_plot'] = str(tvt_path)
        print(f"📊 Generated: {tvt_path.name}")

    # 3. Feature Correlation Heatmap (Tonnetz + Chroma)
    heatmap_path = output_dir / f"{filename_base}_correlation.png"
    result = generate_feature_correlation_heatmap(
        tonnetz, chroma,
        str(heatmap_path),
        title=f"Feature Correlation: {filename_base}"
    )
    if result:
        viz_paths['correlation_heatmap'] = str(heatmap_path)
        print(f"📊 Generated: {heatmap_path.name}")

    return viz_paths

# --- WORKER FUNCTION ---

def dsp_worker_task(payload: dict) -> dict:
    try:
        path = Path(payload['path'])
        words = payload['words']
        segments = payload['segments']
        lexicon = payload.get('lexicon')
        visualize = payload.get('visualize', False)
        output_dir = Path(payload.get('output_dir', path.parent))

        # 1. AUDIO PROCESSING
        y, sr = librosa.load(str(path), sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        audio_arousal = float(np.mean(rms)) * 5 # Scale roughly to 0-1

        # 2. FEATURE EXTRACTION
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        # 3. AAVE ANALYSIS (Hardcoded + External)
        full_text = " ".join(words).lower()

        # Build set from lexicon
        if lexicon:
            targets = set()
            if 'contractions' in lexicon:
                targets.update(t.lower() for t in lexicon['contractions'])
            if 'slang_terms' in lexicon:
                for category in lexicon['slang_terms'].values():
                    if isinstance(category, list):
                        targets.update(t.lower() for t in category)
            if 'pronouns_and_articles' in lexicon:
                targets.update(t.lower() for t in lexicon['pronouns_and_articles'])
        else:
            targets = DEFAULT_AAVE

        found_terms = [w for w in words if w.lower() in targets]
        grammar_hits = sum(len(re.findall(p, full_text)) for p, n in AAVE_GRAMMAR_PATTERNS)

        aave_density = (len(found_terms) + grammar_hits) / max(len(words), 1)

        # 4. REINMAN METRICS CALCULATION
        # Lyric Valence (Sentiment)
        lyric_valence = 0.0
        if HAS_VADER:
            analyzer = SentimentIntensityAnalyzer()
            lyric_valence = analyzer.polarity_scores(full_text)['compound']

        # SDS (Irony)
        sds_score = calculate_sds(lyric_valence, audio_arousal)

        # TVT (Topology)
        mfcc_std = np.std(mfcc, axis=1)
        tvt_score = calculate_tvt_score(tonnetz, mfcc_std)

        # Spectral Rigidity
        rigidity = 0.0
        if chroma is not None:
            eigs = np.linalg.eigvalsh(np.corrcoef(chroma))
            spacings = np.diff(np.sort(eigs))
            rigidity = float(np.std(spacings) / (np.mean(spacings) + 1e-10))

        # 5. VISUALIZATIONS (if requested)
        viz_paths = {}
        if visualize:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            viz_paths = generate_visualizations(y, sr, viz_dir, path.stem)

        # 6. ASSEMBLE
        result = {
            "metadata": {"file": path.name, "duration": round(duration, 2), "version": "7.0-sovereign"},
            "cultural_metrics": {
                "aave_density": round(aave_density, 4),
                "unique_terms": len(set(found_terms)),
                "grammar_patterns": grammar_hits,
                "found_terms": list(set(found_terms))[:20],  # Top 20 for reference
                "source": "EXTERNAL" if lexicon else "FALLBACK"
            },
            "reinman_metrics": {
                "sds_score": sds_score,
                "tvt_score": tvt_score,
                "spectral_rigidity": round(rigidity, 4),
                "lyric_valence": round(lyric_valence, 4),
                "audio_arousal": round(audio_arousal, 4)
            },
            "transcript": segments
        }

        if viz_paths:
            result["visualizations"] = viz_paths

        return result

    except Exception as e:
        return {"error": str(e), "file": str(payload.get('path'))}

# --- ORCHESTRATOR ---

class SovereignOrchestrator:
    def __init__(self, model_size="medium", force_cpu=False):
        self.model_size = model_size
        self.model = None
        # MPS can cause NaN issues with Whisper - allow CPU fallback
        if force_cpu:
            self.device = "cpu"
        else:
            self.device = "mps" if HAS_TORCH and torch.backends.mps.is_available() else "cpu"

    def run(self, input_dir, lexicon_path, visualize=False, output_dir=None):
        files = list(Path(input_dir).glob("*.mp3")) + list(Path(input_dir).glob("*.wav"))
        files += list(Path(input_dir).glob("*.flac")) + list(Path(input_dir).glob("*.m4a"))
        lexicon = load_lexicon(lexicon_path)
        output_path = Path(output_dir) if output_dir else Path(input_dir)

        # Ensure output directories exist
        output_path.mkdir(parents=True, exist_ok=True)
        if visualize:
            (output_path / "visualizations").mkdir(parents=True, exist_ok=True)

        # Print dependency status
        print("=" * 60)
        print("AoR v7.0 SOVEREIGN - DEPENDENCY STATUS")
        print("=" * 60)
        print(f"  PyTorch:    {'✅' if HAS_TORCH else '❌'} {'(MPS)' if HAS_TORCH and self.device == 'mps' else ''}")
        print(f"  Whisper:    {'✅' if HAS_WHISPER else '❌'}")
        print(f"  Librosa:    {'✅' if HAS_LIBROSA else '❌'}")
        print(f"  VADER:      {'✅' if HAS_VADER else '❌'}")
        print(f"  Matplotlib: {'✅' if HAS_MATPLOTLIB else '❌'}")
        print(f"  UMAP:       {'✅' if HAS_UMAP else '❌'}")
        print(f"  Essentia:   {'✅' if HAS_ESSENTIA else '❌'}")
        print(f"  GUDHI:      {'✅' if HAS_GUDHI else '❌'}")
        print("=" * 60)

        # LOAD WHISPER
        if HAS_WHISPER and HAS_TORCH:
            print(f"🛡️  Loading Whisper ({self.model_size}) on {self.device.upper()}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
        else:
            print("⚠️  Whisper unavailable - transcription disabled")
            return

        print(f"🚀 AoR v7.0 SOVEREIGN RUNNING. {len(files)} files queued.")
        if visualize:
            print(f"📊 Visualization mode ENABLED")

        with ProcessPoolExecutor(max_workers=max(1, mp.cpu_count()-2)) as executor:
            futures = {}
            for f in files:
                print(f"🎤 Transcribing: {f.name}")
                try:
                    # GPU transcription
                    res = self.model.transcribe(str(f), word_timestamps=True)
                    words = []
                    for seg in res.get('segments', []):
                        for w in seg.get('words', []):
                            if 'word' in w:
                                words.append(w['word'].strip())

                    # CPU Handoff for DSP
                    payload = {
                        'path': str(f),
                        'words': words,
                        'segments': res.get('segments', []),
                        'lexicon': lexicon,
                        'visualize': visualize,
                        'output_dir': str(output_path)
                    }
                    futures[executor.submit(dsp_worker_task, payload)] = f
                except Exception as e:
                    print(f"❌ Error transcribing {f.name}: {e}")

            for fut in as_completed(futures):
                f = futures[fut]
                data = fut.result()

                if 'error' in data:
                    print(f"❌ Error processing {f.name}: {data['error']}")
                    continue

                # Check for "Hacker" interference (Empty results)
                if data.get('cultural_metrics', {}).get('aave_density', 0) == 0:
                    print(f"⚠️  WARNING: Zero AAVE detected in {f.name}. Check Lexicon.")

                out = output_path / f"{f.stem}_sovereign.json"
                with open(out, 'w') as jf: json.dump(data, jf, indent=2)
                print(f"✅ Secure: {f.name}")

        print(f"\n🏁 Complete. Output in: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AoR v7.0 SOVEREIGN - Audio analysis with AAVE detection and Reinman metrics"
    )
    parser.add_argument("audio", help="Audio file or directory")
    parser.add_argument("--lexicon", help="Path to AAVE lexicon JSON")
    parser.add_argument("--batch", action="store_true", help="Process directory of files")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations (Roughness, TVT UMAP)")
    parser.add_argument("--output", help="Output directory (default: same as input)")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (fixes MPS NaN issues)")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    orch = SovereignOrchestrator(model_size=args.model, force_cpu=args.cpu)
    path = args.audio if args.batch else os.path.dirname(args.audio) if os.path.isfile(args.audio) else args.audio
    orch.run(path, args.lexicon, visualize=args.visualize, output_dir=args.output)
