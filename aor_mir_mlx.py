#!/usr/bin/env python3
"""
AoR v7.0 MLX - Apple Silicon Optimized
Uses mlx-whisper for 5-10x faster transcription on M-series chips
"""

import json
import os
import sys
import argparse
from pathlib import Path

# MLX Whisper for fast transcription
import mlx_whisper

# Import analysis functions from main script
import numpy as np

# Audio processing
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def load_lexicon(lexicon_path):
    """Load AAVE lexicon"""
    with open(lexicon_path, 'r') as f:
        return json.load(f)


def analyze_aave(text, lexicon):
    """Detect AAVE terms in text - handles nested lexicon structure"""
    import re
    text_lower = text.lower()
    words = text_lower.split()

    # Clean words for matching
    clean_words = [re.sub(r'[^\w\']', '', w) for w in words]

    found_terms = []
    grammar_patterns = 0

    # Build flat list of AAVE terms to search for
    aave_terms = set()

    # Flat list categories
    for cat in ['contractions', 'pronouns_and_articles']:
        if cat in lexicon and isinstance(lexicon[cat], list):
            aave_terms.update(lexicon[cat])

    # Semantic inversions - get the terms themselves
    if 'semantic_inversions' in lexicon:
        for term, info in lexicon['semantic_inversions'].items():
            aave_terms.add(term)

    # Slang terms - nested dict with lists
    if 'slang_terms' in lexicon:
        for category, terms in lexicon['slang_terms'].items():
            if isinstance(terms, list):
                aave_terms.update(terms)

    # Classic hip-hop terms by era
    if 'classic_hip_hop_terms' in lexicon:
        for era, terms in lexicon['classic_hip_hop_terms'].items():
            if isinstance(terms, list):
                aave_terms.update(terms)

    # Regional variants - get AAVE forms
    if 'regional_variants' in lexicon:
        for region, variants in lexicon['regional_variants'].items():
            if isinstance(variants, list):
                aave_terms.update(variants)

    # Context markers (all types: positive, negative, affirmations, exclamations)
    if 'context_markers' in lexicon:
        for marker_type, markers in lexicon['context_markers'].items():
            if isinstance(markers, list):
                aave_terms.update(markers)

    # Contractions
    if 'contractions' in lexicon and isinstance(lexicon['contractions'], list):
        aave_terms.update(lexicon['contractions'])

    # Phonological - mappings are {aave: standard}
    # We detect AAVE forms directly, AND standard forms that Whisper may have normalized
    standard_to_aave = {}  # Track standard forms that indicate AAVE usage
    if 'phonological_patterns' in lexicon:
        for pattern, info in lexicon['phonological_patterns'].items():
            if isinstance(info, dict) and 'mappings' in info:
                aave_terms.update(info['mappings'].keys())  # AAVE forms (keys)
                # Map standard -> aave for Whisper normalization detection
                for aave_form, std_form in info['mappings'].items():
                    standard_to_aave[std_form.lower()] = aave_form

    # Common false positives to exclude
    false_positives = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'this', 'that', 'these', 'those', 'it', 'its', 'word'}
    aave_terms -= false_positives

    # Search for AAVE terms
    for term in aave_terms:
        term_lower = term.lower().strip()
        if not term_lower or len(term_lower) < 2:
            continue

        # Check as whole word
        if term_lower in clean_words:
            found_terms.append(term)
        # Also check with apostrophes
        elif "'" in term_lower and term_lower in text_lower:
            found_terms.append(term)

    # Also check standard English forms that Whisper may have normalized
    # (e.g., "this" in hip-hop likely spoken as "dis")
    for std_form, aave_form in standard_to_aave.items():
        if std_form in clean_words and aave_form not in found_terms:
            found_terms.append(f"{aave_form}")

    # Check grammar patterns
    if 'grammatical_patterns' in lexicon:
        for pattern, info in lexicon['grammatical_patterns'].items():
            if isinstance(info, dict) and 'examples' in info:
                for ex in info['examples']:
                    if ex.lower() in text_lower:
                        grammar_patterns += 1

    unique_terms = list(set(found_terms))
    density = len(unique_terms) / max(len(words), 1)

    return {
        'density': density,
        'unique_terms': len(unique_terms),
        'found_terms': unique_terms,
        'grammar_patterns': grammar_patterns,
        'source': 'enhanced_v3'
    }


def compute_reinman_metrics(y, sr, sentiment_score):
    """Compute Reinman topology metrics"""
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # RMS for audio arousal
    rms = librosa.feature.rms(y=y)[0]
    audio_arousal = float(np.mean(rms))

    # Normalize arousal to 0-1
    audio_arousal = min(1.0, audio_arousal * 10)

    # SDS: Semantic Dissonance Score (irony detection)
    # High when lyric sentiment and audio energy diverge
    sds = abs(sentiment_score - (audio_arousal - 0.5) * 2)

    # TVT: Topological Valence Trajectory
    # Measures complexity of spectral trajectory
    centroid_diff = np.diff(spectral_centroid)
    tvt = float(np.std(centroid_diff))

    # Spectral Rigidity
    spectral_rigidity = float(np.mean(spectral_rolloff) / sr)

    return {
        'sds_score': round(sds, 4),
        'tvt_score': round(tvt, 2),
        'spectral_rigidity': round(spectral_rigidity, 4),
        'lyric_valence': round(sentiment_score, 4),
        'audio_arousal': round(audio_arousal, 4)
    }


def generate_visualizations(y, sr, output_dir, filename_base):
    """Generate visualization PNGs"""
    if not HAS_MATPLOTLIB:
        return

    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Roughness/Spectral plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    hop_length = 512
    times = librosa.times_like(spectral_centroid, sr=sr, hop_length=hop_length)

    ax.plot(times, spectral_centroid, color='#00d4ff', linewidth=0.5)
    ax.fill_between(times, spectral_centroid, alpha=0.3, color='#00d4ff')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spectral Centroid (Hz)')
    ax.set_title(f'Spectral Roughness: {filename_base}')

    plt.tight_layout()
    plt.savefig(viz_dir / f"{filename_base}_roughness.png", dpi=150, facecolor='black')
    plt.close()
    print(f"📊 Generated: {filename_base}_roughness.png")

    # 2. TVT UMAP (if available)
    if HAS_UMAP:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.vstack([chroma, mfcc]).T

            if len(features) > 15:
                reducer = umap.UMAP(n_neighbors=min(15, len(features)-1), min_dist=0.1, random_state=42)
                embedding = reducer.fit_transform(features)

                fig, ax = plt.subplots(figsize=(8, 8))
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                   c=np.arange(len(embedding)), cmap='plasma',
                                   s=10, alpha=0.7)
                ax.set_title(f'TVT UMAP: {filename_base}')
                plt.colorbar(scatter, label='Time progression')
                plt.tight_layout()
                plt.savefig(viz_dir / f"{filename_base}_tvt_umap.png", dpi=150, facecolor='black')
                plt.close()
                print(f"📊 Generated: {filename_base}_tvt_umap.png")
        except Exception as e:
            print(f"⚠️ UMAP viz failed: {e}")

    # 3. Feature correlation heatmap
    try:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        combined = np.corrcoef(np.vstack([tonnetz, chroma]))

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(combined, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'Feature Correlation: {filename_base}')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(viz_dir / f"{filename_base}_correlation.png", dpi=150, facecolor='black')
        plt.close()
        print(f"📊 Generated: {filename_base}_correlation.png")
    except Exception as e:
        print(f"⚠️ Correlation viz failed: {e}")


def process_track(audio_path, lexicon, output_dir, visualize=False):
    """Process a single track with mlx-whisper"""
    audio_path = Path(audio_path)
    filename = audio_path.stem

    print(f"\n{'='*60}")
    print(f"🎵 {filename}")
    print('='*60)

    # Transcribe with mlx-whisper (Apple Silicon optimized)
    print("🎤 Transcribing with mlx-whisper (Apple Silicon)...")
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo="mlx-community/whisper-medium-mlx")
    transcript = result.get('text', '')

    print(f"📝 Transcript: {len(transcript)} chars")

    # Load audio for analysis
    print("🔊 Analyzing audio features...")
    y, sr = librosa.load(str(audio_path), sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # AAVE analysis
    print("🗣️ AAVE detection...")
    aave_results = analyze_aave(transcript, lexicon)

    # Sentiment analysis
    sentiment_score = 0.0
    if HAS_VADER:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(transcript)
        sentiment_score = sentiment['compound']

    # Reinman metrics
    print("📐 Computing Reinman metrics...")
    reinman = compute_reinman_metrics(y, sr, sentiment_score)

    # Visualizations
    if visualize:
        print("🎨 Generating visualizations...")
        generate_visualizations(y, sr, output_dir, filename)

    # Build output
    output = {
        'metadata': {
            'file': str(audio_path),
            'duration': round(duration, 2),
            'analyzer': 'AoR v7.0 MLX'
        },
        'transcript': transcript[:500] + '...' if len(transcript) > 500 else transcript,
        'cultural_metrics': {
            'aave_density': round(aave_results['density'], 4),
            'unique_terms': aave_results['unique_terms'],
            'grammar_patterns': aave_results['grammar_patterns'],
            'found_terms': aave_results['found_terms'][:20],
            'source': aave_results['source']
        },
        'reinman_metrics': reinman
    }

    # Save JSON
    output_path = Path(output_dir) / f"{filename}_sovereign.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved: {output_path.name}")
    print(f"   AAVE Density: {aave_results['density']*100:.2f}%")
    print(f"   SDS (Irony): {reinman['sds_score']}")
    print(f"   TVT (Complexity): {reinman['tvt_score']}")

    return output


def main():
    parser = argparse.ArgumentParser(description='AoR v7.0 MLX - Apple Silicon Optimized')
    parser.add_argument('tracks_file', help='File with list of track paths')
    parser.add_argument('--lexicon', default='aave_lexicon.json', help='AAVE lexicon path')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    print("="*60)
    print("AoR v7.0 MLX - APPLE SILICON OPTIMIZED")
    print("="*60)
    print(f"  mlx-whisper: ✅")
    print(f"  Librosa:     {'✅' if HAS_LIBROSA else '❌'}")
    print(f"  VADER:       {'✅' if HAS_VADER else '❌'}")
    print(f"  Matplotlib:  {'✅' if HAS_MATPLOTLIB else '❌'}")
    print(f"  UMAP:        {'✅' if HAS_UMAP else '❌'}")
    print("="*60)

    # Load lexicon
    lexicon = load_lexicon(args.lexicon)
    print(f"📚 Loaded lexicon: {args.lexicon}")

    # Load track list
    with open(args.tracks_file, 'r') as f:
        tracks = [line.strip() for line in f if line.strip()]

    print(f"🎵 Processing {len(tracks)} tracks\n")

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Process each track
    results = []
    for i, track in enumerate(tracks, 1):
        print(f"\n[{i}/{len(tracks)}] ", end='')
        try:
            result = process_track(track, lexicon, args.output, args.visualize)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Processed: {len(results)}/{len(tracks)} tracks")
    print(f"Output: {args.output}")

    if results:
        avg_aave = np.mean([r['cultural_metrics']['aave_density'] for r in results])
        avg_sds = np.mean([r['reinman_metrics']['sds_score'] for r in results])
        print(f"Avg AAVE Density: {avg_aave*100:.2f}%")
        print(f"Avg SDS (Irony): {avg_sds:.4f}")


if __name__ == '__main__':
    main()
