#!/usr/bin/env python3
"""
AoR v7.1 MLX - Apple Silicon Optimized, Full Feature Extraction
================================================================
Uses mlx-whisper for 5-10x faster transcription on M-series chips.
300+ features per track with modular output schema.

Feature Groups:
- metadata: file info, duration, provenance, hash
- spectral: chroma, mfcc, spectral features, rigidity
- rhythmic: tempo, beats, onset strength
- harmonic: tonnetz, harmonic ratios, pitch
- energy: rms, dynamics, loudness percentiles
- transcript: mlx-whisper output
- linguistic: word stats, vocabulary, syllables
- aave: dialect features with per-category breakdown
- reinman_metrics: SDS (irony), TVT (complexity), spectral rigidity
- god_equation: composite scoring (with/without AAVE correction)

Usage:
    python aor_mir_mlx.py tracks.txt --lexicon aave_lexicon.json --output results/ --visualize
"""

import json
import os
import sys
import re
import time
import hashlib
import statistics
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# MLX Whisper for fast transcription
import mlx_whisper

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

# Rhyme analysis
try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False

# ==============================================================================
#  AAVE BUILT-IN FALLBACK LEXICON
# ==============================================================================

AAVE_LEXICON_BUILTIN = {
    "contractions": {
        "ain't", "gon'", "gonna", "gotta", "wanna", "tryna", "finna", "boutta",
        "bout", "cuz", "cause", "y'all", "imma", "lemme", "gimme", "kinda",
        "sorta", "dunno", "wassup", "whatchu", "aint", "yall", "ima", "ion",
        "iono", "aight", "ight", "nah", "yea", "yeah", "nope", "prolly",
        "shoulda", "woulda", "coulda", "mighta", "musta", "oughta", "supposta",
        "useta", "hafta", "needa"
    },
    "intensifiers": {
        "hella", "mad", "crazy", "straight", "real", "true", "dead", "deadass",
        "lowkey", "highkey", "super", "extra", "heavy", "deep", "hard", "raw",
        "wild", "sick", "dope", "fire", "lit", "turnt", "hype", "tight",
        "valid", "bussin", "slaps", "hits", "goated", "based", "cap", "nocap",
        "facts", "bet", "word", "legit", "forreal", "frfr", "ong", "onggod",
        "fasho", "fosho", "no doubt", "big", "major", "massive"
    },
    "markers": {
        "bruh", "bro", "fam", "cuh", "cuz", "dawg", "dog", "homie", "homeboy",
        "shorty", "shawty", "playa", "player", "pimp", "g", "og", "blood",
        "loc", "foo", "fool", "ese", "mane", "mayne", "son", "kid", "yo", "aye",
        "ay", "man", "dude", "folks", "folk", "gang", "gangsta", "thug",
        "brotha", "sista", "queen", "king", "young", "youngin", "lil"
    },
    "verbs": {
        "vibin", "chillin", "coolin", "posted", "postin", "flexin", "stuntin",
        "trappin", "hustlin", "grindin", "stackin", "ballin", "rollin", "mobbin",
        "creepin", "slippin", "trippin", "buggin", "wildin", "wylin", "spazzin",
        "cappin", "lackin", "packin", "strapped", "loaded", "faded", "bent",
        "blunted", "zooted", "twisted", "turnt", "woke", "stay", "been", "finna"
    },
    "expressions": {
        "what's good", "what's up", "what's poppin", "what it do", "what it is",
        "you feel me", "feel me", "know what i'm saying", "know what i mean",
        "you dig", "dig it", "on god", "on my mama", "on everything", "no cap",
        "for real", "real talk", "straight up", "on the real", "keep it real",
        "keep it 100", "hundred", "facts", "period", "periodt", "that's crazy",
        "that's wild", "that's fire", "that's hard", "goes hard", "hits different"
    }
}

AAVE_GRAMMAR_PATTERNS = [
    (r"\bi\s+be\s+\w+ing\b", "habitual_be", 2.0),
    (r"\bhe\s+be\s+\w+ing\b", "habitual_be", 2.0),
    (r"\bshe\s+be\s+\w+ing\b", "habitual_be", 2.0),
    (r"\bthey\s+be\s+\w+ing\b", "habitual_be", 2.0),
    (r"\bwe\s+be\s+\w+ing\b", "habitual_be", 2.0),
    (r"\bi\s+been\s+\w+", "remote_past_been", 2.5),
    (r"\bbeen\s+\w+ing\b", "stressed_been", 2.0),
    (r"\bdone\s+\w+ed\b", "completive_done", 2.0),
    (r"\bain't\s+no\b", "negative_concord", 1.5),
    (r"\bdon't\s+got\s+no\b", "negative_concord", 1.5),
    (r"\bcan't\s+no\b", "negative_concord", 1.5),
    (r"\bwon't\s+no\b", "negative_concord", 1.5),
    (r"\bdidn't\s+\w+\s+no\b", "multiple_negation", 2.0),
    (r"\bit\s+ain't\b", "ain't_negation", 1.0),
    (r"\bthere\s+go\b", "existential_it", 1.5),
    (r"\bit\s+go\b", "existential_it", 1.5),
    (r"\bwhere\s+\w+\s+at\b", "locative_at", 1.0),
    (r"\bwhat\s+\w+\s+like\b", "question_inversion", 1.0),
    (r"\bstay\s+\w+ing\b", "aspectual_stay", 2.0),
    (r"\bcome\s+\w+ing\b", "camouflage_come", 1.5),
]


# ==============================================================================
#  FEATURE EXTRACTION MODULES
# ==============================================================================

def extract_metadata(filepath: Path, y, sr: int) -> Dict:
    """Basic file and audio metadata"""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        "filename": filepath.name,
        "filepath": str(filepath),
        "file_hash": file_hash,
        "file_size_bytes": filepath.stat().st_size,
        "duration_seconds": round(len(y) / sr, 2),
        "sample_rate": sr,
        "samples": len(y),
        "channels": 1,
        "analysis_timestamp": datetime.now().isoformat(),
        "aor_version": "7.1-MLX-FULL"
    }


def extract_spectral_features(y, sr: int) -> Dict:
    """Comprehensive spectral analysis"""
    features = {}

    try:
        # Chroma (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma_mean"] = [round(float(x), 4) for x in chroma.mean(axis=1)]
        features["chroma_std"] = [round(float(x), 4) for x in chroma.std(axis=1)]

        # MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = [round(float(x), 4) for x in mfcc.mean(axis=1)]
        features["mfcc_std"] = [round(float(x), 4) for x in mfcc.std(axis=1)]
        features["mfcc_delta_mean"] = [round(float(x), 4) for x in librosa.feature.delta(mfcc).mean(axis=1)]

        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = round(float(cent.mean()), 2)
        features["spectral_centroid_std"] = round(float(cent.std()), 2)

        # Spectral bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = round(float(bw.mean()), 2)
        features["spectral_bandwidth_std"] = round(float(bw.std()), 2)

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff_mean"] = round(float(rolloff.mean()), 2)

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = round(float(flatness.mean()), 6)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["spectral_contrast_mean"] = [round(float(x), 4) for x in contrast.mean(axis=1)]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zero_crossing_rate_mean"] = round(float(zcr.mean()), 6)

        # Spectral rigidity (eigenvalue analysis of correlation matrix)
        try:
            corr = np.corrcoef(chroma)
            eigs = np.linalg.eigvalsh(corr)
            eigs_sorted = np.sort(eigs)
            spacings = np.diff(eigs_sorted)
            features["spectral_rigidity"] = round(float(np.std(spacings) / (np.mean(spacings) + 1e-10)), 4)
            features["eigenvalue_spread"] = round(float(eigs.max() - eigs.min()), 4)
        except:
            features["spectral_rigidity"] = 0.0
            features["eigenvalue_spread"] = 0.0

    except Exception as e:
        features["error"] = str(e)

    return features


def extract_rhythmic_features(y, sr: int) -> Dict:
    """Rhythm and tempo analysis"""
    features = {}

    try:
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features["onset_strength_mean"] = round(float(onset_env.mean()), 4)
        features["onset_strength_std"] = round(float(onset_env.std()), 4)

        # Tempo
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        features["tempo_bpm"] = round(float(tempo[0]), 1) if len(tempo) > 0 else 0

        # Beat tracking
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features["beat_count"] = len(beats)

        if len(beats) > 1:
            beat_times = librosa.frames_to_time(beats, sr=sr)
            intervals = np.diff(beat_times)
            features["beat_interval_mean"] = round(float(intervals.mean()), 4)
            features["beat_interval_std"] = round(float(intervals.std()), 4)
            features["beat_regularity"] = round(1.0 - min(float(intervals.std() / (intervals.mean() + 1e-10)), 1.0), 4)

        # Tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        features["tempogram_ratio"] = round(float(tempogram.max() / (tempogram.mean() + 1e-10)), 4)

    except Exception as e:
        features["error"] = str(e)

    return features


def extract_harmonic_features(y, sr: int) -> Dict:
    """Harmonic and tonal analysis"""
    features = {}

    try:
        # Harmonic ratio via spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features["harmonic_ratio"] = round(float(1.0 - flatness.mean()), 4)

        # Tonnetz - use small segment to avoid memory issues
        segment_len = min(len(y), sr * 60)
        y_segment = y[:segment_len]

        try:
            chroma = librosa.feature.chroma_cqt(y=y_segment, sr=sr)
            tonnetz = librosa.feature.tonnetz(chroma=chroma)
            features["tonnetz_mean"] = [round(float(x), 4) for x in tonnetz.mean(axis=1)]
            features["tonnetz_std"] = [round(float(x), 4) for x in tonnetz.std(axis=1)]
        except:
            features["tonnetz_mean"] = [0.0] * 6
            features["tonnetz_std"] = [0.0] * 6

        # Pitch estimation
        pitches, magnitudes = librosa.piptrack(y=y_segment, sr=sr)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_vals) > 0:
            features["pitch_mean"] = round(float(pitch_vals.mean()), 2)
            features["pitch_std"] = round(float(pitch_vals.std()), 2)
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

    except Exception as e:
        features["error"] = str(e)

    return features


def extract_energy_features(y, sr: int) -> Dict:
    """Energy and dynamics analysis"""
    features = {}

    try:
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = round(float(rms.mean()), 6)
        features["rms_std"] = round(float(rms.std()), 6)
        features["rms_max"] = round(float(rms.max()), 6)
        features["rms_min"] = round(float(rms.min()), 6)

        # Dynamic range
        features["dynamic_range_db"] = round(float(20 * np.log10(rms.max() / (rms.min() + 1e-10))), 2)

        # Loudness variation
        features["loudness_variance"] = round(float(np.var(rms)), 8)

        # Energy percentiles
        features["energy_p10"] = round(float(np.percentile(rms, 10)), 6)
        features["energy_p50"] = round(float(np.percentile(rms, 50)), 6)
        features["energy_p90"] = round(float(np.percentile(rms, 90)), 6)

    except Exception as e:
        features["error"] = str(e)

    return features


def extract_linguistic_features(text: str, segments: List[Dict], duration: float = 0.0) -> Dict:
    """Text and linguistic analysis (non-AAVE)"""
    features = {}

    words = re.findall(r'\b\w+\b', text.lower())
    features["word_count"] = len(words)
    features["unique_words"] = len(set(words))
    features["vocabulary_richness"] = round(len(set(words)) / max(len(words), 1), 4)

    features["char_count"] = len(text)
    features["avg_word_length"] = round(sum(len(w) for w in words) / max(len(words), 1), 2)

    features["segment_count"] = len(segments)

    if segments:
        words_per_seg = [len(s.get('text', '').split()) for s in segments]
        features["words_per_segment_mean"] = round(statistics.mean(words_per_seg) if words_per_seg else 0, 2)
        features["words_per_segment_std"] = round(statistics.stdev(words_per_seg) if len(words_per_seg) > 1 else 0, 2)

    # Syllable estimation
    def count_syllables(word):
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)

    syllables = sum(count_syllables(w) for w in words)
    features["syllable_count"] = syllables
    features["syllables_per_word"] = round(syllables / max(len(words), 1), 2)
    features["syllables_per_second"] = round(syllables / max(duration, 0.01), 2) if duration > 0 else 0.0
    features["words_per_second"] = round(len(words) / max(duration, 0.01), 2) if duration > 0 else 0.0

    # Rhyme analysis
    if HAS_PRONOUNCING:
        try:
            rhyme_pairs = 0
            for i, w1 in enumerate(words[:-1]):
                rhymes = set(pronouncing.rhymes(w1))
                if words[i + 1] in rhymes:
                    rhyme_pairs += 1
            features["adjacent_rhymes"] = rhyme_pairs
        except:
            features["adjacent_rhymes"] = 0

    return features


# ==============================================================================
#  AAVE ANALYSIS (Enhanced - supports both JSON lexicon and built-in fallback)
# ==============================================================================

def load_lexicon(lexicon_path):
    """Load AAVE lexicon from JSON file"""
    if not lexicon_path or not os.path.exists(lexicon_path):
        print("⚠️ No external lexicon found. Using HARDENED FALLBACK.")
        return None
    try:
        with open(lexicon_path, 'r') as f:
            return json.load(f)
    except:
        print("⚠️ Failed to load lexicon. Using HARDENED FALLBACK.")
        return None


def analyze_aave(text: str, lexicon: Dict = None) -> Dict:
    """
    AAVE Dialect Analysis with per-category breakdown.
    Supports both the JSON lexicon (807 terms) and built-in fallback.
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    clean_words = [re.sub(r'[^\w\']', '', w) for w in text_lower.split()]
    word_set = set(clean_words)

    features = {
        "lexicon_source": "external_json" if lexicon else "builtin_fallback",
        "categories": {}
    }

    total_matches = 0
    all_matched_terms = []

    if lexicon:
        # --- JSON lexicon mode (structured with nested categories) ---

        # Common false positives to exclude
        false_positives = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                           'this', 'that', 'these', 'those', 'it', 'its', 'word'}

        # Process each category from the JSON lexicon
        category_terms = {}

        # Flat list categories
        for cat in ['contractions', 'pronouns_and_articles']:
            if cat in lexicon and isinstance(lexicon[cat], list):
                category_terms[cat] = set(t.lower() for t in lexicon[cat]) - false_positives

        # Semantic inversions
        if 'semantic_inversions' in lexicon:
            terms = set()
            for term in lexicon['semantic_inversions']:
                terms.add(term.lower())
            category_terms['semantic_inversions'] = terms - false_positives

        # Slang terms (nested dict with lists)
        if 'slang_terms' in lexicon:
            for sub_cat, terms in lexicon['slang_terms'].items():
                if isinstance(terms, list):
                    key = f"slang_{sub_cat}"
                    category_terms[key] = set(t.lower() for t in terms) - false_positives

        # Classic hip-hop terms by era
        if 'classic_hip_hop_terms' in lexicon:
            for era, terms in lexicon['classic_hip_hop_terms'].items():
                if isinstance(terms, list):
                    key = f"classic_{era}"
                    category_terms[key] = set(t.lower() for t in terms) - false_positives

        # Regional variants
        if 'regional_variants' in lexicon:
            for region, variants in lexicon['regional_variants'].items():
                if isinstance(variants, list):
                    key = f"regional_{region}"
                    category_terms[key] = set(t.lower() for t in variants) - false_positives

        # Context markers
        if 'context_markers' in lexicon:
            for marker_type, markers in lexicon['context_markers'].items():
                if isinstance(markers, list):
                    key = f"context_{marker_type}"
                    category_terms[key] = set(t.lower() for t in markers) - false_positives

        # Phonological patterns (detect both AAVE forms and Whisper-normalized forms)
        standard_to_aave = {}
        if 'phonological_patterns' in lexicon:
            phon_terms = set()
            for pattern, info in lexicon['phonological_patterns'].items():
                if isinstance(info, dict) and 'mappings' in info:
                    for aave_form, std_form in info['mappings'].items():
                        phon_terms.add(aave_form.lower())
                        standard_to_aave[std_form.lower()] = aave_form
            category_terms['phonological'] = phon_terms - false_positives

        # Match each category
        for cat_name, terms in category_terms.items():
            matched = []
            for term in terms:
                term_clean = term.strip()
                if not term_clean or len(term_clean) < 2:
                    continue
                if term_clean in word_set:
                    matched.append(term_clean)
                elif "'" in term_clean and term_clean in text_lower:
                    matched.append(term_clean)

            features["categories"][cat_name] = {
                "matched_count": len(matched),
                "matched_terms": matched[:20],
                "density": round(len(matched) / max(len(words), 1), 6)
            }
            total_matches += len(matched)
            all_matched_terms.extend(matched)

        # Also check standard forms Whisper may have normalized
        whisper_normalized = []
        for std_form, aave_form in standard_to_aave.items():
            if std_form in word_set and aave_form not in all_matched_terms:
                whisper_normalized.append(aave_form)
                all_matched_terms.append(aave_form)
                total_matches += 1

        if whisper_normalized:
            features["categories"]["whisper_normalized"] = {
                "matched_count": len(whisper_normalized),
                "matched_terms": whisper_normalized[:20],
                "density": round(len(whisper_normalized) / max(len(words), 1), 6),
                "note": "AAVE forms likely spoken but normalized by Whisper ASR"
            }

    else:
        # --- Built-in fallback mode (dict of sets) ---
        for category, terms in AAVE_LEXICON_BUILTIN.items():
            matched = [w for w in word_set if w in terms]
            features["categories"][category] = {
                "matched_count": len(matched),
                "matched_terms": matched[:20],
                "density": round(len(matched) / max(len(words), 1), 6)
            }
            total_matches += len(matched)
            all_matched_terms.extend(matched)

    # Grammar patterns (always applied)
    grammar_matches = []
    grammar_score = 0.0
    for pattern, name, weight in AAVE_GRAMMAR_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            grammar_matches.append({
                "pattern": name,
                "count": len(matches),
                "weight": weight,
                "examples": matches[:3]
            })
            grammar_score += len(matches) * weight

    features["grammar"] = {
        "patterns_found": len(grammar_matches),
        "weighted_score": round(grammar_score, 2),
        "matches": grammar_matches
    }

    # Aggregate metrics
    unique_matched = list(set(all_matched_terms))
    features["totals"] = {
        "lexical_matches": total_matches,
        "grammar_matches": sum(g["count"] for g in grammar_matches),
        "combined_matches": total_matches + sum(g["count"] for g in grammar_matches),
        "aave_density": round((total_matches + grammar_score) / max(len(words), 1), 6),
        "unique_terms": len(unique_matched),
        "aave_richness": round(len(unique_matched) / max(total_matches, 1), 4)
    }

    features["all_matched_terms"] = unique_matched[:50]

    return features


# ==============================================================================
#  REINMAN METRICS
# ==============================================================================

def compute_reinman_metrics(y, sr, sentiment_score):
    """Compute Reinman topology metrics"""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    rms = librosa.feature.rms(y=y)[0]
    audio_arousal = float(np.mean(rms))
    audio_arousal = min(1.0, audio_arousal * 10)

    # SDS: Semantic Dissonance Score (irony detection)
    sds = abs(sentiment_score - (audio_arousal - 0.5) * 2)

    # TVT: Topological Valence Trajectory
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


# ==============================================================================
#  GOD EQUATION - Composite Scoring
# ==============================================================================

def calculate_god_equation(spectral: Dict, aave: Dict, rhythmic: Dict,
                           harmonic: Dict, energy: Dict, linguistic: Dict,
                           reinman: Dict, file_hash: str = "unknown") -> Dict:
    """
    SPECTRAL COHERENCE INDEX (SCI) - The God Equation
    =================================================
    Restored to original v7/v8 scale where 6+ is exceptional.

    INTELLIGENCE SCORE (I) = w_R*R + w_T*T + w_C*C + w_S*S + w_V*V
    SCI = I / (1 + D*0.1)   [Decay penalizes repetitive content]

    Components (all normalized to 0-10):
    - R = Spectral Rigidity (eigenvalue variance - structural intelligence)
    - T = Topological Complexity (TVT score - trajectory complexity)
    - C = AAVE Density (cultural intelligence marker)
    - S = Semantic Compression (vocabulary richness + syllabic density)
    - V = Volatility Composite (SDS irony + energy dynamics)
    - D = Decay (repetition penalty from vocab repetition)
    - H = Harmonic Ratio (musical coherence - legacy component)

    Weights (from Reinman artifact analysis):
    - R: 0.25 (structural)
    - T: 0.15 (topological)
    - C: 0.25 (cultural)
    - S: 0.20 (semantic)
    - V: 0.15 (dynamic)
    """

    # === Normalize components to [0, 10] scale ===

    # R: Spectral Rigidity
    R = min(spectral.get("spectral_rigidity", 0), 10.0)

    # T: Topological Complexity (TVT typically 0-1000, normalize)
    tvt_raw = reinman.get("tvt_score", 0)
    T = min(tvt_raw / 100.0, 10.0)

    # C: AAVE Cultural Density (density typically 0-0.15, scale *100 to 0-10)
    aave_density = aave.get("totals", {}).get("aave_density", 0)
    C = min(aave_density * 100, 10.0)

    # S: Semantic Compression (vocab richness 0-1, + syllabic density bonus)
    vocab_richness = linguistic.get("vocabulary_richness", 0)
    syl_per_sec = linguistic.get("syllables_per_second", 0)
    # Richness * 10, plus bonus for fast delivery (capped at 2)
    S = min(vocab_richness * 10 + min(syl_per_sec / 5.0, 2.0), 10.0)

    # V: Volatility Composite (SDS irony + dynamic range)
    sds = reinman.get("sds_score", 0)
    dyn_range = energy.get("dynamic_range_db", 0)
    # SDS is 0-2ish, scale *3; dynamic range normalized from dB (0-100 -> 0-3)
    V = min(sds * 3 + min(dyn_range / 33.0, 3.0), 10.0)

    # D: Decay (repetition penalty - inverse of vocab richness)
    D = min((1.0 - vocab_richness) * 10, 10.0)

    # H: Harmonic Ratio (legacy, for backwards compatibility reporting)
    H = min(harmonic.get("harmonic_ratio", 0.5) * 10, 10.0)

    # === Weights ===
    w_R, w_T, w_C, w_S, w_V = 0.25, 0.15, 0.25, 0.20, 0.15

    # === Intelligence Score (weighted sum) ===
    I = (w_R * R) + (w_T * T) + (w_C * C) + (w_S * S) + (w_V * V)

    # === SCI with AAVE ===
    sci_with_aave = I / (1 + D * 0.1)

    # === SCI without AAVE (biased baseline - cultural component zeroed) ===
    I_no_aave = (w_R * R) + (w_T * T) + (w_C * 0) + (w_S * S) + (w_V * V)
    sci_no_aave = I_no_aave / (1 + D * 0.1)

    # === Variance shows bias magnitude ===
    variance_pct = round(((sci_with_aave - sci_no_aave) / max(sci_no_aave, 0.01)) * 100, 2)

    # Legacy MIR score (v7.0 compatible: R*0.3 + C*0.5 + H*0.2)
    R_legacy = min(spectral.get("spectral_rigidity", 0), 10.0)
    C_legacy = min(aave_density * 10, 10.0)
    H_legacy = min(harmonic.get("harmonic_ratio", 0.5) * 10, 10.0)
    mir_legacy = (R_legacy * 0.3) + (C_legacy * 0.5) + (H_legacy * 0.2)

    tag = f"AOR-SCI-{file_hash[:8]}-{int(sci_with_aave*100):04d}"

    return {
        "sci_with_aave": round(sci_with_aave, 4),
        "sci_without_aave": round(sci_no_aave, 4),
        "variance_percent": variance_pct,
        "intelligence_score": round(I, 4),
        "mir_score_legacy_v7": round(mir_legacy, 4),
        "components": {
            "R_spectral_rigidity": round(R, 4),
            "T_topological_complexity": round(T, 4),
            "C_aave_density": round(C, 4),
            "S_semantic_compression": round(S, 4),
            "V_volatility_composite": round(V, 4),
            "D_decay_penalty": round(D, 4),
            "H_harmonic_ratio": round(H, 4)
        },
        "weighted_contributions": {
            "rigidity": round(w_R * R, 4),
            "topology": round(w_T * T, 4),
            "cultural": round(w_C * C, 4),
            "semantic": round(w_S * S, 4),
            "volatility": round(w_V * V, 4)
        },
        "provenance_tag": tag
    }


# ==============================================================================
#  VISUALIZATIONS
# ==============================================================================

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
    print(f"  📊 Generated: {filename_base}_roughness.png")

    # 2. TVT UMAP
    if HAS_UMAP:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.vstack([chroma, mfcc]).T

            if len(features) > 15:
                reducer = umap.UMAP(n_neighbors=min(15, len(features) - 1), min_dist=0.1, random_state=42)
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
                print(f"  📊 Generated: {filename_base}_tvt_umap.png")
        except Exception as e:
            print(f"  ⚠️ UMAP viz failed: {e}")

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
        print(f"  📊 Generated: {filename_base}_correlation.png")
    except Exception as e:
        print(f"  ⚠️ Correlation viz failed: {e}")


# ==============================================================================
#  FEATURE COUNTER
# ==============================================================================

def count_features(obj):
    """Count total individual features in nested output"""
    if isinstance(obj, dict):
        return sum(count_features(v) for v in obj.values())
    elif isinstance(obj, list):
        return len(obj) if obj and isinstance(obj[0], (int, float)) else sum(count_features(i) for i in obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return 1
    return 0


# ==============================================================================
#  TRACK PROCESSOR
# ==============================================================================

def process_track(audio_path, lexicon, output_dir, visualize=False):
    """Process a single track with mlx-whisper and full feature extraction"""
    audio_path = Path(audio_path)
    filename = audio_path.stem
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"🎵 {filename}")
    print('=' * 60)

    # 1. Transcribe with mlx-whisper (Apple Silicon optimized)
    print("  🎤 Transcribing with mlx-whisper (Apple Silicon)...")
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo="mlx-community/whisper-medium-mlx")
    transcript = result.get('text', '')
    segments = result.get('segments', [])
    print(f"  📝 Transcript: {len(transcript)} chars")

    # 2. Load audio for feature extraction
    print("  🔊 Loading audio...")
    y, sr = librosa.load(str(audio_path), sr=22050)

    # Limit to 5 min to avoid memory issues on long tracks
    max_samples = sr * 300
    if len(y) > max_samples:
        y = y[:max_samples]
        print(f"  ⚠️ Truncated to 5 minutes for feature extraction")

    # 3. Full feature extraction
    print("  📊 Extracting features...")

    # Metadata
    metadata = extract_metadata(audio_path, y, sr)

    # Spectral features
    spectral = extract_spectral_features(y, sr)

    # Rhythmic features
    rhythmic = extract_rhythmic_features(y, sr)

    # Harmonic features
    harmonic = extract_harmonic_features(y, sr)

    # Energy features
    energy = extract_energy_features(y, sr)

    # Linguistic features
    duration = metadata["duration_seconds"]
    linguistic = extract_linguistic_features(transcript, segments, duration)

    # AAVE analysis
    print("  🗣️ AAVE detection...")
    aave = analyze_aave(transcript, lexicon)

    # Sentiment analysis
    sentiment_score = 0.0
    if HAS_VADER:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(transcript)
        sentiment_score = sentiment['compound']

    # Reinman metrics
    print("  📐 Computing Reinman metrics...")
    reinman = compute_reinman_metrics(y, sr, sentiment_score)

    # God Equation (SCI - restored to original scale)
    god_eq = calculate_god_equation(
        spectral, aave, rhythmic, harmonic, energy, linguistic, reinman,
        file_hash=metadata.get("file_hash", "unknown")
    )

    # Visualizations
    if visualize:
        print("  🎨 Generating visualizations...")
        generate_visualizations(y, sr, output_dir, filename)

    # 4. Assemble full output
    output = {
        'metadata': metadata,
        'spectral': spectral,
        'rhythmic': rhythmic,
        'harmonic': harmonic,
        'energy': energy,
        'linguistic': linguistic,
        'aave': aave,
        'reinman_metrics': reinman,
        'god_equation': god_eq,
        'transcript': {
            'text': transcript[:500] + '...' if len(transcript) > 500 else transcript,
            'segments': segments
        },
        'processing_time_seconds': round(time.time() - start_time, 2)
    }

    output['total_features'] = count_features(output)

    # 5. Save JSON
    output_path = Path(output_dir) / f"{filename}_sovereign.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    feat_count = output['total_features']
    sci = god_eq['sci_with_aave']
    sci_no = god_eq['sci_without_aave']
    variance = god_eq['variance_percent']
    aave_density_val = aave['totals']['aave_density']

    print(f"  ✅ Saved: {output_path.name}")
    print(f"     Features: {feat_count}")
    print(f"     AAVE Density: {aave_density_val * 100:.2f}%")
    print(f"     SDS (Irony): {reinman['sds_score']}")
    print(f"     TVT (Complexity): {reinman['tvt_score']}")
    print(f"     SCI: {sci:.4f} (w/AAVE) | {sci_no:.4f} (no AAVE) | Δ{variance:+.1f}%")

    return output


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AoR v7.1 MLX - Apple Silicon Optimized, Full Feature Extraction (300+ features)'
    )
    parser.add_argument('tracks_file', help='File with list of track paths (one per line, # for comments)')
    parser.add_argument('--lexicon', default='aave_lexicon.json', help='AAVE lexicon path')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    print("=" * 60)
    print("AoR v7.1 MLX - FULL FEATURE EXTRACTION (300+)")
    print("Apple Silicon Optimized")
    print("=" * 60)
    print(f"  mlx-whisper:  ✅")
    print(f"  Librosa:      {'✅' if HAS_LIBROSA else '❌'}")
    print(f"  VADER:        {'✅' if HAS_VADER else '❌'}")
    print(f"  Matplotlib:   {'✅' if HAS_MATPLOTLIB else '❌'}")
    print(f"  UMAP:         {'✅' if HAS_UMAP else '❌'}")
    print(f"  Pronouncing:  {'✅' if HAS_PRONOUNCING else '❌'}")
    print("=" * 60)

    # Load lexicon
    lexicon = load_lexicon(args.lexicon)
    if lexicon:
        print(f"📚 Loaded lexicon: {args.lexicon}")
    else:
        print("📚 Using built-in AAVE lexicon (fallback)")

    # Load track list (skip empty lines and comments)
    with open(args.tracks_file, 'r') as f:
        tracks = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    # Validate paths
    valid_tracks = []
    for t in tracks:
        if os.path.exists(t):
            valid_tracks.append(t)
        else:
            print(f"⚠️  Track not found: {t}")
    tracks = valid_tracks

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

    # Final summary
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(results)}/{len(tracks)} tracks")
    print(f"Output: {args.output}")

    if results:
        avg_features = int(statistics.mean([r['total_features'] for r in results]))
        avg_aave = statistics.mean([r['aave']['totals']['aave_density'] for r in results])
        avg_sds = statistics.mean([r['reinman_metrics']['sds_score'] for r in results])
        avg_sci = statistics.mean([r['god_equation']['sci_with_aave'] for r in results])
        avg_sci_no = statistics.mean([r['god_equation']['sci_without_aave'] for r in results])
        avg_variance = statistics.mean([r['god_equation']['variance_percent'] for r in results])

        print(f"Avg Features/Track: {avg_features}")
        print(f"Avg AAVE Density: {avg_aave * 100:.2f}%")
        print(f"Avg SDS (Irony): {avg_sds:.4f}")
        print(f"Avg SCI (w/AAVE): {avg_sci:.4f}")
        print(f"Avg SCI (no AAVE): {avg_sci_no:.4f}")
        print(f"Avg AAVE Variance: {avg_variance:+.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
