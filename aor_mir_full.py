#!/usr/bin/env python3
"""
AoR-MIR v7.1 - FULL FEATURE EXTRACTION
=======================================
Maximum features, modular schema, AAVE as correctable layer.

Feature Groups (selectable by MIR teams):
- metadata: file info, duration, provenance
- spectral: chroma, mfcc, spectral features, rigidity
- rhythmic: tempo, beats, onset strength
- harmonic: tonnetz, harmonic ratios
- energy: rms, dynamics, loudness
- transcript: whisper output, word timing
- linguistic: word stats, vocabulary
- aave: dialect features (can apply post-hoc for bias correction)
- god_equation: composite scoring

AAVE Bias Correction:
- Can be applied real-time during analysis
- Can be applied post-processing to existing JSON
- Comparison mode: score with/without AAVE weighting
"""

import os
import sys
import json
import time
import hashlib
import warnings
import re
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ==============================================================================
#  DEPENDENCY DETECTION
# ==============================================================================

AVAILABLE_MODULES = {}

try:
    import numpy as np
    AVAILABLE_MODULES['numpy'] = True
except ImportError:
    AVAILABLE_MODULES['numpy'] = False

try:
    import librosa
    AVAILABLE_MODULES['librosa'] = True
except ImportError:
    AVAILABLE_MODULES['librosa'] = False

try:
    import torch
    AVAILABLE_MODULES['torch'] = True
    AVAILABLE_MODULES['mps'] = torch.backends.mps.is_available()
except ImportError:
    AVAILABLE_MODULES['torch'] = False
    AVAILABLE_MODULES['mps'] = False

try:
    import whisper
    AVAILABLE_MODULES['whisper'] = True
except ImportError:
    AVAILABLE_MODULES['whisper'] = False

try:
    import essentia.standard as es
    AVAILABLE_MODULES['essentia'] = True
except ImportError:
    AVAILABLE_MODULES['essentia'] = False

try:
    import pronouncing
    AVAILABLE_MODULES['pronouncing'] = True
except ImportError:
    AVAILABLE_MODULES['pronouncing'] = False

print(f"🔧 Available: {[k for k,v in AVAILABLE_MODULES.items() if v]}")

# ==============================================================================
#  AAVE KNOWLEDGE BASE (Comprehensive)
# ==============================================================================

AAVE_LEXICON = {
    "contractions": {
        "ain't", "gon'", "gonna", "gotta", "wanna", "tryna", "finna", "boutta",
        "bout", "cuz", "cause", "y'all", "imma", "lemme", "gimme", "kinda",
        "sorta", "dunno", "wassup", "whatchu", "aint", "yall", "ima", "ion",
        "iono", "aight", "ight", "nah", "yea", "yeah", "nope", "prolly",
        "shoulda", "woulda", "coulda", "mighta", "musta", "oughta", "supposta",
        "useta", "hafta", "needa", "wanna", "gotta", "gonna", "tryna", "finna"
    },
    "intensifiers": {
        "hella", "mad", "crazy", "straight", "real", "true", "dead", "deadass",
        "lowkey", "highkey", "super", "extra", "heavy", "deep", "hard", "raw",
        "wild", "sick", "dope", "fire", "lit", "turnt", "hype", "tight",
        "valid", "bussin", "slaps", "hits", "goated", "based", "cap", "nocap",
        "facts", "bet", "word", "legit", "forreal", "frfr", "ong", "onggod",
        "fasho", "fosho", "straight", "no doubt", "big", "major", "massive"
    },
    "markers": {
        "bruh", "bro", "fam", "cuh", "cuz", "dawg", "dog", "homie", "homeboy",
        "shorty", "shawty", "playa", "player", "pimp", "g", "og", "blood",
        "loc", "foo", "fool", "ese", "mane", "mayne", "son", "kid", "yo", "aye",
        "ay", "man", "dude", "folks", "folk", "gang", "gangsta", "thug", "nigga",
        "niggas", "brotha", "sista", "queen", "king", "young", "youngin", "lil"
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
        "file_hash": file_hash,
        "file_size_bytes": filepath.stat().st_size,
        "duration_seconds": round(len(y) / sr, 2),
        "sample_rate": sr,
        "samples": len(y),
        "channels": 1,  # mono
        "analysis_timestamp": datetime.now().isoformat(),
        "aor_version": "7.1-FULL"
    }

def extract_spectral_features(y, sr: int) -> Dict:
    """Comprehensive spectral analysis"""
    if not AVAILABLE_MODULES['librosa']:
        return {"error": "librosa not available"}

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

        # Spectral features
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = round(float(cent.mean()), 2)
        features["spectral_centroid_std"] = round(float(cent.std()), 2)

        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = round(float(bw.mean()), 2)
        features["spectral_bandwidth_std"] = round(float(bw.std()), 2)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff_mean"] = round(float(rolloff.mean()), 2)

        flatness = librosa.feature.spectral_flatness(y=y)
        features["spectral_flatness_mean"] = round(float(flatness.mean()), 6)

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
    if not AVAILABLE_MODULES['librosa']:
        return {"error": "librosa not available"}

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
    if not AVAILABLE_MODULES['librosa']:
        return {"error": "librosa not available"}

    features = {}

    try:
        # Harmonic ratio via spectral flatness (avoid HPSS crash)
        flatness = librosa.feature.spectral_flatness(y=y)
        features["harmonic_ratio"] = round(float(1.0 - flatness.mean()), 4)

        # Tonnetz (tonal centroid features) - use small segment to avoid crash
        segment_len = min(len(y), sr * 60)  # Max 60 seconds
        y_segment = y[:segment_len]

        try:
            # Compute harmonic component using CQT instead of HPSS
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
    if not AVAILABLE_MODULES['librosa']:
        return {"error": "librosa not available"}

    features = {}

    try:
        # RMS energy
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

def extract_linguistic_features(text: str, segments: List[Dict]) -> Dict:
    """Text and linguistic analysis (non-AAVE)"""
    features = {}

    words = re.findall(r'\b\w+\b', text.lower())
    features["word_count"] = len(words)
    features["unique_words"] = len(set(words))
    features["vocabulary_richness"] = round(len(set(words)) / max(len(words), 1), 4)

    # Character stats
    features["char_count"] = len(text)
    features["avg_word_length"] = round(sum(len(w) for w in words) / max(len(words), 1), 2)

    # Sentence approximation (by segments)
    features["segment_count"] = len(segments)

    # Words per segment
    if segments:
        words_per_seg = [len(s.get('text', '').split()) for s in segments]
        features["words_per_segment_mean"] = round(statistics.mean(words_per_seg) if words_per_seg else 0, 2)
        features["words_per_segment_std"] = round(statistics.stdev(words_per_seg) if len(words_per_seg) > 1 else 0, 2)

    # Syllable estimation (simple)
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

    # Rhyme analysis (if pronouncing available)
    if AVAILABLE_MODULES.get('pronouncing'):
        try:
            rhyme_pairs = 0
            for i, w1 in enumerate(words[:-1]):
                rhymes = set(pronouncing.rhymes(w1))
                if words[i+1] in rhymes:
                    rhyme_pairs += 1
            features["adjacent_rhymes"] = rhyme_pairs
        except:
            features["adjacent_rhymes"] = 0

    return features

def analyze_aave(text: str, lexicon: Dict = None) -> Dict:
    """
    AAVE Dialect Analysis

    This module can be applied:
    1. Real-time during analysis
    2. Post-processing to existing transcripts

    Returns detailed AAVE features for bias correction.
    """
    if lexicon is None:
        lexicon = AAVE_LEXICON

    features = {
        "lexicon_source": "custom" if lexicon != AAVE_LEXICON else "default",
        "categories": {}
    }

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_set = set(words)

    total_matches = 0
    all_matched_terms = []

    # Check each category
    for category, terms in lexicon.items():
        if isinstance(terms, set):
            matched = [w for w in word_set if w in terms]
            features["categories"][category] = {
                "matched_count": len(matched),
                "matched_terms": matched[:20],  # Limit for output size
                "density": round(len(matched) / max(len(words), 1), 6)
            }
            total_matches += len(matched)
            all_matched_terms.extend(matched)

    # Grammar patterns
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
    features["totals"] = {
        "lexical_matches": total_matches,
        "grammar_matches": sum(g["count"] for g in grammar_matches),
        "combined_matches": total_matches + sum(g["count"] for g in grammar_matches),
        "aave_density": round((total_matches + grammar_score) / max(len(words), 1), 6),
        "aave_richness": round(len(set(all_matched_terms)) / max(len(lexicon.get("markers", set())), 1), 4)
    }

    # All matched terms for reference
    features["all_matched_terms"] = list(set(all_matched_terms))[:50]

    return features

def calculate_god_equation(spectral: Dict, aave: Dict, rhythmic: Dict,
                           apply_aave_correction: bool = True) -> Dict:
    """
    THE GOD EQUATION - Composite Scoring

    Two modes:
    1. WITH AAVE correction (apply_aave_correction=True) - fair scoring
    2. WITHOUT AAVE correction (apply_aave_correction=False) - biased baseline

    This allows comparison to demonstrate systemic bias.
    """

    # Base components
    rigidity = spectral.get("spectral_rigidity", 0)
    harmonic = spectral.get("spectral_flatness_mean", 0.5)
    harmonic_ratio = 1.0 - harmonic  # Invert flatness
    tempo_norm = min(rhythmic.get("tempo_bpm", 120) / 180, 1.0)
    beat_reg = rhythmic.get("beat_regularity", 0.5)

    # AAVE component
    aave_density = aave.get("totals", {}).get("aave_density", 0)
    aave_richness = aave.get("totals", {}).get("aave_richness", 0)

    # Calculate SCI (Spectral Coherence Index) WITHOUT AAVE
    sci_base = (
        rigidity * 2.0 +
        harmonic_ratio * 1.5 +
        tempo_norm * 1.0 +
        beat_reg * 1.5
    ) / 6.0  # Normalize to ~0-1 range

    sci_base = min(max(sci_base * 10, 0), 10)  # Scale to 0-10

    # Calculate SCI WITH AAVE correction
    aave_boost = (aave_density * 15 + aave_richness * 5) if apply_aave_correction else 0
    sci_corrected = sci_base + aave_boost
    sci_corrected = min(max(sci_corrected, 0), 10)

    # MIR (Maximum Intelligence Resonance) - the final score
    mir_score = sci_corrected

    # Variance calculation (for bias demonstration)
    variance_pct = round(((sci_corrected - sci_base) / max(sci_base, 0.01)) * 100, 2) if sci_base > 0 else 0

    return {
        "sci_without_aave": round(sci_base, 4),
        "sci_with_aave": round(sci_corrected, 4),
        "aave_correction_applied": apply_aave_correction,
        "aave_boost": round(aave_boost, 4),
        "variance_percent": variance_pct,
        "mir_score": round(mir_score, 4),
        "components": {
            "rigidity": round(rigidity * 2.0, 4),
            "harmonic": round(harmonic_ratio * 1.5, 4),
            "tempo": round(tempo_norm * 1.0, 4),
            "beat_regularity": round(beat_reg * 1.5, 4),
            "aave_density": round(aave_density * 15, 4) if apply_aave_correction else 0,
            "aave_richness": round(aave_richness * 5, 4) if apply_aave_correction else 0
        },
        "provenance_tag": f"AOR-MIR-{datetime.now().strftime('%Y%m%d')}-{int(mir_score*100):04d}"
    }

# ==============================================================================
#  MAIN ANALYZER
# ==============================================================================

class FullAnalyzer:
    """Full-feature MIR analyzer with modular output"""

    def __init__(self, whisper_model: str = "small", lexicon: Dict = None):
        self.whisper_model = whisper_model
        self.model = None
        self.lexicon = lexicon or AAVE_LEXICON
        self.device = "cpu"  # CPU for stability

    def load_whisper(self):
        if not AVAILABLE_MODULES['whisper']:
            print("❌ Whisper not available")
            return
        print(f"🧠 Loading Whisper '{self.whisper_model}'...")
        self.model = whisper.load_model(self.whisper_model, device=self.device)
        print("✅ Whisper loaded")

    def transcribe(self, filepath: Path) -> Dict:
        if not self.model:
            return {"text": "", "segments": []}
        try:
            result = self.model.transcribe(str(filepath), word_timestamps=True, language="en")
            return {"text": result.get("text", ""), "segments": result.get("segments", [])}
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return {"text": "", "segments": []}

    def analyze_file(self, filepath: Path, apply_aave: bool = True) -> Dict:
        """Analyze single file with full feature extraction"""
        filepath = Path(filepath)
        start_time = time.time()

        print(f"📂 Analyzing: {filepath.name}")

        # Load audio
        print("  🔊 Loading audio...")
        y, sr = librosa.load(str(filepath), sr=22050, mono=True)

        # Limit to 5 min to avoid memory issues
        max_samples = sr * 300
        if len(y) > max_samples:
            y = y[:max_samples]
            print(f"  ⚠️ Truncated to 5 minutes")

        # Transcribe
        print("  🎤 Transcribing...")
        transcript = self.transcribe(filepath)

        # Extract all feature groups
        print("  📊 Extracting features...")

        result = {
            "metadata": extract_metadata(filepath, y, sr),
            "spectral": extract_spectral_features(y, sr),
            "rhythmic": extract_rhythmic_features(y, sr),
            "harmonic": extract_harmonic_features(y, sr),
            "energy": extract_energy_features(y, sr),
            "linguistic": extract_linguistic_features(transcript["text"], transcript["segments"]),
            "aave": analyze_aave(transcript["text"], self.lexicon),
            "transcript": {
                "text": transcript["text"],
                "segments": transcript["segments"]
            }
        }

        # Calculate God Equation (both with and without AAVE for comparison)
        result["god_equation_with_aave"] = calculate_god_equation(
            result["spectral"], result["aave"], result["rhythmic"],
            apply_aave_correction=True
        )
        result["god_equation_without_aave"] = calculate_god_equation(
            result["spectral"], result["aave"], result["rhythmic"],
            apply_aave_correction=False
        )

        # Use requested mode for primary score
        result["god_equation"] = result["god_equation_with_aave"] if apply_aave else result["god_equation_without_aave"]

        result["processing_time_seconds"] = round(time.time() - start_time, 2)

        # Count total features
        def count_features(obj):
            if isinstance(obj, dict):
                return sum(count_features(v) for v in obj.values())
            elif isinstance(obj, list):
                return len(obj) if obj and isinstance(obj[0], (int, float)) else sum(count_features(i) for i in obj)
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return 1
            return 0

        result["total_features"] = count_features(result)

        return result

    def process_batch(self, input_dir: str, output_dir: str, apply_aave: bool = True):
        """Process directory of audio files"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find files
        extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
        files = []
        for ext in extensions:
            files.extend(input_path.glob(ext))

        if not files:
            print(f"❌ No audio files found in {input_dir}")
            return

        print(f"\n{'='*60}")
        print(f"🚀 AoR-MIR v7.1 - FULL FEATURE EXTRACTION")
        print(f"{'='*60}")
        print(f"📁 Input:  {input_path}")
        print(f"📂 Output: {output_path}")
        print(f"🎵 Files:  {len(files)}")
        print(f"🔧 AAVE Correction: {'ON' if apply_aave else 'OFF'}")
        print(f"{'='*60}\n")

        self.load_whisper()

        results = []
        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {f.name}")

            result = self.analyze_file(f, apply_aave)

            # Save individual result
            out_file = output_path / f"{f.stem}_aor-mir-full.json"
            with open(out_file, 'w') as jf:
                json.dump(result, jf, indent=2, default=str)

            mir = result["god_equation"]["mir_score"]
            variance = result["god_equation_with_aave"]["variance_percent"]
            features = result["total_features"]

            print(f"  ✅ MIR: {mir:.2f} | Variance: {variance:+.1f}% | Features: {features}")
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print(f"📊 BATCH COMPLETE")
        print(f"{'='*60}")
        mir_scores = [r["god_equation"]["mir_score"] for r in results]
        variances = [r["god_equation_with_aave"]["variance_percent"] for r in results]
        features = [r["total_features"] for r in results]
        print(f"   Files: {len(results)}")
        print(f"   Avg MIR: {statistics.mean(mir_scores):.2f}")
        print(f"   Avg AAVE Variance: {statistics.mean(variances):+.1f}%")
        print(f"   Avg Features: {int(statistics.mean(features))}")
        print(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AoR-MIR v7.1 Full Feature Extraction")
    parser.add_argument("audio_dir", help="Directory with audio files")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--model", "-m", default="small", help="Whisper model")
    parser.add_argument("--no-aave", action="store_true", help="Disable AAVE correction (biased mode)")

    args = parser.parse_args()

    analyzer = FullAnalyzer(whisper_model=args.model)
    analyzer.process_batch(args.audio_dir, args.output, apply_aave=not args.no_aave)


if __name__ == "__main__":
    main()
