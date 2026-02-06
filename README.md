# AoR-DMMA: Digital Music Metadata Analysis

**A computational framework for analyzing African American Vernacular English (AAVE) in hip-hop music, with novel Reinman topology metrics for semantic analysis.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-aor--source-blue)](https://github.com/aor-source/aor-dmma)

---

## Overview

AoR-DMMA is a Python-based toolkit for:

1. **AAVE Detection** - Lexicon-based identification of African American Vernacular English in transcribed lyrics
2. **Reinman Metrics** - Novel semantic topology measurements (SDS, TVT, Spectral Rigidity)
3. **Human vs AI Comparison** - Framework for comparing authentic hip-hop with AI-generated content
4. **Publication-Ready Visualizations** - Automated generation of research figures

## Key Findings

This research produced **two critical discoveries** about AI systems and Black linguistic representation:

### Discovery 1: ASR Erasure Bias (Whisper)

During development, we discovered that **OpenAI's Whisper systematically erases AAVE phonological markers**:

| AAVE Form | Standard Form | Preservation Rate |
|-----------|---------------|-------------------|
| dis       | this          | 0%                |
| dat       | that          | 0%                |
| dey       | they          | 0%                |
| dem       | them          | 0%                |
| dese      | these         | 0%                |
| dose      | those         | 0%                |

**100% of AAVE phonological features are normalized to Standard American English.**

This constitutes **algorithmic whitewashing** - a form of ML bias that erases Black linguistic identity at the transcription layer. Downstream NLP systems trained on Whisper transcripts inherit this bias. Full documentation in `output/FINDING_ASR_AAVE_BIAS.md`.

### Discovery 2: AI Cultural Intelligence Gap (Updated v7.1)

Using our expanded lexicon (807 terms), Reinman metrics, and corrected SCI formula across a 34-track corpus (17 human hip-hop, 7 AI/Suno-generated, 10 experimental), we found that **AI-generated hip-hop is closing the cultural intelligence gap**:

| Metric | Human Hip-Hop (17) | AI/Suno (7) | Gap |
|--------|-------------------|-------------|-----|
| **SCI (Spectral Coherence Index)** | **3.32** | **3.05** | 8% |
| AAVE Density | Higher | Lower | Varies |
| SDS (Semantic Dissonance) | 1.51 | 0.72 | +110% |
| TVT (Tonal Complexity) | 588 | 499 | +18% |

**Key Findings (v7.1):**

1. **AI tracks achieve 92% of human SCI** — The cultural intelligence gap is narrowing. AI-generated hip-hop (lyrics by Claude, audio by Suno) scores within 8% of canonical human hip-hop on the composite SCI metric.

2. **AAVE bias causes 50-88% undervaluation** — Tracks scored without AAVE context lose up to 88% of their true cultural intelligence score. This confirms AAVE detection is critical infrastructure, not optional.

3. **The Fractal (Z=Z²+C)** — An AI-written mathematical hip-hop track featuring a Fibonacci sequence in both lyric content AND structural form, demonstrating emergent creative sophistication.

4. **SCI Scale Correction** — The v7.1 SCI formula now matches the original research scale where scores above 4.0 are exceptional and 6.0+ was theoretically unreachable. Previous versions inflated scores to a 10/10 ceiling.

### SCI Formula (v7.1 — "The God Equation")

The Spectral Coherence Index combines five weighted components with a decay penalty:

```
R = Spectral Rigidity       (w=0.25)
T = Topological Valence      (w=0.15)
C = Cultural/AAVE Density    (w=0.25)
S = Semantic Richness         (w=0.20)
V = Volatility/Dynamics       (w=0.15)
D = Decay Penalty (vocab sparsity)

I = (0.25*R) + (0.15*T) + (0.25*C) + (0.20*S) + (0.15*V)
SCI = I / (1 + D*0.1)
```

Scale: 0-5 range. Scores above 4.0 indicate exceptional cultural-sonic coherence.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aor-source/aor-dmma.git
cd aor-dmma

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- PyTorch 2.0+
- mlx-whisper (Apple Silicon optimized) or openai-whisper
- librosa
- vaderSentiment
- matplotlib, seaborn
- umap-learn
- numpy, pandas

---

## Usage

### Basic Analysis

```bash
# Analyze a single track
python aor_mir.py track.mp3 --lexicon aave_lexicon.json --visualize
```

### Batch Processing

There are two ways to batch process multiple tracks:

**Option 1: Directory mode** — point at a folder and all audio files (`.mp3`, `.wav`, `.flac`, `.m4a`) inside it are processed:

```bash
python aor_mir.py /path/to/music/ --batch --lexicon aave_lexicon.json --output results/
```

**Option 2: Track list mode** — create a `tracks.txt` file with one absolute path per line, then pass it directly:

```bash
# Create tracks.txt (one absolute path per line)
find /path/to/music -name "*.wav" -o -name "*.mp3" -o -name "*.m4a" > tracks.txt

# Or create it manually:
# /Users/you/music/track1.wav
# /Users/you/music/track2.mp3
# Lines starting with # are ignored

# Run analysis
python aor_mir.py tracks.txt --lexicon aave_lexicon.json --output results/ --visualize
```

Both `aor_mir.py` and `aor_mir_mlx.py` accept `tracks.txt` as input.

### MLX-Optimized (Apple Silicon)

For 5-10x faster processing on M-series Macs:

```bash
python aor_mir_mlx.py tracks.txt --lexicon aave_lexicon.json --output results/ --visualize
```

### Generate Analytics

```bash
# Publication-ready visualizations
python advanced_analytics.py results/ --output results/analytics/
```

---

## Output Files

Each analyzed track produces:

- `{track}_sovereign.json` - Full metrics and transcript
- `{track}_roughness.png` - Spectral centroid visualization
- `{track}_tvt_umap.png` - Topological trajectory UMAP
- `{track}_correlation.png` - Feature correlation heatmap

Corpus-level analytics (v7.1):

- `sci_comparison.png` - SCI scores by corpus group (Human vs AI vs Experimental)
- `aave_bias_variance.png` - AAVE bias impact ranked by track
- `reinman_plane_comparison.png` - SDS vs TVT scatter with group coloring
- `sci_radar_comparison.png` - Spider chart of 5 SCI components
- `sci_bias_scatter.png` - With/without AAVE bias demonstration

Legacy analytics:

- `reinman_plane.png` - SDS vs TVT scatter plot
- `aave_distribution.png` - AAVE density histogram
- `irony_matrix.png` - Sentiment-audio alignment
- `paper_statistics.json` - Key metrics for publication

---

## Reinman Metrics

### SDS - Semantic Dissonance Score

Measures the divergence between lyric sentiment and audio energy. High SDS indicates ironic or subversive content where lyrics and music convey opposite affects.

```
SDS = |lyric_valence - (audio_arousal - 0.5) * 2|
```

### TVT - Topological Valence Trajectory

Quantifies the spectral complexity of the audio, measuring how much the sonic texture varies over time.

```
TVT = std(diff(spectral_centroid))
```

### Spectral Rigidity

Normalized measure of spectral rolloff indicating timbral consistency.

---

## AAVE Lexicon

The lexicon (`aave_lexicon.json`) contains 807 searchable terms across categories:

- **Contractions** - ain't, gonna, finna, tryna, etc.
- **Semantic Inversions** - Words with inverted sentiment (bad=good, sick=excellent)
- **Slang Terms** - Money, locations, people, actions, etc.
- **Regional Variants** - South, East Coast, West Coast, Midwest, Atlanta
- **Classic Hip-Hop Terms** - 80s/90s, Golden Era, Modern

### Lexicon Statistics

```
Version: 3.0-expanded
Total searchable terms: 807
Categories: 12
Semantic inversions: 31
Regional variants: 133
```

---

## Project Structure

```
aor-dmma/
├── aor_mir.py              # Main analysis script (v7.0 SOVEREIGN)
├── aor_mir_mlx.py          # Apple Silicon MLX — full 300+ feature extraction (v7.1)
├── aor_mir_full.py         # Reference full-feature script
├── aave_lexicon.json       # AAVE lexicon v3.0 (807 terms)
├── tracks.txt              # Batch track list (one absolute path per line)
├── check_progress.sh       # Quick progress checker for batch runs
├── progress_monitor.sh     # Live progress monitor for terminal
├── advanced_analytics.py   # Publication visualization suite
├── compare_human_ai.py     # Corpus comparison tools
├── components/             # Pipeline components (assemble, analyze, etc.)
├── requirements.txt        # Python dependencies
├── output/
│   ├── full_v71/               # v7.1 MLX results (34 tracks + comparisons)
│   ├── corpus_analysis/        # Human corpus results
│   ├── ai_corpus_analysis/     # AI corpus results
│   ├── FINDING_ASR_AAVE_BIAS.md
│   └── FINDING_ASR_AAVE_BIAS.json
└── README.md
```

---

## Research Applications

- **Hip-Hop Scholarship** - Quantitative analysis of AAVE in rap lyrics
- **AI Detection** - Distinguishing human vs AI-generated content
- **ASR Bias Auditing** - Evaluating transcription systems for dialectal bias
- **Cultural Preservation** - Documenting linguistic patterns in Black music

---

## Citation

If you use AoR-MIR in your research, please cite:

```bibtex
@software{aor_mir_2026,
  author = {Wright, Jon},
  title = {AoR-MIR: Music Information Retrieval},
  year = {2026},
  url = {https://github.com/aor-source/aor-dmma}
}
```

---

## Known Limitations

1. **ASR Normalization** - Whisper erases AAVE phonological markers; detection relies on lexical items that survive transcription
2. **Lexicon Coverage** - Not all AAVE terms are included; contributions welcome
3. **Context Sensitivity** - Some terms have AAVE meaning only in specific contexts

---

## Contributing

Contributions welcome! Areas of interest:

- Expanded AAVE lexicon terms
- Dialect-aware ASR alternatives
- Additional Reinman metrics
- Regional variant detection

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

### AAVE Corpora Dataset
This project is built upon the foundational work of **Jazmia Henry** and the [AAVE Corpora](https://github.com/jazmiahenry/aave_corpora) project (MIT License).

> *"This dataset was created by AAVE speakers for AAVE speakers and the engineers, academics, researchers, and builders that endeavor to create NLP models that represent the beauty and complexity of the AAVE sociolect."*

We gratefully acknowledge this contribution to preserving and analyzing African American linguistic heritage.

### Additional Credits
- Whisper by OpenAI
- MLX by Apple

---

**Note:** This tool is designed for non-consumptive research under Fair Use principles. It analyzes linguistic patterns without reproducing copyrighted content.
