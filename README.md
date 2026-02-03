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

### Discovery 2: AI Cultural Intelligence Gap

Using our expanded lexicon (807 terms) and Reinman metrics, we found that **AI-generated hip-hop demonstrates significantly lower Semantic Cultural Intelligence (SCI)**:

| Metric | Human Artists | AI Generated | Human Advantage |
|--------|---------------|--------------|-----------------|
| AAVE Density | 6.18% | 3.72% | **+66%** |
| Unique AAVE Terms | 135 | 60 | **+125%** |
| SDS (Semantic Dissonance) | 1.51 | 0.72 | **+110%** |
| TVT (Tonal Complexity) | 588 | 499 | **+18%** |

**Interpretation:** AI systems trained without adequate AAVE representation produce content that:
- Uses less diverse Black vernacular vocabulary
- Lacks the ironic/subversive layering (low SDS) characteristic of authentic hip-hop
- Demonstrates flattened tonal complexity

This suggests current AI models have a **cultural intelligence deficit** when generating content rooted in Black American linguistic traditions. The absence of AAVE in training data results in outputs that are measurably less authentic.

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

# Batch processing
python aor_mir.py /path/to/music/ --batch --output results/
```

### MLX-Optimized (Apple Silicon)

For 5-10x faster processing on M-series Macs:

```bash
# Create track list
find /path/to/music -name "*.m4a" > tracks.txt

# Run with MLX acceleration
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

Corpus-level analytics:

- `reinman_plane.png` - SDS vs TVT scatter plot
- `aave_distribution.png` - AAVE density histogram
- `irony_matrix.png` - Sentiment-audio alignment
- `aave_wordcloud.png` - Detected term frequencies
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
├── aor_mir_mlx.py          # Apple Silicon optimized version
├── aave_lexicon.json       # AAVE lexicon v3.0 (807 terms)
├── advanced_analytics.py   # Publication visualization suite
├── compare_human_ai.py     # Corpus comparison tools
├── requirements.txt        # Python dependencies
├── output/
│   ├── corpus_analysis_v3/     # Human corpus results
│   ├── ai_corpus_analysis_v3/  # AI corpus results
│   ├── human_vs_ai_comparison_v3.png
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
