# AoR-MIR: Architect of Rhyme - Music Information Retrieval

**A computational framework for analyzing African American Vernacular English (AAVE) in hip-hop music, with novel Reinman topology metrics for semantic analysis.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AoR-MIR (Architect of Rhyme - Music Information Retrieval) is a Python-based toolkit for:

1. **AAVE Detection** - Lexicon-based identification of African American Vernacular English in transcribed lyrics
2. **Reinman Metrics** - Novel semantic topology measurements (SDS, TVT, Spectral Rigidity)
3. **Human vs AI Comparison** - Framework for comparing authentic hip-hop with AI-generated content
4. **Publication-Ready Visualizations** - Automated generation of research figures

## Key Findings

### ASR Bias Discovery

During development, we discovered that **OpenAI's Whisper systematically erases AAVE phonological markers**:

| AAVE Form | Standard Form | Preservation Rate |
|-----------|---------------|-------------------|
| dis       | this          | 0%                |
| dat       | that          | 0%                |
| dey       | they          | 0%                |
| dem       | them          | 0%                |

**100% of AAVE phonological features are normalized to Standard American English.**

This constitutes algorithmic bias that erases Black linguistic identity in transcription. Full documentation in `output/FINDING_ASR_AAVE_BIAS.md`.

### Human vs AI Analysis

Using our expanded lexicon (807 terms), we found significant differences between human-created and AI-generated hip-hop:

| Metric | Human Artists | AI Generated | Difference |
|--------|---------------|--------------|------------|
| AAVE Density | 6.18% | 3.72% | +66% |
| Unique Terms | 135 | 60 | +125% |
| SDS (Irony) | 1.51 | 0.72 | +110% |
| TVT (Complexity) | 588 | 499 | +18% |

**Human artists demonstrate significantly higher AAVE vocabulary diversity and semantic complexity.**

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aor-mir.git
cd aor-mir

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
aor-mir/
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
  title = {AoR-MIR: Architect of Rhyme Music Information Retrieval},
  year = {2026},
  url = {https://github.com/yourusername/aor-mir}
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

- AAVE corpus data from [jazmiahenry/aave_corpora](https://github.com/jazmiahenry/aave_corpora)
- Whisper by OpenAI
- MLX by Apple

---

**Note:** This tool is designed for non-consumptive research under Fair Use principles. It analyzes linguistic patterns without reproducing copyrighted content.
