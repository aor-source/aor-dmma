#!/usr/bin/env python3
"""
AoR Advanced Analytics & Visualization Suite
============================================
Generates publication-ready visualizations from sovereign JSON outputs.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try additional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Style config
plt.style.use('dark_background')
COLORS = {
    'primary': '#00d4ff',
    'secondary': '#ff006e',
    'accent': '#7b2cbf',
    'positive': '#00ff88',
    'negative': '#ff4444'
}


def load_results(results_dir):
    """Load all sovereign JSON results into a DataFrame"""
    data = []
    for json_file in glob.glob(f"{results_dir}/*_sovereign.json"):
        with open(json_file, 'r') as f:
            result = json.load(f)

        # Extract artist from path/filename
        filename = result['metadata']['file']

        row = {
            'file': filename,
            'duration': result['metadata']['duration'],
            'aave_density': result['cultural_metrics']['aave_density'],
            'unique_terms': result['cultural_metrics']['unique_terms'],
            'grammar_patterns': result['cultural_metrics']['grammar_patterns'],
            'sds_score': result['reinman_metrics']['sds_score'],
            'tvt_score': result['reinman_metrics']['tvt_score'],
            'spectral_rigidity': result['reinman_metrics']['spectral_rigidity'],
            'lyric_valence': result['reinman_metrics']['lyric_valence'],
            'audio_arousal': result['reinman_metrics']['audio_arousal'],
            'found_terms': result['cultural_metrics'].get('found_terms', []),
            'source': result['cultural_metrics']['source']
        }
        data.append(row)

    return pd.DataFrame(data)


def plot_reinman_scatter(df, output_dir):
    """SDS vs TVT scatter plot - The Reinman Plane"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
    ax.set_facecolor('black')

    scatter = ax.scatter(
        df['sds_score'],
        df['tvt_score'],
        c=df['aave_density'],
        cmap='plasma',
        s=df['duration'] / 5,  # Size by duration
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('AAVE Density', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xlabel('Semantic Dissonance Score (SDS) - Irony Index', color='white', fontsize=12)
    ax.set_ylabel('Topological Valence Trajectory (TVT) - Complexity', color='white', fontsize=12)
    ax.set_title('The Reinman Plane: Hip-Hop Semantic Topology', color='white', fontsize=14)
    ax.tick_params(colors='white')

    # Add quadrant labels
    ax.axhline(y=df['tvt_score'].median(), color='white', linestyle='--', alpha=0.3)
    ax.axvline(x=df['sds_score'].median(), color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reinman_plane.png", dpi=200, facecolor='black')
    plt.close()
    print(f"📊 Generated: reinman_plane.png")


def plot_aave_distribution(df, output_dir):
    """AAVE density distribution with KDE"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='black')

    # Histogram with KDE
    ax1 = axes[0]
    ax1.set_facecolor('black')
    sns.histplot(df['aave_density'] * 100, kde=True, ax=ax1, color=COLORS['primary'], alpha=0.7)
    ax1.set_xlabel('AAVE Density (%)', color='white')
    ax1.set_ylabel('Track Count', color='white')
    ax1.set_title('AAVE Density Distribution Across Corpus', color='white')
    ax1.tick_params(colors='white')

    # Box plot by approximate artist (from filename)
    ax2 = axes[1]
    ax2.set_facecolor('black')

    # Simple artist extraction
    df['artist'] = df['file'].apply(lambda x: x.split('_')[0][:15] if '_' in x else x[:15])

    top_artists = df.groupby('artist')['aave_density'].mean().nlargest(10).index
    df_top = df[df['artist'].isin(top_artists)]

    sns.boxplot(data=df_top, x='artist', y='aave_density', ax=ax2, palette='plasma')
    ax2.set_xlabel('Track', color='white')
    ax2.set_ylabel('AAVE Density', color='white')
    ax2.set_title('AAVE Density by Track', color='white')
    ax2.tick_params(colors='white', axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aave_distribution.png", dpi=200, facecolor='black')
    plt.close()
    print(f"📊 Generated: aave_distribution.png")


def plot_sentiment_audio_matrix(df, output_dir):
    """Lyric Valence vs Audio Arousal - The Irony Detector"""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')

    # Create quadrant colors based on alignment/dissonance
    colors = []
    for _, row in df.iterrows():
        if row['lyric_valence'] > 0 and row['audio_arousal'] > 0.5:
            colors.append(COLORS['positive'])  # Aligned positive
        elif row['lyric_valence'] < 0 and row['audio_arousal'] < 0.5:
            colors.append(COLORS['secondary'])  # Aligned negative
        else:
            colors.append(COLORS['accent'])  # Dissonant/Ironic

    scatter = ax.scatter(
        df['lyric_valence'],
        df['audio_arousal'],
        c=colors,
        s=100,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )

    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.3)

    # Quadrant labels
    ax.text(0.7, 0.9, 'ALIGNED\nPositive Lyrics\nHigh Energy', color=COLORS['positive'],
            fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.3, 0.1, 'ALIGNED\nNegative Lyrics\nLow Energy', color=COLORS['secondary'],
            fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.7, 0.1, 'IRONIC\nPositive Lyrics\nLow Energy', color=COLORS['accent'],
            fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.3, 0.9, 'IRONIC\nNegative Lyrics\nHigh Energy', color=COLORS['accent'],
            fontsize=10, ha='center', transform=ax.transAxes)

    ax.set_xlabel('Lyric Valence (VADER Sentiment)', color='white', fontsize=12)
    ax.set_ylabel('Audio Arousal (Normalized RMS)', color='white', fontsize=12)
    ax.set_title('Semantic-Audio Alignment Matrix\n(Irony Detection via SDS)', color='white', fontsize=14)
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/irony_matrix.png", dpi=200, facecolor='black')
    plt.close()
    print(f"📊 Generated: irony_matrix.png")


def plot_correlation_heatmap(df, output_dir):
    """Full metric correlation heatmap"""
    numeric_cols = ['aave_density', 'sds_score', 'tvt_score', 'spectral_rigidity',
                    'lyric_valence', 'audio_arousal', 'duration', 'unique_terms']

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
    ax.set_facecolor('black')

    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                ax=ax, fmt='.2f', linewidths=0.5,
                cbar_kws={'label': 'Correlation'})

    ax.set_title('AoR Metric Correlation Matrix', color='white', fontsize=14)
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metric_correlations.png", dpi=200, facecolor='black')
    plt.close()
    print(f"📊 Generated: metric_correlations.png")


def plot_aave_wordcloud(df, output_dir):
    """Word cloud of found AAVE terms"""
    if not HAS_WORDCLOUD:
        print("⚠️ wordcloud not installed - skipping")
        return

    # Collect all found terms
    all_terms = []
    for terms in df['found_terms']:
        if isinstance(terms, list):
            all_terms.extend(terms)

    if not all_terms:
        print("⚠️ No AAVE terms found - skipping wordcloud")
        return

    # Count frequencies
    term_freq = defaultdict(int)
    for term in all_terms:
        term_freq[term.lower()] += 1

    wc = WordCloud(
        width=1600, height=800,
        background_color='black',
        colormap='plasma',
        max_words=50,
        min_font_size=8,
        relative_scaling=0.5
    ).generate_from_frequencies(term_freq)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='black')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('AAVE Terms Detected Across Corpus', color='white', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/aave_wordcloud.png", dpi=200, facecolor='black')
    plt.close()
    print(f"📊 Generated: aave_wordcloud.png")


def plot_radar_comparison(df, output_dir, top_n=5):
    """Radar chart comparing top tracks by AAVE density"""
    metrics = ['aave_density', 'sds_score', 'tvt_score', 'spectral_rigidity', 'lyric_valence']

    # Normalize metrics to 0-1 for radar
    df_norm = df.copy()
    for m in metrics:
        if df_norm[m].max() > df_norm[m].min():
            df_norm[m] = (df_norm[m] - df_norm[m].min()) / (df_norm[m].max() - df_norm[m].min())

    # Get top tracks by AAVE density
    top_tracks = df_norm.nlargest(top_n, 'aave_density')

    # Radar setup
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor='black')
    ax.set_facecolor('black')

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, top_n))

    for idx, (_, row) in enumerate(top_tracks.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=row['file'][:25], color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['AAVE\nDensity', 'SDS\n(Irony)', 'TVT\n(Complexity)',
                        'Spectral\nRigidity', 'Lyric\nValence'], color='white')
    ax.set_title('Multi-Metric Radar: Top AAVE-Dense Tracks', color='white', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=8)
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_comparison.png", dpi=200, facecolor='black', bbox_inches='tight')
    plt.close()
    print(f"📊 Generated: radar_comparison.png")


def generate_paper_stats(df, output_dir):
    """Generate key statistics for the paper"""
    stats = {
        "corpus_summary": {
            "total_tracks": len(df),
            "total_duration_minutes": round(df['duration'].sum() / 60, 1),
            "avg_duration_seconds": round(df['duration'].mean(), 1)
        },
        "aave_metrics": {
            "mean_density": round(df['aave_density'].mean() * 100, 2),
            "std_density": round(df['aave_density'].std() * 100, 2),
            "max_density": round(df['aave_density'].max() * 100, 2),
            "min_density": round(df['aave_density'].min() * 100, 2),
            "total_unique_terms": int(df['unique_terms'].sum()),
            "total_grammar_patterns": int(df['grammar_patterns'].sum())
        },
        "reinman_metrics": {
            "mean_sds": round(df['sds_score'].mean(), 4),
            "mean_tvt": round(df['tvt_score'].mean(), 2),
            "mean_spectral_rigidity": round(df['spectral_rigidity'].mean(), 4),
            "sds_tvt_correlation": round(df['sds_score'].corr(df['tvt_score']), 4)
        },
        "sentiment_analysis": {
            "mean_lyric_valence": round(df['lyric_valence'].mean(), 4),
            "mean_audio_arousal": round(df['audio_arousal'].mean(), 4),
            "valence_arousal_correlation": round(df['lyric_valence'].corr(df['audio_arousal']), 4)
        },
        "high_impact_tracks": df.nlargest(5, 'aave_density')[['file', 'aave_density', 'sds_score']].to_dict('records')
    }

    with open(f"{output_dir}/paper_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"📄 Generated: paper_statistics.json")
    return stats


def main(results_dir, output_dir):
    """Run all analytics"""
    print("=" * 60)
    print("AoR ADVANCED ANALYTICS SUITE")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_results(results_dir)
    print(f"📁 Loaded {len(df)} track results")

    if len(df) == 0:
        print("❌ No results found!")
        return

    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    plot_reinman_scatter(df, output_dir)
    plot_aave_distribution(df, output_dir)
    plot_sentiment_audio_matrix(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_aave_wordcloud(df, output_dir)
    plot_radar_comparison(df, output_dir)

    # Generate stats
    print("\n📊 Generating paper statistics...")
    stats = generate_paper_stats(df, output_dir)

    print("\n" + "=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    print(f"Total Tracks: {stats['corpus_summary']['total_tracks']}")
    print(f"Total Duration: {stats['corpus_summary']['total_duration_minutes']} minutes")
    print(f"Mean AAVE Density: {stats['aave_metrics']['mean_density']}%")
    print(f"Mean SDS (Irony): {stats['reinman_metrics']['mean_sds']}")
    print(f"Mean TVT (Complexity): {stats['reinman_metrics']['mean_tvt']}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory with *_sovereign.json files")
    parser.add_argument("--output", default="analytics_output", help="Output directory")
    args = parser.parse_args()

    main(args.results_dir, args.output)
