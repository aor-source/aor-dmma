#!/usr/bin/env python3
"""
Human vs AI Corpus Comparison Visualization
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Style
plt.style.use('dark_background')

def load_corpus(path):
    """Load all sovereign JSONs from a directory"""
    data = []
    for f in glob.glob(f"{path}/*_sovereign.json"):
        with open(f) as file:
            data.append(json.load(file))
    return data

# Load both corpora
human = load_corpus('output/corpus_analysis')
ai = load_corpus('output/ai_corpus_analysis')

# Extract metrics
def get_metrics(corpus):
    return {
        'aave': [d['cultural_metrics']['aave_density'] * 100 for d in corpus],
        'sds': [d['reinman_metrics']['sds_score'] for d in corpus],
        'tvt': [d['reinman_metrics']['tvt_score'] for d in corpus],
        'valence': [d['reinman_metrics']['lyric_valence'] for d in corpus],
        'arousal': [d['reinman_metrics']['audio_arousal'] for d in corpus]
    }

h = get_metrics(human)
a = get_metrics(ai)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='black')
fig.suptitle('HUMAN vs AI-GENERATED HIP-HOP: AoR Analysis', color='white', fontsize=16, fontweight='bold')

colors = {'human': '#00d4ff', 'ai': '#ff006e'}

# 1. AAVE Density Comparison (Box plot)
ax1 = axes[0, 0]
ax1.set_facecolor('black')
bp = ax1.boxplot([h['aave'], a['aave']], labels=['Human', 'AI'], patch_artist=True)
bp['boxes'][0].set_facecolor(colors['human'])
bp['boxes'][1].set_facecolor(colors['ai'])
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='white')
ax1.set_ylabel('AAVE Density (%)', color='white')
ax1.set_title('AAVE Detection', color='white')
ax1.tick_params(colors='white')

# 2. SDS (Irony) Comparison
ax2 = axes[0, 1]
ax2.set_facecolor('black')
bp2 = ax2.boxplot([h['sds'], a['sds']], labels=['Human', 'AI'], patch_artist=True)
bp2['boxes'][0].set_facecolor(colors['human'])
bp2['boxes'][1].set_facecolor(colors['ai'])
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp2[element], color='white')
ax2.set_ylabel('SDS Score', color='white')
ax2.set_title('Semantic Dissonance (Irony)', color='white')
ax2.tick_params(colors='white')

# 3. TVT (Complexity) Comparison
ax3 = axes[0, 2]
ax3.set_facecolor('black')
bp3 = ax3.boxplot([h['tvt'], a['tvt']], labels=['Human', 'AI'], patch_artist=True)
bp3['boxes'][0].set_facecolor(colors['human'])
bp3['boxes'][1].set_facecolor(colors['ai'])
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp3[element], color='white')
ax3.set_ylabel('TVT Score', color='white')
ax3.set_title('Topological Complexity', color='white')
ax3.tick_params(colors='white')

# 4. Reinman Plane Scatter (both corpora)
ax4 = axes[1, 0]
ax4.set_facecolor('black')
ax4.scatter(h['sds'], h['tvt'], c=colors['human'], s=100, alpha=0.7, label='Human', edgecolors='white', linewidth=0.5)
ax4.scatter(a['sds'], a['tvt'], c=colors['ai'], s=100, alpha=0.7, label='AI', edgecolors='white', linewidth=0.5)
ax4.set_xlabel('SDS (Irony)', color='white')
ax4.set_ylabel('TVT (Complexity)', color='white')
ax4.set_title('Reinman Plane: Human vs AI', color='white')
ax4.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax4.tick_params(colors='white')

# 5. Sentiment-Arousal Matrix
ax5 = axes[1, 1]
ax5.set_facecolor('black')
ax5.scatter(h['valence'], h['arousal'], c=colors['human'], s=100, alpha=0.7, label='Human', edgecolors='white', linewidth=0.5)
ax5.scatter(a['valence'], a['arousal'], c=colors['ai'], s=100, alpha=0.7, label='AI', edgecolors='white', linewidth=0.5)
ax5.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
ax5.axvline(x=0, color='white', linestyle='--', alpha=0.3)
ax5.set_xlabel('Lyric Valence', color='white')
ax5.set_ylabel('Audio Arousal', color='white')
ax5.set_title('Sentiment-Audio Alignment', color='white')
ax5.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax5.tick_params(colors='white')

# 6. Summary Bar Chart
ax6 = axes[1, 2]
ax6.set_facecolor('black')

metrics = ['AAVE\nDensity', 'SDS\n(Irony)', 'TVT\n(Complexity)']
human_means = [np.mean(h['aave']), np.mean(h['sds']), np.mean(h['tvt'])/100]  # Scale TVT
ai_means = [np.mean(a['aave']), np.mean(a['sds']), np.mean(a['tvt'])/100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, human_means, width, label='Human', color=colors['human'], alpha=0.8)
bars2 = ax6.bar(x + width/2, ai_means, width, label='AI', color=colors['ai'], alpha=0.8)

ax6.set_ylabel('Score (normalized)', color='white')
ax6.set_title('Mean Metrics Comparison', color='white')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, color='white')
ax6.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax6.tick_params(colors='white')

# Add percentage difference annotations
for i, (hm, am) in enumerate(zip(human_means, ai_means)):
    diff = ((hm - am) / am) * 100 if am > 0 else 0
    ax6.annotate(f'+{diff:.0f}%' if diff > 0 else f'{diff:.0f}%',
                xy=(i, max(hm, am) + 0.1),
                ha='center', color='white', fontsize=9)

plt.tight_layout()
plt.savefig('output/human_vs_ai_comparison.png', dpi=200, facecolor='black', bbox_inches='tight')
plt.close()

print("✅ Generated: output/human_vs_ai_comparison.png")

# Print summary
print("\n" + "="*60)
print("HUMAN vs AI SUMMARY")
print("="*60)
print(f"{'Metric':<20} {'Human':>12} {'AI':>12} {'Diff':>12}")
print("-"*60)
print(f"{'AAVE Density':<20} {np.mean(h['aave']):>11.2f}% {np.mean(a['aave']):>11.2f}% {((np.mean(h['aave'])-np.mean(a['aave']))/np.mean(a['aave'])*100):>+11.1f}%")
print(f"{'SDS (Irony)':<20} {np.mean(h['sds']):>12.4f} {np.mean(a['sds']):>12.4f} {((np.mean(h['sds'])-np.mean(a['sds']))/np.mean(a['sds'])*100):>+11.1f}%")
print(f"{'TVT (Complexity)':<20} {np.mean(h['tvt']):>12.2f} {np.mean(a['tvt']):>12.2f} {((np.mean(h['tvt'])-np.mean(a['tvt']))/np.mean(a['tvt'])*100):>+11.1f}%")
print("="*60)
