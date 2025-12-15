#!/usr/bin/env python
"""
Generate visualization graphs for pitch accent comparison results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# Set style first
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

# Configure matplotlib to display Japanese characters AFTER setting style
# Try to find a Japanese-compatible font
def setup_japanese_font():
    """Find and set a Japanese font for matplotlib."""
    # List of potential Japanese fonts (macOS, Linux, Windows)
    japanese_fonts = [
        'Hiragino Sans',           # macOS
        'Hiragino Kaku Gothic Pro', # macOS
        'Hiragino Maru Gothic Pro', # macOS
        'Yu Gothic',                # Windows
        'MS Gothic',                # Windows
        'Meiryo',                   # Windows
        'IPAGothic',                # Linux
        'IPAPGothic',               # Linux
        'Noto Sans CJK JP',         # Cross-platform
        'Noto Sans JP',             # Cross-platform
        'TakaoPGothic',             # Linux
    ]
    
    # Get list of available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Try to find a Japanese font
    for font in japanese_fonts:
        if font in available_fonts:
            # Set font in multiple ways to ensure it takes effect
            plt.rcParams['font.family'] = font
            plt.rcParams['font.sans-serif'] = [font]
            print(f"✓ Using font: {font}")
            return font
    
    # Fallback: use sans-serif with negative unicode settings
    print("⚠ Warning: No Japanese font found. Japanese characters may not display correctly.")
    print("Available fonts include:", ", ".join(sorted(set(available_fonts))[:10]), "...")
    return None

# Setup Japanese font support
japanese_font = setup_japanese_font()

# Prevent negative sign display issues
plt.rcParams['axes.unicode_minus'] = False

# Load results
with open('comparison/comparison_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Prepare data
sentences = []
mine_scores = []
ondoku_scores = []
google_scores = []
eleven_labs_scores = []

for sentence, comparisons in results.items():
    if comparisons:  # Skip sentences without comparisons
        sentences.append(sentence[:20] + '...' if len(sentence) > 20 else sentence)
        mine_scores.append(comparisons.get('mine', {}).get('dtw_distance', None))
        ondoku_scores.append(comparisons.get('ondoku_wav', {}).get('dtw_distance', None))
        google_scores.append(comparisons.get('google_translate_wav', {}).get('dtw_distance', None))
        eleven_labs_scores.append(comparisons.get('eleven_labs_eiko_wav', {}).get('dtw_distance', None))

# Calculate averages (excluding None values)
mine_avg = np.mean([s for s in mine_scores if s is not None])
ondoku_avg = np.mean([s for s in ondoku_scores if s is not None])
google_avg = np.mean([s for s in google_scores if s is not None])
eleven_labs_avg = np.mean([s for s in eleven_labs_scores if s is not None])

# Graph 1: Overall Average Scores
fig, ax = plt.subplots(figsize=(12, 6))
sources = ['Your Recordings\n(mine)', 'Ondoku', 'Google Translate', 'Eleven Labs']
averages = [mine_avg, ondoku_avg, google_avg, eleven_labs_avg]
bars = ax.bar(sources, averages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
# Sort to determine ranks
sorted_indices = sorted(range(len(averages)), key=lambda i: averages[i])
rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
rank_emojis = {1: '🥇 1st', 2: '🥈 2nd', 3: '🥉 3rd', 4: '4th'}

for i, (bar, avg) in enumerate(zip(bars, averages)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{avg:.3f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add rank
    rank = rank_map[i]
    ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
            rank_emojis[rank],
            ha='center', va='center', fontsize=16, fontweight='bold', color='white')

ax.set_ylabel('Average DTW Distance', fontsize=12, fontweight='bold')
ax.set_title('Overall Pitch Accent Match to Sample\n(Lower is Better)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(averages) * 1.2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison/graph_overall_averages.png', dpi=300, bbox_inches='tight')
print("✓ Created: graph_overall_averages.png")

# Graph 2: Per-Sentence Comparison
fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(sentences))
width = 0.2

bars1 = ax.bar(x - 1.5*width, mine_scores, width, label='Your Recordings', color=colors[0], alpha=0.8)
bars2 = ax.bar(x - 0.5*width, ondoku_scores, width, label='Ondoku', color=colors[1], alpha=0.8)
bars3 = ax.bar(x + 0.5*width, google_scores, width, label='Google Translate', color=colors[2], alpha=0.8)
bars4 = ax.bar(x + 1.5*width, eleven_labs_scores, width, label='Eleven Labs', color=colors[3], alpha=0.8)

# Highlight best scores per sentence
for i in range(len(sentences)):
    scores = [mine_scores[i], ondoku_scores[i], google_scores[i], eleven_labs_scores[i]]
    if all(s is not None for s in scores):
        min_score = min(scores)
        if mine_scores[i] == min_score:
            bars1[i].set_edgecolor('gold')
            bars1[i].set_linewidth(3)
        elif ondoku_scores[i] == min_score:
            bars2[i].set_edgecolor('gold')
            bars2[i].set_linewidth(3)
        elif google_scores[i] == min_score:
            bars3[i].set_edgecolor('gold')
            bars3[i].set_linewidth(3)
        elif eleven_labs_scores[i] == min_score:
            bars4[i].set_edgecolor('gold')
            bars4[i].set_linewidth(3)

ax.set_xlabel('Sentences', fontsize=12, fontweight='bold')
ax.set_ylabel('DTW Distance', fontsize=12, fontweight='bold')
ax.set_title('Pitch Accent Comparison by Sentence\n(Gold border = Best for that sentence)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
# Set Japanese text labels with explicit font
if japanese_font:
    ax.set_xticklabels(sentences, rotation=45, ha='right', fontsize=8, fontfamily=japanese_font)
else:
    ax.set_xticklabels(sentences, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison/graph_by_sentence.png', dpi=300, bbox_inches='tight')
print("✓ Created: graph_by_sentence.png")

# Graph 3: Win Count (how many sentences each source won)
fig, ax = plt.subplots(figsize=(12, 6))

mine_wins = 0
ondoku_wins = 0
google_wins = 0
eleven_labs_wins = 0

for i in range(len(sentences)):
    scores = [mine_scores[i], ondoku_scores[i], google_scores[i], eleven_labs_scores[i]]
    if all(s is not None for s in scores):
        min_score = min(scores)
        if mine_scores[i] == min_score:
            mine_wins += 1
        elif ondoku_scores[i] == min_score:
            ondoku_wins += 1
        elif google_scores[i] == min_score:
            google_wins += 1
        elif eleven_labs_scores[i] == min_score:
            eleven_labs_wins += 1

wins = [mine_wins, ondoku_wins, google_wins, eleven_labs_wins]
bars = ax.bar(sources, wins, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, win_count in zip(bars, wins):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{win_count}',
            ha='center', va='bottom', fontsize=20, fontweight='bold')

ax.set_ylabel('Number of Sentences Won', fontsize=12, fontweight='bold')
ax.set_title('Best Pitch Accent Match Count by Source\n(Number of sentences where each source had the lowest score)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(wins) * 1.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison/graph_win_count.png', dpi=300, bbox_inches='tight')
print("✓ Created: graph_win_count.png")

# Graph 4: Score Distribution (Box plot)
fig, ax = plt.subplots(figsize=(12, 6))

data_to_plot = [
    [s for s in mine_scores if s is not None],
    [s for s in ondoku_scores if s is not None],
    [s for s in google_scores if s is not None],
    [s for s in eleven_labs_scores if s is not None]
]

bp = ax.boxplot(data_to_plot, labels=sources, patch_artist=True,
                medianprops=dict(color='red', linewidth=2),
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('DTW Distance', fontsize=12, fontweight='bold')
ax.set_title('Score Distribution Across All Sentences\n(Box shows quartiles, red line is median)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison/graph_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Created: graph_distribution.png")

# Graph 5: Heatmap of scores
fig, ax = plt.subplots(figsize=(14, 10))

# Create matrix for heatmap
score_matrix = []
for i in range(len(sentences)):
    row = [mine_scores[i] or 0, ondoku_scores[i] or 0, google_scores[i] or 0, eleven_labs_scores[i] or 0]
    score_matrix.append(row)

score_matrix = np.array(score_matrix)

im = ax.imshow(score_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=2.5)

# Set ticks
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(len(sentences)))
ax.set_xticklabels(['Your Recordings', 'Ondoku', 'Google Translate', 'Eleven Labs'], fontsize=11)
# Set Japanese text labels with explicit font
if japanese_font:
    ax.set_yticklabels(sentences, fontsize=9, fontfamily=japanese_font)
else:
    ax.set_yticklabels(sentences, fontsize=9)

# Rotate the tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(sentences)):
    for j in range(4):
        if score_matrix[i, j] > 0:
            text = ax.text(j, i, f'{score_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')

ax.set_title('Heatmap of DTW Distances\n(Green = Better, Red = Worse)', 
             fontsize=16, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('DTW Distance', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison/graph_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Created: graph_heatmap.png")

print("\n✅ All graphs created successfully!")
print("\nGenerated files:")
print("  - comparison/graph_overall_averages.png")
print("  - comparison/graph_by_sentence.png")
print("  - comparison/graph_win_count.png")
print("  - comparison/graph_distribution.png")
print("  - comparison/graph_heatmap.png")

