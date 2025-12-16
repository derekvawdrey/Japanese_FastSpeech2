#!/usr/bin/env python3
"""
Visualization tool for pitch accent comparison results

Generates charts and graphs from the comparison results JSON file.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Rectangle

# Configure matplotlib to support Japanese characters
# Try to find a Japanese-capable font
def setup_japanese_font():
    """Configure matplotlib to use a font that supports Japanese characters"""
    # List of fonts to try (in order of preference)
    japanese_fonts = [
        'Hiragino Sans',  # macOS default Japanese font
        'Hiragino Sans GB',
        'Arial Unicode MS',
        'Noto Sans CJK JP',
        'Yu Gothic',
        'Meiryo',
        'MS Gothic'
    ]
    
    # Try to find an available font
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    for font in japanese_fonts:
        if font in available_fonts:
            matplotlib.rcParams['font.family'] = font
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            print(f"Using font: {font}")
            return font
    
    # If no Japanese font found, try to use a fallback
    # On macOS, we can try to use the system default
    try:
        import platform
        if platform.system() == 'Darwin':  # macOS
            matplotlib.rcParams['font.family'] = 'Hiragino Sans'
            matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans'] + matplotlib.rcParams['font.sans-serif']
        else:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass
    
    # Ensure we have a fallback for English characters
    matplotlib.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif'] + ['Arial', 'DejaVu Sans', 'sans-serif']
    
    print("Warning: No Japanese font found. Japanese characters may not display correctly.")
    return None

# Setup Japanese font support
setup_japanese_font()


def load_results(json_file: Path) -> Dict:
    """Load comparison results from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_overall_comparison(results: Dict, output_dir: Path):
    """Create bar chart comparing overall performance"""
    # Filter models with successful comparisons
    models = []
    avg_distances = []
    avg_scores = []
    
    for model_name, stats in results.items():
        if stats['average_distance'] is not None:
            models.append(model_name)
            avg_distances.append(stats['average_distance'])
            avg_scores.append(stats['average_score'])
    
    # Sort by distance (lower is better)
    sorted_indices = np.argsort(avg_distances)
    models = [models[i] for i in sorted_indices]
    avg_distances = [avg_distances[i] for i in sorted_indices]
    avg_scores = [avg_scores[i] for i in sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Average Distance (lower is better)
    colors = ['#2ecc71' if i == 0 else '#3498db' if i == 1 else '#95a5a6' 
              for i in range(len(models))]
    
    bars1 = ax1.barh(models, avg_distances, color=colors)
    ax1.set_xlabel('Average Distance (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Pitch Accent Accuracy - Distance Metric', fontsize=14, fontweight='bold')
    ax1.set_yticklabels(models, fontsize=10)
    ax1.invert_xaxis()  # Lower values on the right
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, avg_distances)):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}',
                va='center', ha='right' if i == 0 else 'left', fontsize=10, fontweight='bold')
    
    # Plot 2: Average Score (higher is better)
    bars2 = ax2.barh(models, avg_scores, color=colors)
    ax2.set_xlabel('Average Score % (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('Pitch Accent Accuracy - Score Metric', fontsize=14, fontweight='bold')
    ax2.set_yticklabels(models, fontsize=10)
    ax2.set_xlim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars2, avg_scores):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'pitch_accent_overall_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_per_sentence_heatmap(results: Dict, output_dir: Path):
    """Create heatmap showing performance per sentence"""
    # Get all sentences from the first model
    first_model = list(results.values())[0]
    sentences = [r['sentence'] for r in first_model['results']]
    
    # Get models with successful comparisons
    models = []
    distance_matrix = []
    
    for model_name, stats in results.items():
        if stats['average_distance'] is not None:
            models.append(model_name)
            distances = []
            for result in stats['results']:
                if result['success'] and result['mean_distance'] is not None:
                    distances.append(result['mean_distance'])
                else:
                    distances.append(np.nan)
            distance_matrix.append(distances)
    
    # Sort models by average distance
    avg_distances = [np.nanmean(d) for d in distance_matrix]
    sorted_indices = np.argsort(avg_distances)
    models = [models[i] for i in sorted_indices]
    distance_matrix = [distance_matrix[i] for i in sorted_indices]
    
    # Create heatmap with better sizing
    # Calculate appropriate figure size: more height per model, adequate width for sentences
    fig_height = max(8, len(models) * 0.8 + 3)  # At least 0.8 units per model, minimum 8
    fig_width = max(18, len(sentences) * 1.2 + 2)  # At least 1.2 units per sentence, minimum 18
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Convert to numpy array
    data = np.array(distance_matrix)
    
    # Create custom colormap (green = good/low distance, red = bad/high distance)
    # Use aspect='equal' or a fixed aspect ratio to prevent squishing
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(sentences)))
    ax.set_yticks(np.arange(len(models)))
    
    # Truncate long sentences for display
    sentence_labels = [s[:50] + '...' if len(s) > 50 else s for s in sentences]
    ax.set_xticklabels(sentence_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(models, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (lower is better)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add value annotations
    for i in range(len(models)):
        for j in range(len(sentences)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('Pitch Accent Distance per Sentence (Heatmap)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout with padding to prevent squishing
    plt.tight_layout(pad=2.0)
    output_file = output_dir / 'pitch_accent_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Saved: {output_file}")
    plt.close()


def plot_distance_distribution(results: Dict, output_dir: Path):
    """Create box plot showing distance distribution for each model"""
    models = []
    all_distances = []
    
    for model_name, stats in results.items():
        if stats['average_distance'] is not None:
            models.append(model_name)
            distances = [
                r['mean_distance'] for r in stats['results']
                if r['success'] and r['mean_distance'] is not None
            ]
            all_distances.append(distances)
    
    # Sort by median distance
    medians = [np.median(d) for d in all_distances]
    sorted_indices = np.argsort(medians)
    models = [models[i] for i in sorted_indices]
    all_distances = [all_distances[i] for i in sorted_indices]
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    box_parts = ax.boxplot(all_distances, labels=models, vert=False, patch_artist=True,
                           showmeans=True, meanline=True)
    
    # Color boxes
    colors = ['#2ecc71', '#3498db', '#95a5a6', '#95a5a6']
    for patch, color in zip(box_parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Distance (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Pitch Accent Distance Across All Sentences', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    ax.legend([box_parts['medians'][0], box_parts['means'][0]], 
             ['Median', 'Mean'], loc='upper right')
    
    plt.tight_layout()
    output_file = output_dir / 'pitch_accent_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_success_rate(results: Dict, output_dir: Path):
    """Create bar chart showing success rate for each model"""
    models = []
    success_rates = []
    successful_counts = []
    total_counts = []
    
    for model_name, stats in results.items():
        models.append(model_name)
        total = stats['total_sentences']
        successful = stats['successful_comparisons']
        success_rate = (successful / total * 100) if total > 0 else 0
        
        success_rates.append(success_rate)
        successful_counts.append(successful)
        total_counts.append(total)
    
    # Sort by success rate
    sorted_indices = np.argsort(success_rates)[::-1]
    models = [models[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    successful_counts = [successful_counts[i] for i in sorted_indices]
    total_counts = [total_counts[i] for i in sorted_indices]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71' if sr == 100 else '#f39c12' if sr >= 80 else '#e74c3c' 
              for sr in success_rates]
    bars = ax.barh(models, success_rates, color=colors)
    
    ax.set_xlabel('Success Rate %', fontsize=12, fontweight='bold')
    ax.set_title('Comparison Success Rate by Model', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    
    # Add value labels
    for bar, val, successful, total in zip(bars, success_rates, successful_counts, total_counts):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, 
               f'{val:.1f}% ({successful}/{total})',
               va='center', ha='left', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'pitch_accent_success_rate.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_best_worst_sentences(results: Dict, output_dir: Path):
    """Identify and visualize best and worst performing sentences"""
    # Calculate variance for each sentence across models
    first_model = list(results.values())[0]
    sentences = [r['sentence'] for r in first_model['results']]
    
    sentence_stats = []
    for i, sentence in enumerate(sentences):
        distances = []
        for model_name, stats in results.items():
            if i < len(stats['results']):
                result = stats['results'][i]
                if result['success'] and result['mean_distance'] is not None:
                    distances.append(result['mean_distance'])
        
        if len(distances) > 0:
            sentence_stats.append({
                'sentence': sentence,
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'range': np.max(distances) - np.min(distances)
            })
    
    # Sort by range (highest variance = most discriminative)
    sentence_stats.sort(key=lambda x: x['range'], reverse=True)
    
    # Get top 5 most discriminative and top 5 easiest
    most_discriminative = sentence_stats[:5]
    easiest = sorted(sentence_stats, key=lambda x: x['mean'])[:5]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot most discriminative sentences
    sentences_disc = [s['sentence'][:50] + '...' if len(s['sentence']) > 50 else s['sentence'] 
                     for s in most_discriminative]
    means_disc = [s['mean'] for s in most_discriminative]
    ranges_disc = [s['range'] for s in most_discriminative]
    
    y_pos = np.arange(len(sentences_disc))
    bars1 = ax1.barh(y_pos, means_disc, xerr=[[0]*len(ranges_disc), ranges_disc], 
                     color='#e74c3c', alpha=0.7, capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sentences_disc)
    ax1.set_xlabel('Mean Distance ± Range', fontsize=11, fontweight='bold')
    ax1.set_title('Most Discriminative Sentences (highest variance across models)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot easiest sentences
    sentences_easy = [s['sentence'][:50] + '...' if len(s['sentence']) > 50 else s['sentence'] 
                     for s in easiest]
    means_easy = [s['mean'] for s in easiest]
    ranges_easy = [s['range'] for s in easiest]
    
    y_pos = np.arange(len(sentences_easy))
    bars2 = ax2.barh(y_pos, means_easy, xerr=[[0]*len(ranges_easy), ranges_easy], 
                     color='#2ecc71', alpha=0.7, capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sentences_easy)
    ax2.set_xlabel('Mean Distance ± Range', fontsize=11, fontweight='bold')
    ax2.set_title('Easiest Sentences (lowest mean distance)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'pitch_accent_sentence_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main entry point"""
    # Find results file
    script_dir = Path(__file__).parent
    json_file = script_dir / "pitch_accent_comparison_results.json"
    
    if not json_file.exists():
        print(f"Error: Results file not found: {json_file}")
        print()
        print("Please run the comparison first:")
        print("  python3 compare_pitch_accent.py")
        sys.exit(1)
    
    print("=" * 80)
    print("Pitch Accent Comparison Visualizer")
    print("=" * 80)
    print()
    print(f"Loading results from: {json_file}")
    
    # Load results
    results = load_results(json_file)
    
    print(f"Found {len(results)} models")
    print()
    
    # Create visualizations
    output_dir = script_dir
    
    print("Generating visualizations...")
    print()
    
    plot_overall_comparison(results, output_dir)
    plot_per_sentence_heatmap(results, output_dir)
    plot_distance_distribution(results, output_dir)
    plot_success_rate(results, output_dir)
    plot_best_worst_sentences(results, output_dir)
    
    print()
    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - pitch_accent_overall_comparison.png")
    print("  - pitch_accent_heatmap.png")
    print("  - pitch_accent_distribution.png")
    print("  - pitch_accent_success_rate.png")
    print("  - pitch_accent_sentence_analysis.png")
    print()


if __name__ == "__main__":
    main()

