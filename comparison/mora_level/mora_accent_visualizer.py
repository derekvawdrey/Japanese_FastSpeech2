#!/usr/bin/env python3
"""
Mora-Level Pitch Accent Visualization Tool

Creates visualizations comparing pitch accent patterns across TTS models.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np

# Configure matplotlib for Japanese font support
def setup_japanese_font():
    """Setup matplotlib to display Japanese characters correctly"""
    import platform
    
    # Try different fonts based on OS
    if platform.system() == 'Darwin':  # macOS
        fonts = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'AppleGothic', 'Yu Gothic']
    elif platform.system() == 'Windows':
        fonts = ['Yu Gothic', 'MS Gothic', 'Meiryo']
    else:  # Linux
        fonts = ['Noto Sans CJK JP', 'IPAGothic', 'IPAPGothic', 'TakaoPGothic', 'VL PGothic']
    
    # Add DejaVu Sans as fallback
    fonts.append('DejaVu Sans')
    
    # Try to find an available font
    available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
    
    for font in fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"Using font: {font}")
            return
    
    # If no Japanese font found, try setting font.sans-serif
    plt.rcParams['font.sans-serif'] = fonts + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Warning: No dedicated Japanese font found. Trying fallback fonts.")

# Setup Japanese font on import
setup_japanese_font()

# Color scheme
COLORS = {
    'FastSpeech2_Mine': '#2ecc71',      # Green
    'ElevenLabs_Eiko': '#3498db',        # Blue
    'Google_Translate': '#e74c3c',       # Red
    'Ondoku': '#9b59b6',                 # Purple
    'Reference': '#34495e',              # Dark gray
}

ACCENT_TYPE_COLORS = {
    'HEIBAN': '#3498db',      # Blue
    'ATAMADAKA': '#e74c3c',   # Red
    'NAKADAKA': '#f39c12',    # Orange
    'ODAKA': '#9b59b6',       # Purple
}


def load_results(json_path: Path) -> Dict:
    """Load comparison results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_overall_accuracy(results: Dict, output_dir: Path) -> None:
    """Create bar chart of overall accuracy metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(results.keys())
    
    # Accent Position Accuracy
    ax1 = axes[0]
    values = [results[m]['accent_position_accuracy'] * 100 for m in models]
    colors = [COLORS.get(m, '#95a5a6') for m in models]
    bars = ax1.bar(models, values, color=colors)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accent Position Accuracy')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Accent Type Accuracy
    ax2 = axes[1]
    values = [results[m]['accent_type_accuracy'] * 100 for m in models]
    bars = ax2.bar(models, values, color=colors)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accent Type Accuracy')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Mora-Level Accuracy
    ax3 = axes[2]
    values = [results[m]['average_mora_accuracy'] * 100 for m in models]
    bars = ax3.bar(models, values, color=colors)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Average Mora H/L Accuracy')
    ax3.set_ylim(0, 100)
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mora_accent_overall_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mora_accent_overall_accuracy.png'}")


def plot_accuracy_by_accent_type(results: Dict, output_dir: Path) -> None:
    """Create grouped bar chart of accuracy by accent type"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    
    # Collect all accent types
    all_types = set()
    for m in models:
        all_types.update(results[m].get('accuracy_by_type', {}).keys())
    all_types = sorted(all_types)
    
    if not all_types:
        print("No accent type breakdown available")
        return
    
    x = np.arange(len(all_types))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        accuracy_by_type = results[model].get('accuracy_by_type', {})
        values = [accuracy_by_type.get(t, 0) * 100 for t in all_types]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, 
                     color=COLORS.get(model, '#95a5a6'))
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Accent Type')
    ax.set_xticks(x)
    ax.set_xticklabels(all_types)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mora_accent_by_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mora_accent_by_type.png'}")


def plot_pattern_comparison_heatmap(results: Dict, output_dir: Path) -> None:
    """Create heatmap showing pattern match rates per sentence"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    models = list(results.keys())
    
    # Get all sentences
    sentences = []
    for model in models:
        for r in results[model]['results']:
            if r['sentence'] not in sentences:
                sentences.append(r['sentence'])
    
    # Build accuracy matrix
    matrix = []
    for sentence in sentences:
        row = []
        for model in models:
            # Find result for this sentence
            result = next(
                (r for r in results[model]['results'] if r['sentence'] == sentence),
                None
            )
            if result and result['success']:
                row.append(result['mora_accuracy'] * 100)
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mora Accuracy (%)', rotation=-90, va="bottom")
    
    # Labels
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(sentences)))
    ax.set_xticklabels(models)
    ax.set_yticklabels([s[:30] + '...' if len(s) > 30 else s for s in sentences])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(sentences)):
        for j in range(len(models)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.0f}',
                              ha="center", va="center", 
                              color="white" if val < 50 else "black",
                              fontsize=8)
    
    ax.set_title('Mora Accuracy by Sentence and Model')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mora_accent_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mora_accent_heatmap.png'}")


def plot_pattern_examples(results: Dict, output_dir: Path, num_examples: int = 5) -> None:
    """Create visualization showing example pattern comparisons"""
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples))
    
    if num_examples == 1:
        axes = [axes]
    
    models = list(results.keys())
    
    # Get sentences with varied results
    sentences_with_results = []
    for model in models:
        for r in results[model]['results']:
            if r['success'] and r['sentence'] not in [s[0] for s in sentences_with_results]:
                sentences_with_results.append((r['sentence'], r))
    
    for idx, ax in enumerate(axes):
        if idx >= len(sentences_with_results):
            ax.axis('off')
            continue
        
        sentence, _ = sentences_with_results[idx]
        
        # Get all model results for this sentence
        patterns = []
        for model in models:
            result = next(
                (r for r in results[model]['results'] if r['sentence'] == sentence),
                None
            )
            if result and result['success']:
                patterns.append({
                    'model': model,
                    'pattern': result['test_pattern'],
                    'correct': result['is_correct'],
                    'mora_acc': result['mora_accuracy']
                })
        
        # Get reference pattern
        ref_pattern = None
        for model in models:
            result = next(
                (r for r in results[model]['results'] if r['sentence'] == sentence),
                None
            )
            if result and result['success'] and result['reference_pattern']:
                ref_pattern = result['reference_pattern']
                break
        
        # Plot patterns
        y_positions = list(range(len(patterns) + 1))
        
        # Reference first
        if ref_pattern:
            _plot_pattern_line(ax, ref_pattern, 0, 'Reference', COLORS['Reference'], is_reference=True)
        
        # Then each model
        for i, p in enumerate(patterns):
            color = COLORS.get(p['model'], '#95a5a6')
            _plot_pattern_line(ax, p['pattern'], i + 1, 
                             f"{p['model']} ({'✓' if p['correct'] else '✗'} {p['mora_acc']:.0%})",
                             color)
        
        ax.set_ylim(-0.5, len(patterns) + 1)
        ax.set_title(f"Sentence: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
        ax.set_xlabel('Mora Position')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mora_accent_pattern_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mora_accent_pattern_examples.png'}")


def _plot_pattern_line(ax, pattern: str, y_pos: int, label: str, color: str, is_reference: bool = False):
    """Plot a single H/L pattern line"""
    x = list(range(len(pattern)))
    y = [y_pos + 0.3 if p == 'H' else y_pos - 0.1 for p in pattern]
    
    linewidth = 3 if is_reference else 2
    linestyle = '-' if is_reference else '--'
    
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, marker='o', markersize=8)
    
    # Add H/L labels
    for i, (xi, yi, p) in enumerate(zip(x, y, pattern)):
        ax.text(xi, yi + 0.15, p, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add model label
    ax.text(-0.5, y_pos + 0.1, label, ha='right', va='center', fontsize=9)


def plot_success_failure_breakdown(results: Dict, output_dir: Path) -> None:
    """Create pie charts showing success/failure breakdown"""
    models = list(results.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, model in zip(axes, models):
        stats = results[model]
        
        # Count correct vs incorrect vs failed
        correct = sum(1 for r in stats['results'] if r['success'] and r['is_correct'])
        incorrect = sum(1 for r in stats['results'] if r['success'] and not r['is_correct'])
        failed = stats['failed_analyses']
        
        values = [correct, incorrect, failed]
        labels = ['Correct', 'Incorrect', 'Failed']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        # Remove zero values
        non_zero = [(v, l, c) for v, l, c in zip(values, labels, colors) if v > 0]
        if non_zero:
            values, labels, colors = zip(*non_zero)
        
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(model)
    
    plt.suptitle('Accent Pattern Correctness Breakdown', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'mora_accent_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'mora_accent_breakdown.png'}")


def create_all_visualizations(json_path: Path, output_dir: Path) -> None:
    """Generate all visualizations from results JSON"""
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    
    print(f"Creating visualizations in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_overall_accuracy(results, output_dir)
    plot_accuracy_by_accent_type(results, output_dir)
    plot_pattern_comparison_heatmap(results, output_dir)
    plot_pattern_examples(results, output_dir)
    plot_success_failure_breakdown(results, output_dir)
    
    print("\nAll visualizations created!")


def main():
    """Main entry point"""
    # Default paths
    comparison_dir = Path(__file__).parent
    json_path = comparison_dir / "mora_accent_comparison_results.json"
    output_dir = comparison_dir / "mora_accent_graphs"
    
    # Check for command line args
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    if not json_path.exists():
        print(f"Error: Results file not found: {json_path}")
        print("\nPlease run mora_accent_comparison.py first to generate results.")
        sys.exit(1)
    
    create_all_visualizations(json_path, output_dir)


if __name__ == "__main__":
    main()
