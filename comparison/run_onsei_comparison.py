#!/usr/bin/env python
"""
Use actual onsei tool to compare pitch accent across different recordings.
"""

import sys
import os
from pathlib import Path
import json
from collections import defaultdict

# Add onsei to path
sys.path.insert(0, str(Path(__file__).parent / 'onsei_tool'))

from onsei.speech_record import SpeechRecord
import tempfile
import contextlib

def load_sentences(sentences_file):
    """Load sentences from text file."""
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

def compare_recordings(sample_path, test_path):
    """
    Compare test recording against sample using onsei.
    Returns distance score (lower = better match).
    """
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                sample_rec = SpeechRecord(sample_path, None, name="Sample")
                test_rec = SpeechRecord(test_path, None, name="Test")
                
                # Align and compare
                test_rec.align_with(sample_rec, method="phonemes")
                distance = test_rec.compare_pitch()
                
                return float(distance) if distance is not None else None
    except Exception as e:
        print(f"    Error: {str(e)}")
        return None

def main():
    print("=" * 80)
    print("ONSEI PITCH ACCENT COMPARISON TOOL")
    print("Using actual onsei: https://github.com/itsupera/onsei")
    print("=" * 80)
    print()
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    sample_dir = base_dir / "voice_splits" / "sample"
    sentences_file = base_dir / "voice_splits" / "sample_text.txt"
    
    comparison_dirs = [
        base_dir / "voice_splits" / "mine",
        base_dir / "voice_splits" / "ondoku",
        base_dir / "voice_splits" / "google translate"
    ]
    
    # Load sentences
    sentences = load_sentences(sentences_file)
    
    # Results storage
    results = {}
    
    # Compare each sentence
    for sentence in sentences:
        print(f"\nProcessing: {sentence}")
        results[sentence] = {}
        
        # Find sample file
        sample_files = list(sample_dir.glob(f"{sentence}.*"))
        if not sample_files:
            print(f"  Warning: No sample file found")
            continue
        
        sample_path = str(sample_files[0])
        print(f"  Sample: {sample_path}")
        
        # Compare against each directory
        for comp_dir in comparison_dirs:
            dir_name = comp_dir.name
            
            # Find matching file
            test_file = None
            for ext in ['.wav', '.mp3', '.flac']:
                test_files = list(comp_dir.glob(f"{sentence}{ext}"))
                if test_files:
                    test_file = str(test_files[0])
                    break
            
            if not test_file:
                print(f"  No file found in {dir_name}")
                continue
            
            print(f"  Comparing with {dir_name}...", end=" ")
            
            distance = compare_recordings(sample_path, test_file)
            
            if distance is not None:
                results[sentence][dir_name] = {
                    'distance': distance,
                    'file': test_file
                }
                print(f"Distance: {distance:.3f}")
            else:
                print("Failed")
    
    # Save results
    output_path = Path(__file__).parent / 'onsei_comparison_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # Calculate average scores per directory
    dir_scores = defaultdict(list)
    for sentence, comparisons in results.items():
        for dir_name, metrics in comparisons.items():
            if 'distance' in metrics:
                dir_scores[dir_name].append(metrics['distance'])
    
    # Sort by average score
    dir_averages = {
        dir_name: sum(scores) / len(scores)
        for dir_name, scores in dir_scores.items()
        if scores
    }
    
    sorted_dirs = sorted(dir_averages.items(), key=lambda x: x[1])
    
    print("OVERALL RANKINGS (lower distance = better pitch accent match):")
    print("-" * 80)
    for rank, (dir_name, avg_distance) in enumerate(sorted_dirs, 1):
        num_samples = len(dir_scores[dir_name])
        print(f"{rank}. {dir_name:20s} - Average Distance: {avg_distance:.3f} "
              f"({num_samples} sentences)")
    
    print()
    print("=" * 80)
    print("DETAILED RESULTS BY SENTENCE")
    print("=" * 80)
    
    for sentence, comparisons in results.items():
        if not comparisons:
            continue
            
        print(f"\n{sentence}")
        print("-" * len(sentence))
        
        # Sort by distance
        sorted_comps = sorted(comparisons.items(), key=lambda x: x[1].get('distance', float('inf')))
        
        for rank, (dir_name, metrics) in enumerate(sorted_comps, 1):
            if 'distance' in metrics:
                print(f"  {rank}. {dir_name:15s} - Distance: {metrics['distance']:.3f}")
    
    print()
    print("=" * 80)
    print("NOTE: Distances are computed using onsei's DTW-based pitch comparison.")
    print("Lower distances indicate closer pitch accent match to the sample.")
    print("=" * 80)
    
    # Save summary
    summary_path = Path(__file__).parent / 'onsei_comparison_summary.txt'
    # (save summary to file - similar output as printed above)

if __name__ == "__main__":
    main()
