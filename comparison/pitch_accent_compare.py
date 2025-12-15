"""
Pitch Accent Comparison Tool
Inspired by the onsei project: https://github.com/itsupera/onsei

Compares pitch accent of different recordings against sample recordings
to determine which has the best pitch accent match.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy.ndimage import median_filter
import json
from typing import Dict, List, Tuple
import warnings
import subprocess
import tempfile
warnings.filterwarnings('ignore')


class PitchAccentComparer:
    def __init__(self, sample_dir: str, sentences_file: str):
        self.sample_dir = Path(sample_dir)
        self.sentences = self._load_sentences(sentences_file)
        
    def _load_sentences(self, sentences_file: str) -> List[str]:
        """Load sentences from text file."""
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences
    
    def extract_pitch(self, audio_path: str, sr: int = 22050) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch (F0) from audio file using librosa.
        Returns: (times, pitches)
        """
        # Load audio using soundfile directly (Python 3.13 compatibility)
        y, orig_sr = sf.read(audio_path)
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        # Resample if needed
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        else:
            sr = orig_sr
        
        # Extract pitch using pyin algorithm (more robust for voice)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sr
        )
        
        # Create time array
        times = librosa.times_like(f0, sr=sr)
        
        # Handle unvoiced regions (NaN values)
        # Interpolate over short gaps
        mask = ~np.isnan(f0)
        if np.sum(mask) > 0:
            f0_interp = np.interp(
                np.arange(len(f0)),
                np.where(mask)[0],
                f0[mask]
            )
        else:
            f0_interp = np.zeros_like(f0)
        
        # Apply median filter to smooth pitch contour
        f0_smooth = median_filter(f0_interp, size=3)
        
        return times, f0_smooth
    
    def normalize_pitch(self, pitch: np.ndarray) -> np.ndarray:
        """
        Normalize pitch to remove speaker-dependent variations.
        Converts to semitones relative to the median pitch.
        """
        # Remove zeros
        pitch_nonzero = pitch[pitch > 0]
        if len(pitch_nonzero) == 0:
            return pitch
        
        # Calculate median pitch
        median_pitch = np.median(pitch_nonzero)
        
        # Convert to semitones relative to median
        pitch_normalized = np.where(
            pitch > 0,
            12 * np.log2(pitch / median_pitch),
            0
        )
        
        return pitch_normalized
    
    def dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Calculate Dynamic Time Warping distance between two sequences.
        This aligns sequences of different lengths and computes similarity.
        """
        n, m = len(seq1), len(seq2)
        
        # Create cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill the matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Normalize by path length
        distance = dtw_matrix[n, m] / (n + m)
        
        return distance
    
    def compare_recordings(
        self,
        sample_path: str,
        test_path: str
    ) -> Dict[str, float]:
        """
        Compare test recording against sample recording.
        Returns dictionary with comparison metrics.
        """
        # Extract pitches
        _, sample_pitch = self.extract_pitch(sample_path)
        _, test_pitch = self.extract_pitch(test_path)
        
        # Normalize pitches
        sample_pitch_norm = self.normalize_pitch(sample_pitch)
        test_pitch_norm = self.normalize_pitch(test_pitch)
        
        # Calculate DTW distance
        dtw_dist = self.dtw_distance(sample_pitch_norm, test_pitch_norm)
        
        # Calculate other metrics
        # Pitch range similarity
        sample_range = np.ptp(sample_pitch_norm[sample_pitch_norm != 0])
        test_range = np.ptp(test_pitch_norm[test_pitch_norm != 0])
        range_diff = abs(sample_range - test_range)
        
        # Mean pitch difference
        sample_mean = np.mean(sample_pitch_norm[sample_pitch_norm != 0])
        test_mean = np.mean(test_pitch_norm[test_pitch_norm != 0])
        mean_diff = abs(sample_mean - test_mean)
        
        return {
            'dtw_distance': float(dtw_dist),
            'range_difference': float(range_diff),
            'mean_difference': float(mean_diff),
            'overall_score': float(dtw_dist)  # Lower is better
        }
    
    def compare_all_recordings(
        self,
        comparison_dirs: List[str],
        output_file: str = 'comparison_results.json'
    ) -> Dict:
        """
        Compare all recordings in specified directories against samples.
        Returns and saves comparison results.
        """
        results = {}
        
        for sentence in self.sentences:
            print(f"\nProcessing: {sentence}")
            results[sentence] = {}
            
            # Find sample file
            sample_files = list(self.sample_dir.glob(f"{sentence}.*"))
            if not sample_files:
                print(f"  Warning: No sample file found for '{sentence}'")
                continue
            
            sample_path = str(sample_files[0])
            print(f"  Sample: {sample_path}")
            
            # Compare against each comparison directory
            for comp_dir in comparison_dirs:
                comp_path = Path(comp_dir)
                dir_name = comp_path.name
                
                # Find matching file in comparison directory
                # Try multiple extensions
                test_file = None
                for ext in ['.wav', '.mp3', '.flac']:
                    test_files = list(comp_path.glob(f"{sentence}{ext}"))
                    if test_files:
                        test_file = str(test_files[0])
                        break
                
                if not test_file:
                    print(f"  Warning: No file found in {dir_name} for '{sentence}'")
                    continue
                
                print(f"  Comparing with {dir_name}...")
                
                try:
                    comparison = self.compare_recordings(sample_path, test_file)
                    results[sentence][dir_name] = comparison
                    print(f"    DTW Distance: {comparison['dtw_distance']:.3f}")
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    results[sentence][dir_name] = {
                        'error': str(e)
                    }
        
        # Save results
        output_path = Path('comparison') / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n\nResults saved to: {output_path}")
        
        return results
    
    def generate_summary(self, results: Dict) -> str:
        """Generate a summary report of the comparison results."""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("PITCH ACCENT COMPARISON SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # Calculate average scores for each directory
        dir_scores = {}
        
        for sentence, comparisons in results.items():
            for dir_name, metrics in comparisons.items():
                if 'error' not in metrics:
                    if dir_name not in dir_scores:
                        dir_scores[dir_name] = []
                    dir_scores[dir_name].append(metrics['overall_score'])
        
        # Calculate averages
        dir_averages = {
            dir_name: np.mean(scores)
            for dir_name, scores in dir_scores.items()
        }
        
        # Sort by average score (lower is better)
        sorted_dirs = sorted(dir_averages.items(), key=lambda x: x[1])
        
        summary_lines.append("OVERALL RANKINGS (lower score = better pitch accent match):")
        summary_lines.append("-" * 80)
        for rank, (dir_name, avg_score) in enumerate(sorted_dirs, 1):
            num_samples = len(dir_scores[dir_name])
            summary_lines.append(
                f"{rank}. {dir_name:20s} - Average Score: {avg_score:.3f} "
                f"({num_samples} sentences)"
            )
        
        summary_lines.append("")
        summary_lines.append("=" * 80)
        summary_lines.append("DETAILED RESULTS BY SENTENCE")
        summary_lines.append("=" * 80)
        
        for sentence, comparisons in results.items():
            summary_lines.append("")
            summary_lines.append(f"\n{sentence}")
            summary_lines.append("-" * len(sentence))
            
            # Sort comparisons by score
            valid_comps = {
                k: v for k, v in comparisons.items()
                if 'error' not in v
            }
            
            if valid_comps:
                sorted_comps = sorted(
                    valid_comps.items(),
                    key=lambda x: x[1]['overall_score']
                )
                
                for rank, (dir_name, metrics) in enumerate(sorted_comps, 1):
                    summary_lines.append(
                        f"  {rank}. {dir_name:15s} - Score: {metrics['overall_score']:.3f} "
                        f"(DTW: {metrics['dtw_distance']:.3f})"
                    )
        
        summary_lines.append("")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append("NOTE: Scores are based on Dynamic Time Warping (DTW) distance.")
        summary_lines.append("Lower scores indicate closer pitch accent match to the sample.")
        summary_lines.append("")
        
        return "\n".join(summary_lines)


def main():
    """Main function to run the comparison."""
    print("Pitch Accent Comparison Tool")
    print("Inspired by: https://github.com/itsupera/onsei")
    print("=" * 80)
    print()
    
    # Setup paths
    sample_dir = "voice_splits/sample"
    sentences_file = "voice_splits/sample_text.txt"
    
    comparison_dirs = [
        "voice_splits/mine",
        "voice_splits/ondoku_wav",
        "voice_splits/google_translate_wav"
    ]
    
    # Create comparison directory if it doesn't exist
    os.makedirs("comparison", exist_ok=True)
    
    # Initialize comparer
    comparer = PitchAccentComparer(sample_dir, sentences_file)
    
    # Run comparison
    print("Starting comparison...")
    print()
    results = comparer.compare_all_recordings(comparison_dirs)
    
    # Generate and save summary
    summary = comparer.generate_summary(results)
    print("\n" + summary)
    
    summary_path = Path('comparison') / 'comparison_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

