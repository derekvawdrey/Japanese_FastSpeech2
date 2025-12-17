#!/usr/bin/env python3
"""
Pitch Accent Comparison Tool

This script compares multiple TTS model outputs against reference audio samples
using the Onsei API to determine which model best follows Japanese pitch accent patterns.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from statistics import mean, stdev
from dataclasses import dataclass, asdict
from collections import defaultdict
import time


# Configuration
API_URL = "http://127.0.0.1:8000"
VOICE_SPLITS_DIR = Path(__file__).parent.parent / "voice_splits"
SAMPLE_DIR = VOICE_SPLITS_DIR / "sample"
SAMPLE_TEXT_FILE = VOICE_SPLITS_DIR / "sample_text.txt"

# TTS models to evaluate
TTS_MODELS = {
    "FastSpeech2 (Mine)": VOICE_SPLITS_DIR / "mine",
    "ElevenLabs Eiko": VOICE_SPLITS_DIR / "eleven_labs_eiko_wav",
    "Google Translate": VOICE_SPLITS_DIR / "google_translate_wav",
    "Ondoku": VOICE_SPLITS_DIR / "ondoku_wav",
}


@dataclass
class ComparisonResult:
    """Result of comparing a single TTS output to reference"""
    model_name: str
    sentence: str
    filename: str
    score: Optional[int]
    mean_distance: Optional[float]
    success: bool
    error: Optional[str] = None


@dataclass
class ModelStats:
    """Statistics for a single TTS model"""
    model_name: str
    total_sentences: int
    successful_comparisons: int
    failed_comparisons: int
    average_score: Optional[float]
    average_distance: Optional[float]
    std_distance: Optional[float]
    min_distance: Optional[float]
    max_distance: Optional[float]
    results: List[ComparisonResult]


class PitchAccentComparator:
    """Compares TTS models' pitch accent accuracy using Onsei API"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.sentences = self._load_sentences()
        
    def _load_sentences(self) -> List[str]:
        """Load Japanese sentences from text file"""
        with open(SAMPLE_TEXT_FILE, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
            return [sentence.split('_')[0] for sentence in sentences]
    
    def check_api_health(self) -> bool:
        """Check if the Onsei API is running"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def compare_audio_files(
        self,
        reference_file: Path,
        test_file: Path,
        sentence: str
    ) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """
        Compare two audio files using the Onsei API
        
        Returns:
            (score, mean_distance, error_message)
        """
        endpoint = f"{self.api_url}/compare/data"
        
        try:
            with open(reference_file, 'rb') as ref_f, open(test_file, 'rb') as test_f:
                files = {
                    'teacher_audio_file': (reference_file.name, ref_f, 'audio/wav'),
                    'student_audio_file': (test_file.name, test_f, 'audio/wav'),
                }

                data = {
                    'sentence': sentence,
                    'align_audios': 'true',
                    'alignment_method': 'phonemes',
                    'fallback_if_no_alignment': 'true',
                }
                
                response = requests.post(endpoint, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('score'), result.get('mean_distance'), None
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    return None, None, f"API error: {error_detail}"
                    
        except requests.exceptions.Timeout:
            return None, None, "Request timeout"
        except requests.exceptions.RequestException as e:
            return None, None, f"Request failed: {str(e)}"
        except Exception as e:
            return None, None, f"Unexpected error: {str(e)}"
    
    def compare_model(self, model_name: str, model_dir: Path) -> ModelStats:
        """Compare all samples for a single TTS model"""
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}")
        
        results = []
        
        for i, sentence in enumerate(self.sentences, 1):
            # Find the corresponding audio files
            filename = f"{sentence}.wav"
            reference_file = SAMPLE_DIR / filename
            test_file = model_dir / filename
            
            # Check if both files exist
            if not reference_file.exists():
                print(f"[{i}/{len(self.sentences)}] ⚠️  Reference file not found: {filename}")
                results.append(ComparisonResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    score=None,
                    mean_distance=None,
                    success=False,
                    error="Reference file not found"
                ))
                continue
            
            if not test_file.exists():
                print(f"[{i}/{len(self.sentences)}] ⚠️  Test file not found: {filename}")
                results.append(ComparisonResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    score=None,
                    mean_distance=None,
                    success=False,
                    error="Test file not found"
                ))
                continue
            
            # Compare the files
            print(f"[{i}/{len(self.sentences)}] Comparing: {sentence[:50]}...")
            score, mean_distance, error = self.compare_audio_files(
                reference_file, test_file, sentence
            )
            
            if error:
                print(f"                ❌ Failed: {error}")
                results.append(ComparisonResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    score=score,
                    mean_distance=mean_distance,
                    success=False,
                    error=error
                ))
            else:
                print(f"                ✓ Score: {score}%, Distance: {mean_distance:.3f}")
                results.append(ComparisonResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    score=score,
                    mean_distance=mean_distance,
                    success=True
                ))
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        # Calculate statistics
        successful_results = [r for r in results if r.success and r.mean_distance is not None]
        scores = [r.score for r in successful_results if r.score is not None]
        distances = [r.mean_distance for r in successful_results]
        
        return ModelStats(
            model_name=model_name,
            total_sentences=len(self.sentences),
            successful_comparisons=len(successful_results),
            failed_comparisons=len(results) - len(successful_results),
            average_score=mean(scores) if scores else None,
            average_distance=mean(distances) if distances else None,
            std_distance=stdev(distances) if len(distances) > 1 else None,
            min_distance=min(distances) if distances else None,
            max_distance=max(distances) if distances else None,
            results=results
        )
    
    def compare_all_models(self) -> Dict[str, ModelStats]:
        """Compare all TTS models"""
        all_stats = {}
        
        for model_name, model_dir in TTS_MODELS.items():
            if not model_dir.exists():
                print(f"\n⚠️  Skipping {model_name}: Directory not found at {model_dir}")
                continue
            
            stats = self.compare_model(model_name, model_dir)
            all_stats[model_name] = stats
        
        return all_stats
    
    def generate_report(self, all_stats: Dict[str, ModelStats]) -> str:
        """Generate a comprehensive comparison report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("PITCH ACCENT COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall Rankings
        report_lines.append("OVERALL RANKINGS (by Average Distance - Lower is Better)")
        report_lines.append("-" * 80)
        
        # Sort models by average distance
        ranked_models = [
            (name, stats) 
            for name, stats in all_stats.items() 
            if stats.average_distance is not None
        ]
        ranked_models.sort(key=lambda x: x[1].average_distance)
        
        for rank, (model_name, stats) in enumerate(ranked_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            report_lines.append(
                f"{medal} {model_name:30s} | "
                f"Avg Distance: {stats.average_distance:.4f} | "
                f"Avg Score: {stats.average_score:.1f}%"
            )
        
        report_lines.append("")
        report_lines.append("")
        
        # Detailed Statistics
        report_lines.append("DETAILED STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for model_name, stats in ranked_models:
            report_lines.append(f"Model: {model_name}")
            report_lines.append("-" * 80)
            report_lines.append(f"  Total Sentences:       {stats.total_sentences}")
            report_lines.append(f"  Successful:            {stats.successful_comparisons}")
            report_lines.append(f"  Failed:                {stats.failed_comparisons}")
            
            if stats.average_distance is not None:
                report_lines.append(f"  Average Distance:      {stats.average_distance:.4f}")
                report_lines.append(f"  Std Deviation:         {stats.std_distance:.4f}" if stats.std_distance else "  Std Deviation:         N/A")
                report_lines.append(f"  Min Distance:          {stats.min_distance:.4f}")
                report_lines.append(f"  Max Distance:          {stats.max_distance:.4f}")
                report_lines.append(f"  Average Score:         {stats.average_score:.1f}%")
            else:
                report_lines.append("  No successful comparisons")
            
            report_lines.append("")
        
        # Per-sentence breakdown
        report_lines.append("")
        report_lines.append("PER-SENTENCE BREAKDOWN")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Group results by sentence
        sentence_results = defaultdict(list)
        for model_name, stats in all_stats.items():
            for result in stats.results:
                sentence_results[result.sentence].append((model_name, result))
        
        for sentence, results in sentence_results.items():
            report_lines.append(f"Sentence: {sentence}")
            report_lines.append("-" * 80)
            
            # Sort by distance (lower is better)
            successful_results = [
                (name, res) for name, res in results 
                if res.success and res.mean_distance is not None
            ]
            successful_results.sort(key=lambda x: x[1].mean_distance)
            
            for model_name, result in successful_results:
                report_lines.append(
                    f"  {model_name:30s} | "
                    f"Distance: {result.mean_distance:.4f} | "
                    f"Score: {result.score}%"
                )
            
            # Show failed attempts
            failed_results = [
                (name, res) for name, res in results 
                if not res.success
            ]
            for model_name, result in failed_results:
                report_lines.append(
                    f"  {model_name:30s} | ❌ Failed: {result.error}"
                )
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_json_results(self, all_stats: Dict[str, ModelStats], output_file: Path):
        """Save detailed results in JSON format"""
        json_data = {
            model_name: {
                **asdict(stats),
                'results': [asdict(r) for r in stats.results]
            }
            for model_name, stats in all_stats.items()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main entry point"""
    print("=" * 80)
    print("Japanese TTS Pitch Accent Comparison Tool")
    print("=" * 80)
    print()
    
    # Initialize comparator
    comparator = PitchAccentComparator()
    
    # Check API health
    print("Checking Onsei API status...")
    if not comparator.check_api_health():
        print("❌ ERROR: Onsei API is not running!")
        print()
        print("Please start the API first:")
        print("  cd comparison/OnseiModified")
        print("  docker build -t onsei .")
        print("  docker build -f Dockerfile.api -t onsei-api .")
        print("  docker run -p 8000:8000 onsei-api")
        print()
        sys.exit(1)
    
    print("✓ Onsei API is running")
    print()
    
    # Show configuration
    print(f"Sample directory: {SAMPLE_DIR}")
    print(f"Number of sentences: {len(comparator.sentences)}")
    print(f"Models to evaluate: {len(TTS_MODELS)}")
    for model_name, model_dir in TTS_MODELS.items():
        status = "✓" if model_dir.exists() else "✗"
        print(f"  {status} {model_name}")
    print()
    
    input("Press Enter to start the comparison (this may take several minutes)...")
    
    # Run comparisons
    start_time = time.time()
    all_stats = comparator.compare_all_models()
    elapsed_time = time.time() - start_time
    
    # Generate report
    print("\n" * 2)
    report = comparator.generate_report(all_stats)
    print(report)
    
    # Save results
    output_dir = Path(__file__).parent
    
    # Save text report
    report_file = output_dir / "pitch_accent_comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write(f"\n\nTotal comparison time: {elapsed_time:.1f} seconds\n")
    print(f"\nReport saved to: {report_file}")
    
    # Save JSON results
    json_file = output_dir / "pitch_accent_comparison_results.json"
    comparator.save_json_results(all_stats, json_file)
    
    print()
    print("=" * 80)
    print(f"Comparison complete! Total time: {elapsed_time:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()

