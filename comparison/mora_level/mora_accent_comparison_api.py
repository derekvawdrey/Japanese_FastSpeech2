#!/usr/bin/env python3
"""
Mora-Level Pitch Accent Comparison Tool (API Version)

This version uses the Onsei API (running in Docker) for phoneme segmentation,
avoiding the need to install Julius locally.

Usage:
1. Start the Onsei API: cd ../OnseiModified && docker run -p 8000:8000 onsei-api
2. Run this script: python mora_accent_comparison_api.py
"""

import json
import os
import sys
import tempfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

import numpy as np
import parselmouth

from mora_accent_analyzer import (
    MoraAccentAnalyzer,
    AccentPattern,
    AccentComparisonResult,
    AccentType,
    PitchLevel,
    julius_phonemes_to_morae,
    visualize_accent_pattern,
    format_comparison_result,
)


# Configuration
API_URL = "http://127.0.0.1:8000"
VOICE_SPLITS_DIR = Path(__file__).parent.parent.parent / "voice_splits"
SAMPLE_DIR = VOICE_SPLITS_DIR / "sample"
SAMPLE_TEXT_FILE = VOICE_SPLITS_DIR / "sample_text.txt"

# TTS models to evaluate
TTS_MODELS = {
    "FastSpeech2_Mine": VOICE_SPLITS_DIR / "mine",
    "FastSpeech2_Mine_NoPitch": VOICE_SPLITS_DIR / "mine_no_pitch",
    "ElevenLabs_Eiko": VOICE_SPLITS_DIR / "eleven_labs_eiko_wav",
    "Google_Translate": VOICE_SPLITS_DIR / "google_translate_wav",
    "Ondoku": VOICE_SPLITS_DIR / "ondoku_wav",
    "Suzuki Kun": VOICE_SPLITS_DIR / "suzuki_kun",
    "Style-Bert-VITS2": VOICE_SPLITS_DIR / "style_bert_vits2",
}


@dataclass
class MoraAccentResult:
    """Result for a single sentence/model comparison"""
    model_name: str
    sentence: str
    filename: str
    success: bool
    error: Optional[str] = None
    
    # Reference info
    reference_pattern: Optional[str] = None
    reference_accent_type: Optional[str] = None
    reference_accent_position: Optional[int] = None
    
    # Test info
    test_pattern: Optional[str] = None
    test_accent_type: Optional[str] = None
    test_accent_position: Optional[int] = None
    
    # Comparison metrics
    patterns_match: bool = False
    accent_type_match: bool = False
    accent_position_match: bool = False
    mora_accuracy: float = 0.0
    is_correct: bool = False
    
    # Detailed info
    morae_text: Optional[str] = None
    confidence: float = 0.0


@dataclass
class MoraAccentModelStats:
    """Aggregated statistics for a model"""
    model_name: str
    total_sentences: int
    successful_analyses: int
    failed_analyses: int
    
    # Accuracy metrics
    pattern_match_rate: float       # Exact H/L pattern match rate
    average_mora_accuracy: float    # Per-mora H/L accuracy (main metric)
    
    # Individual results
    results: List[MoraAccentResult]


class MoraAccentComparatorAPI:
    """Compare TTS models using mora-level pitch accent analysis via Onsei API"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.analyzer = MoraAccentAnalyzer(
            pitch_floor=75.0,
            pitch_ceiling=500.0,
            time_step=0.005,
            high_low_threshold=0.5
        )
        self.sentences = self._load_sentences()
        
        # Cache for phoneme data to avoid redundant API calls
        self._phoneme_cache: Dict[str, List[Tuple[float, float, str]]] = {}
    
    def _load_sentences(self) -> List[str]:
        """Load sentences from text file"""
        if not SAMPLE_TEXT_FILE.exists():
            print(f"Warning: Sample text file not found at {SAMPLE_TEXT_FILE}")
            return []
        
        with open(SAMPLE_TEXT_FILE, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        return sentences
    
    def check_api_health(self) -> bool:
        """Check if the Onsei API is running"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_phonemes_from_api(
        self,
        wav_path: Path,
        sentence: str
    ) -> Tuple[Optional[List[Tuple[float, float, str]]], Optional[str]]:
        """
        Get phoneme segmentation from Onsei API.
        
        Returns:
            (phonemes, error_message)
        """
        cache_key = f"{wav_path}:{sentence}"
        if cache_key in self._phoneme_cache:
            return self._phoneme_cache[cache_key], None
        
        try:
            endpoint = f"{self.api_url}/graph/data"
            
            with open(wav_path, 'rb') as f:
                files = {'audio_file': (wav_path.name, f, 'audio/wav')}
                data = {'sentence': sentence}
                
                response = requests.post(endpoint, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                record_data = result.get('record', {})
                phonemes_data = record_data.get('phonemes', [])
                
                if not phonemes_data:
                    return None, "No phonemes returned from API"
                
                # Convert to tuple format: (start, end, label)
                phonemes = [
                    (p['start'], p['end'], p['label'])
                    for p in phonemes_data
                ]
                
                self._phoneme_cache[cache_key] = phonemes
                return phonemes, None
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return None, f"API error: {error_detail}"
                
        except requests.exceptions.Timeout:
            return None, "API request timeout"
        except requests.exceptions.RequestException as e:
            return None, f"API request failed: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    def analyze_audio_file(
        self,
        wav_path: Path,
        sentence: str
    ) -> Tuple[Optional[AccentPattern], Optional[str]]:
        """
        Analyze a single audio file using API for phoneme segmentation.
        
        Returns:
            (AccentPattern, error_message)
        """
        # Get phonemes from API
        phonemes, error = self.get_phonemes_from_api(wav_path, sentence)
        if error:
            return None, error
        
        if not phonemes:
            return None, "Could not segment phonemes"
        
        try:
            # Analyze with mora-level analyzer
            pattern = self.analyzer.analyze_audio(str(wav_path), phonemes)
            return pattern, None
        except Exception as e:
            return None, str(e)
    
    def compare_audio_pair(
        self,
        reference_wav: Path,
        test_wav: Path,
        sentence: str
    ) -> Tuple[Optional[AccentComparisonResult], Optional[str]]:
        """
        Compare reference and test audio files.
        """
        # Analyze reference
        ref_pattern, ref_error = self.analyze_audio_file(reference_wav, sentence)
        if ref_error:
            return None, f"Reference analysis failed: {ref_error}"
        
        # Analyze test
        test_pattern, test_error = self.analyze_audio_file(test_wav, sentence)
        if test_error:
            return None, f"Test analysis failed: {test_error}"
        
        # Compare
        result = self.analyzer.compare_patterns(ref_pattern, test_pattern)
        return result, None
    
    def compare_model(self, model_name: str, model_dir: Path) -> MoraAccentModelStats:
        """Compare all samples for a single TTS model"""
        print(f"\n{'='*80}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*80}")
        
        results = []
        
        for i, sentence in enumerate(self.sentences, 1):
            # Find audio files
            filename = f"{sentence}.wav"
            reference_file = SAMPLE_DIR / filename
            test_file = model_dir / filename
            
            # Handle accent variants
            base_sentence = sentence.split('_')[0] if '_' in sentence else sentence
            
            if not reference_file.exists() and '_' in sentence:
                base_filename = f"{base_sentence}.wav"
                base_reference_file = SAMPLE_DIR / base_filename
                if base_reference_file.exists():
                    reference_file = base_reference_file
            
            # Validate files exist
            if not reference_file.exists():
                print(f"[{i}/{len(self.sentences)}] ⚠️  Reference not found: {filename}")
                results.append(MoraAccentResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    success=False,
                    error="Reference file not found"
                ))
                continue
            
            if not test_file.exists():
                print(f"[{i}/{len(self.sentences)}] ⚠️  Test file not found: {filename}")
                results.append(MoraAccentResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    success=False,
                    error="Test file not found"
                ))
                continue
            
            # Run comparison
            print(f"[{i}/{len(self.sentences)}] Analyzing: {sentence[:40]}...")
            
            comparison, error = self.compare_audio_pair(
                reference_file, test_file, base_sentence
            )
            
            if error:
                print(f"                ❌ {error}")
                results.append(MoraAccentResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    success=False,
                    error=error
                ))
            else:
                ref_p = comparison.reference_pattern
                test_p = comparison.test_pattern
                
                status = "✓" if comparison.is_correct else "✗"
                print(f"                {status} Ref: {ref_p.pattern_string} | Test: {test_p.pattern_string} | Mora acc: {comparison.mora_accuracy:.0%}")
                
                results.append(MoraAccentResult(
                    model_name=model_name,
                    sentence=sentence,
                    filename=filename,
                    success=True,
                    reference_pattern=ref_p.pattern_string,
                    reference_accent_type=ref_p.accent_type.name,
                    reference_accent_position=ref_p.accent_position,
                    test_pattern=test_p.pattern_string,
                    test_accent_type=test_p.accent_type.name,
                    test_accent_position=test_p.accent_position,
                    patterns_match=comparison.patterns_match,
                    accent_type_match=comparison.accent_type_match,
                    accent_position_match=comparison.accent_position_match,
                    mora_accuracy=comparison.mora_accuracy,
                    is_correct=comparison.is_correct,
                    morae_text=''.join(m.text for m in test_p.morae),
                    confidence=test_p.confidence
                ))
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.3)
        
        # Calculate statistics
        successful = [r for r in results if r.success]
        
        if successful:
            pattern_match_rate = sum(1 for r in successful if r.patterns_match) / len(successful)
            average_mora_accuracy = sum(r.mora_accuracy for r in successful) / len(successful)
        else:
            pattern_match_rate = 0.0
            average_mora_accuracy = 0.0
        
        return MoraAccentModelStats(
            model_name=model_name,
            total_sentences=len(self.sentences),
            successful_analyses=len(successful),
            failed_analyses=len(results) - len(successful),
            pattern_match_rate=pattern_match_rate,
            average_mora_accuracy=average_mora_accuracy,
            results=results
        )
    
    def compare_all_models(self) -> Dict[str, MoraAccentModelStats]:
        """Compare all TTS models"""
        all_stats = {}
        
        for model_name, model_dir in TTS_MODELS.items():
            if not model_dir.exists():
                print(f"\n⚠️  Skipping {model_name}: Directory not found at {model_dir}")
                continue
            
            stats = self.compare_model(model_name, model_dir)
            all_stats[model_name] = stats
        
        return all_stats
    
    def generate_report(self, all_stats: Dict[str, MoraAccentModelStats]) -> str:
        """Generate comparison report"""
        lines = []
        
        lines.append("=" * 80)
        lines.append("MORA-LEVEL PITCH ACCENT COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("This report uses linguistically-motivated mora-level analysis")
        lines.append("comparing HIGH/LOW pitch patterns rather than continuous F0 curves.")
        lines.append("")
        
        # Overall rankings
        lines.append("OVERALL RANKINGS (by Mora H/L Accuracy)")
        lines.append("-" * 80)
        
        ranked = sorted(
            all_stats.items(),
            key=lambda x: x[1].average_mora_accuracy,
            reverse=True
        )
        
        for rank, (model_name, stats) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            lines.append(
                f"{medal} {model_name:25s} | "
                f"Mora Acc: {stats.average_mora_accuracy:.1%} | "
                f"Pattern Match: {stats.pattern_match_rate:.1%}"
            )
        
        lines.append("")
        lines.append("")
        
        # Detailed statistics
        lines.append("DETAILED STATISTICS")
        lines.append("=" * 80)
        
        for model_name, stats in ranked:
            lines.append("")
            lines.append(f"Model: {model_name}")
            lines.append("-" * 80)
            lines.append(f"  Total Sentences:           {stats.total_sentences}")
            lines.append(f"  Successful Analyses:       {stats.successful_analyses}")
            lines.append(f"  Failed Analyses:           {stats.failed_analyses}")
            lines.append("")
            lines.append(f"  Average Mora H/L Accuracy: {stats.average_mora_accuracy:.1%}")
            lines.append(f"  Exact Pattern Match Rate:  {stats.pattern_match_rate:.1%}")
        
        # Per-sentence breakdown
        lines.append("")
        lines.append("")
        lines.append("PER-SENTENCE BREAKDOWN")
        lines.append("=" * 80)
        
        sentence_results = defaultdict(list)
        for model_name, stats in all_stats.items():
            for result in stats.results:
                sentence_results[result.sentence].append((model_name, result))
        
        for sentence, results in sentence_results.items():
            lines.append("")
            lines.append(f"Sentence: {sentence}")
            lines.append("-" * 80)
            
            ref_shown = False
            for model_name, result in results:
                if result.success and result.reference_pattern and not ref_shown:
                    lines.append(f"  Reference: {result.reference_pattern}")
                    ref_shown = True
                    break
            
            successful_results = [
                (name, res) for name, res in results
                if res.success
            ]
            successful_results.sort(key=lambda x: x[1].mora_accuracy, reverse=True)
            
            for model_name, result in successful_results:
                status = "✓" if result.is_correct else "✗"
                lines.append(
                    f"  {status} {model_name:25s} | "
                    f"Pattern: {result.test_pattern:10s} | "
                    f"Mora Acc: {result.mora_accuracy:.0%}"
                )
            
            failed_results = [(name, res) for name, res in results if not res.success]
            for model_name, result in failed_results:
                lines.append(f"  ✗ {model_name:25s} | Failed: {result.error}")
        
        return "\n".join(lines)
    
    def save_results(
        self,
        all_stats: Dict[str, MoraAccentModelStats],
        output_dir: Path
    ) -> None:
        """Save results to files"""
        report = self.generate_report(all_stats)
        report_file = output_dir / "mora_accent_comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")
        
        json_data = {}
        for model_name, stats in all_stats.items():
            json_data[model_name] = {
                'total_sentences': stats.total_sentences,
                'successful_analyses': stats.successful_analyses,
                'failed_analyses': stats.failed_analyses,
                'pattern_match_rate': stats.pattern_match_rate,
                'average_mora_accuracy': stats.average_mora_accuracy,
                'results': [asdict(r) for r in stats.results]
            }
        
        json_file = output_dir / "mora_accent_comparison_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to: {json_file}")


def main():
    """Main entry point"""
    print("=" * 80)
    print("Mora-Level Japanese Pitch Accent Comparison Tool (API Version)")
    print("=" * 80)
    print()
    print("This tool uses the Onsei API for phoneme segmentation.")
    print()
    
    comparator = MoraAccentComparatorAPI()
    
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
    
    input("Press Enter to start the analysis...")
    
    # Run analysis
    start_time = time.time()
    all_stats = comparator.compare_all_models()
    elapsed = time.time() - start_time
    
    # Generate and print report
    print("\n" * 2)
    report = comparator.generate_report(all_stats)
    print(report)
    
    # Save results
    output_dir = Path(__file__).parent
    comparator.save_results(all_stats, output_dir)
    
    print()
    print("=" * 80)
    print(f"Analysis complete! Total time: {elapsed:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
