#!/usr/bin/env python3
"""
Example demonstrating how to use the Onsei API programmatically

This script shows various ways to interact with the API for pitch accent analysis.
"""

import requests
from pathlib import Path
from typing import Optional, Tuple


class OnseiAPI:
    """Wrapper for the Onsei API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
    
    def compare_audio(
        self,
        reference_file: Path,
        student_file: Path,
        sentence: str,
        align_audios: bool = True,
        alignment_method: str = "phonemes",
    ) -> dict:
        """
        Compare two audio files
        
        Args:
            reference_file: Path to reference/teacher audio
            student_file: Path to student/test audio
            sentence: Japanese sentence text
            align_audios: Whether to align the audio files
            alignment_method: 'phonemes' or 'intensity'
            
        Returns:
            Dictionary with comparison results including:
            - score: Similarity score (0-100)
            - mean_distance: Distance metric (lower is better)
            - graphs: Pitch and alignment data
        """
        endpoint = f"{self.base_url}/compare/data"
        
        with open(reference_file, 'rb') as ref_f, open(student_file, 'rb') as stud_f:
            files = {
                'teacher_audio_file': (reference_file.name, ref_f, 'audio/wav'),
                'student_audio_file': (student_file.name, stud_f, 'audio/wav'),
            }
            data = {
                'sentence': sentence,
                'align_audios': str(align_audios).lower(),
                'alignment_method': alignment_method,
                'fallback_if_no_alignment': 'true',
            }
            
            response = requests.post(endpoint, files=files, data=data, timeout=60)
            response.raise_for_status()
            return response.json()
    
    def compare_audio_graph(
        self,
        reference_file: Path,
        student_file: Path,
        sentence: str,
        output_file: Path,
        show_all_graphs: bool = False,
    ) -> None:
        """
        Compare two audio files and save comparison graph as PNG
        
        Args:
            reference_file: Path to reference/teacher audio
            student_file: Path to student/test audio
            sentence: Japanese sentence text
            output_file: Where to save the output PNG
            show_all_graphs: Whether to show individual graphs too
        """
        endpoint = f"{self.base_url}/compare/graph.png"
        
        with open(reference_file, 'rb') as ref_f, open(student_file, 'rb') as stud_f:
            files = {
                'teacher_audio_file': (reference_file.name, ref_f, 'audio/wav'),
                'student_audio_file': (student_file.name, stud_f, 'audio/wav'),
            }
            data = {
                'sentence': sentence,
                'align_audios': 'true',
                'show_all_graphs': str(show_all_graphs).lower(),
                'alignment_method': 'phonemes',
                'fallback_if_no_alignment': 'true',
            }
            
            response = requests.post(endpoint, files=files, data=data, timeout=60)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
    
    def analyze_single_audio(
        self,
        audio_file: Path,
        sentence: str,
        pitch_aggregation: Optional[str] = None,
    ) -> dict:
        """
        Analyze a single audio file (no comparison)
        
        Args:
            audio_file: Path to audio file
            sentence: Japanese sentence text
            pitch_aggregation: Optional 'mean' or 'median' for phoneme pitch aggregation
            
        Returns:
            Dictionary with pitch, intensity, phoneme data
        """
        endpoint = f"{self.base_url}/graph/data"
        
        with open(audio_file, 'rb') as f:
            files = {
                'audio_file': (audio_file.name, f, 'audio/wav'),
            }
            data = {
                'sentence': sentence,
            }
            if pitch_aggregation:
                data['pitch_aggregation'] = pitch_aggregation
            
            response = requests.post(endpoint, files=files, data=data, timeout=60)
            response.raise_for_status()
            return response.json()


def example_1_basic_comparison():
    """Example 1: Basic comparison between two audio files"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Audio Comparison")
    print("=" * 80)
    
    api = OnseiAPI()
    
    # Paths to audio files
    voice_splits = Path(__file__).parent.parent / "voice_splits"
    sentence = "テニスにもあるけど、４大大会って何。"
    
    reference_file = voice_splits / "sample" / f"{sentence}.wav"
    test_file = voice_splits / "mine" / f"{sentence}.wav"
    
    if not reference_file.exists() or not test_file.exists():
        print("Skipping - audio files not found")
        return
    
    print(f"\nComparing:")
    print(f"  Reference: {reference_file.name}")
    print(f"  Test: {test_file.name}")
    print(f"  Sentence: {sentence}")
    
    result = api.compare_audio(reference_file, test_file, sentence)
    
    print(f"\nResults:")
    print(f"  Score: {result['score']}%")
    print(f"  Mean Distance: {result['mean_distance']:.4f}")
    print(f"  Alignment Method: {result['alignment_method']}")


def example_2_save_comparison_graph():
    """Example 2: Save comparison graph as PNG"""
    print("\n" + "=" * 80)
    print("Example 2: Save Comparison Graph")
    print("=" * 80)
    
    api = OnseiAPI()
    
    voice_splits = Path(__file__).parent.parent / "voice_splits"
    sentence = "テニスにもあるけど、４大大会って何。"
    
    reference_file = voice_splits / "sample" / f"{sentence}.wav"
    test_file = voice_splits / "mine" / f"{sentence}.wav"
    output_file = Path(__file__).parent / "example_comparison.png"
    
    if not reference_file.exists() or not test_file.exists():
        print("Skipping - audio files not found")
        return
    
    print(f"\nGenerating comparison graph...")
    print(f"  Output: {output_file}")
    
    api.compare_audio_graph(
        reference_file, 
        test_file, 
        sentence, 
        output_file,
        show_all_graphs=True
    )
    
    print(f"  ✓ Graph saved!")


def example_3_analyze_single_file():
    """Example 3: Analyze a single audio file"""
    print("\n" + "=" * 80)
    print("Example 3: Analyze Single Audio File")
    print("=" * 80)
    
    api = OnseiAPI()
    
    voice_splits = Path(__file__).parent.parent / "voice_splits"
    sentence = "テニスにもあるけど、４大大会って何。"
    audio_file = voice_splits / "sample" / f"{sentence}.wav"
    
    if not audio_file.exists():
        print("Skipping - audio file not found")
        return
    
    print(f"\nAnalyzing: {audio_file.name}")
    
    result = api.analyze_single_audio(
        audio_file, 
        sentence, 
        pitch_aggregation="mean"
    )
    
    print(f"\nResults:")
    print(f"  Mean Pitch: {result['record']['mean_pitch']:.2f} Hz")
    print(f"  Std Pitch: {result['record']['std_pitch']:.2f} Hz")
    print(f"  Number of Phonemes: {len(result['record']['phonemes'])}")
    print(f"  Voice Activity: {result['record']['voice_activity']['begin']:.2f}s - "
          f"{result['record']['voice_activity']['end']:.2f}s")
    
    if result['record'].get('phoneme_pitch'):
        print(f"\n  Phoneme Pitch (mean aggregation):")
        for phoneme in result['record']['phoneme_pitch'][:5]:  # Show first 5
            if phoneme['value'] is not None:
                print(f"    {phoneme['label']}: {phoneme['value']:.2f} Hz")


def example_4_compare_multiple_models():
    """Example 4: Compare multiple TTS models for one sentence"""
    print("\n" + "=" * 80)
    print("Example 4: Compare Multiple Models")
    print("=" * 80)
    
    api = OnseiAPI()
    
    voice_splits = Path(__file__).parent.parent / "voice_splits"
    sentence = "テニスにもあるけど、４大大会って何。"
    reference_file = voice_splits / "sample" / f"{sentence}.wav"
    
    models = {
        "FastSpeech2": voice_splits / "mine" / f"{sentence}.wav",
        "ElevenLabs": voice_splits / "eleven_labs_eiko_wav" / f"{sentence}.wav",
        "Google": voice_splits / "google_translate_wav" / f"{sentence}.wav",
        "Ondoku": voice_splits / "ondoku_wav" / f"{sentence}.wav",
    }
    
    if not reference_file.exists():
        print("Skipping - reference file not found")
        return
    
    print(f"\nSentence: {sentence}")
    print(f"\nComparing {len(models)} models:")
    
    results = []
    for model_name, test_file in models.items():
        if not test_file.exists():
            print(f"  {model_name:15s} - file not found")
            continue
        
        try:
            result = api.compare_audio(reference_file, test_file, sentence)
            results.append({
                'model': model_name,
                'score': result['score'],
                'distance': result['mean_distance']
            })
            print(f"  {model_name:15s} - Score: {result['score']:3d}%, "
                  f"Distance: {result['mean_distance']:.4f}")
        except Exception as e:
            print(f"  {model_name:15s} - Error: {e}")
    
    if results:
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x['distance'])
        print(f"\n  Best model: {results[0]['model']} "
              f"(distance: {results[0]['distance']:.4f})")


def main():
    """Run all examples"""
    print("=" * 80)
    print("Onsei API Usage Examples")
    print("=" * 80)
    print()
    print("These examples demonstrate how to use the Onsei API programmatically.")
    print("Make sure the API is running first: ./start_api.sh")
    print()
    
    # Check if API is running
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        if response.status_code != 200:
            print("⚠️  Warning: API may not be running properly")
    except Exception:
        print("❌ Error: API is not running!")
        print("\nPlease start the API first:")
        print("  ./start_api.sh")
        return
    
    # Run examples
    try:
        example_1_basic_comparison()
        example_2_save_comparison_graph()
        example_3_analyze_single_file()
        example_4_compare_multiple_models()
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

