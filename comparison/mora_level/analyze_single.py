#!/usr/bin/env python3
"""
Analyze pitch accent pattern of a single audio file.

Usage:
    python analyze_single.py <audio_file> <sentence>
    python analyze_single.py <audio_file> <sentence> --verbose

Examples:
    python analyze_single.py audio.wav "なんで"
    python analyze_single.py /path/to/file.wav "こんにちは" --verbose
"""

import argparse
import sys
import requests
from pathlib import Path

from mora_accent_analyzer import (
    MoraAccentAnalyzer,
    visualize_accent_pattern,
    AccentType,
)


API_URL = "http://127.0.0.1:8000"


def get_phonemes_from_api(wav_path: Path, sentence: str):
    """Get phoneme segmentation from Onsei API."""
    try:
        endpoint = f"{API_URL}/graph/data"
        
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
            
            return phonemes, None
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return None, f"API error: {error_detail}"
            
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Onsei API. Make sure it's running on port 8000."
    except requests.exceptions.Timeout:
        return None, "API request timeout"
    except Exception as e:
        return None, f"Error: {str(e)}"


def analyze_audio(wav_path: Path, sentence: str, verbose: bool = False):
    """Analyze a single audio file and return the pitch accent pattern."""
    
    # Get phonemes from API
    phonemes, error = get_phonemes_from_api(wav_path, sentence)
    if error:
        return None, error
    
    if verbose:
        print(f"\nPhonemes detected: {len(phonemes)}")
        for start, end, label in phonemes:
            print(f"  {start:.3f} - {end:.3f}: {label}")
    
    # Analyze with mora-level analyzer
    analyzer = MoraAccentAnalyzer(
        pitch_floor=75.0,
        pitch_ceiling=500.0,
        time_step=0.005,
        high_low_threshold=0.5
    )
    
    try:
        pattern = analyzer.analyze_audio(str(wav_path), phonemes)
        return pattern, None
    except Exception as e:
        return None, f"Analysis error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pitch accent pattern of a single audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_single.py audio.wav "なんで"
    python analyze_single.py /path/to/file.wav "こんにちは" --verbose
    
Note: Requires the Onsei API to be running. Start it with:
    cd comparison/OnseiModified
    docker run -p 8000:8000 onsei-api
        """
    )
    
    parser.add_argument("audio_file", type=str, help="Path to the WAV audio file")
    parser.add_argument("sentence", type=str, help="The Japanese text being spoken")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Show detailed output including phonemes and F0 values")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    
    args = parser.parse_args()
    
    wav_path = Path(args.audio_file)
    
    if not wav_path.exists():
        print(f"Error: File not found: {wav_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check API health
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code != 200:
            raise Exception("API not healthy")
    except:
        print("Error: Onsei API is not running!", file=sys.stderr)
        print("\nPlease start the API first:", file=sys.stderr)
        print("  cd comparison/OnseiModified", file=sys.stderr)
        print("  docker run -p 8000:8000 onsei-api", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    pattern, error = analyze_audio(wav_path, args.sentence, args.verbose)
    
    if error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        import json
        result = {
            "file": str(wav_path),
            "sentence": args.sentence,
            "pattern": pattern.pattern_string,
            "accent_type": pattern.accent_type.name,
            "accent_position": pattern.accent_position,
            "confidence": pattern.confidence,
            "morae": [
                {
                    "text": m.text,
                    "pitch_level": m.pitch_level.value,
                    "mean_f0": m.mean_f0,
                }
                for m in pattern.morae
            ]
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Simple output
        print()
        print(f"File:     {wav_path.name}")
        print(f"Sentence: {args.sentence}")
        print()
        print(f"Pattern:  {pattern.pattern_string}")
        print(f"Type:     {pattern.accent_type.name} (position {pattern.accent_position})")
        print(f"Confidence: {pattern.confidence:.0%}")
        print()
        
        # Show mora breakdown
        mora_text = ''.join(m.text for m in pattern.morae)
        print(f"Morae:    {mora_text}")
        print(f"          {'  '.join(m.pitch_level.value for m in pattern.morae)}")
        
        if args.verbose:
            print()
            print("Detailed mora analysis:")
            for i, mora in enumerate(pattern.morae, 1):
                f0_str = f"{mora.mean_f0:.1f} Hz" if mora.mean_f0 else "unvoiced"
                print(f"  {i}. {mora.text:4s} = {mora.pitch_level.value} ({f0_str})")
            
            print()
            print(visualize_accent_pattern(pattern))


if __name__ == "__main__":
    main()
