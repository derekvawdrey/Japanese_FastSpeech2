#!/usr/bin/env python3
"""
Test script to check if Onsei API is caching results incorrectly.
This compares the same test file against different reference files.
"""

import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
VOICE_SPLITS_DIR = Path(__file__).parent.parent / "voice_splits"
SAMPLE_DIR = VOICE_SPLITS_DIR / "sample"
MINE_DIR = VOICE_SPLITS_DIR / "mine"

def test_onsei_comparison():
    """Test if Onsei returns different results for same test file vs different references"""
    
    # Use the same test file (all three are identical anyway)
    test_file = MINE_DIR / "なんで_tokyo.wav"
    
    # Different reference files
    ref_tokyo = SAMPLE_DIR / "なんで_tokyo.wav"
    ref_kyoto = SAMPLE_DIR / "なんで_kyoto.wav"
    ref_shizuoka = SAMPLE_DIR / "なんで_shizuoka.wav"
    
    sentence = "なんで"
    endpoint = f"{API_URL}/compare/data"
    
    print("=" * 80)
    print("Testing Onsei API with same test file vs different reference files")
    print("=" * 80)
    print()
    
    results = {}
    
    for ref_name, ref_file in [("tokyo", ref_tokyo), ("kyoto", ref_kyoto), ("shizuoka", ref_shizuoka)]:
        print(f"Comparing test file against {ref_name} reference...")
        print(f"  Test: {test_file.name}")
        print(f"  Reference: {ref_file.name}")
        
        try:
            with open(ref_file, 'rb') as ref_f, open(test_file, 'rb') as test_f:
                files = {
                    'teacher_audio_file': (ref_file.name, ref_f, 'audio/wav'),
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
                    score = result.get('score')
                    distance = result.get('mean_distance')
                    results[ref_name] = (score, distance)
                    print(f"  ✓ Score: {score}%, Distance: {distance:.6f}")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"  ❌ API error: {error_detail}")
                    results[ref_name] = (None, None)
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[ref_name] = (None, None)
        
        print()
    
    # Analyze results
    print("=" * 80)
    print("Analysis:")
    print("=" * 80)
    
    distances = [results[k][1] for k in results if results[k][1] is not None]
    scores = [results[k][0] for k in results if results[k][0] is not None]
    
    if len(set(distances)) == 1:
        print("⚠️  WARNING: All distances are identical!")
        print(f"   This suggests the API may be caching or not processing correctly.")
        print(f"   Distance: {distances[0]}")
    else:
        print("✓ Distances are different (as expected):")
        for ref_name, (score, distance) in results.items():
            if distance is not None:
                print(f"   {ref_name}: {distance:.6f}")
    
    if len(set(scores)) == 1:
        print("⚠️  WARNING: All scores are identical!")
        print(f"   Score: {scores[0]}%")
    else:
        print("✓ Scores are different (as expected):")
        for ref_name, (score, distance) in results.items():
            if score is not None:
                print(f"   {ref_name}: {score}%")
    
    return results

if __name__ == "__main__":
    test_onsei_comparison()
