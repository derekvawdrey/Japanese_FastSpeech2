#!/usr/bin/env python3
"""
Quick test script to verify Onsei API is working correctly
"""

import sys
import requests
from pathlib import Path


API_URL = "http://127.0.0.1:8000"
VOICE_SPLITS_DIR = Path(__file__).parent.parent / "voice_splits"


def test_api_health():
    """Test if API is running"""
    print("Testing API health...", end=" ")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print("✓ API is running")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to connect: {e}")
        return False


def test_comparison():
    """Test a single comparison"""
    print("\nTesting audio comparison...")
    
    # Use the first sentence
    sentence = "テニスにもあるけど、４大大会って何。"
    filename = f"{sentence}.wav"
    
    reference_file = VOICE_SPLITS_DIR / "sample" / filename
    test_file = VOICE_SPLITS_DIR / "mine" / filename
    
    # Check files exist
    if not reference_file.exists():
        print(f"✗ Reference file not found: {reference_file}")
        return False
    
    if not test_file.exists():
        print(f"✗ Test file not found: {test_file}")
        print("  Using reference as test file for demo purposes...")
        test_file = reference_file
    
    print(f"  Sentence: {sentence}")
    print(f"  Reference: {reference_file.name}")
    print(f"  Test: {test_file.name}")
    print("  Comparing... (this may take 10-30 seconds)")
    
    try:
        endpoint = f"{API_URL}/compare/data"
        
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
                score = result.get('score')
                distance = result.get('mean_distance')
                
                print(f"\n  ✓ Comparison successful!")
                print(f"    Score: {score}%")
                print(f"    Distance: {distance:.4f}")
                
                if test_file == reference_file:
                    print("\n  Note: Since we compared the reference to itself,")
                    print("        the score should be very high (near 100%).")
                
                return True
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                print(f"  ✗ API error: {error_detail}")
                return False
                
    except requests.exceptions.Timeout:
        print("  ✗ Request timeout (API may be overloaded)")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def main():
    """Main entry point"""
    print("=" * 80)
    print("Onsei API Test Script")
    print("=" * 80)
    print()
    
    # Test API health
    if not test_api_health():
        print()
        print("❌ API is not running or not responding!")
        print()
        print("Please start the API first:")
        print("  ./start_api.sh")
        print()
        print("Or manually:")
        print("  cd OnseiModified")
        print("  docker run -p 8000:8000 onsei-api")
        print()
        sys.exit(1)
    
    # Test comparison
    if not test_comparison():
        print()
        print("❌ Comparison test failed!")
        print()
        print("This could be due to:")
        print("  - Missing audio files in voice_splits/")
        print("  - API not fully initialized")
        print("  - Audio files in wrong format")
        print()
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print()
    print("You can now run the full comparison:")
    print("  python3 compare_pitch_accent.py")
    print()


if __name__ == "__main__":
    main()

