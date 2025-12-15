#!/usr/bin/env python
"""
Convert MP3 files to WAV for comparison.
"""

import subprocess
from pathlib import Path

def convert_mp3_folder(input_dir, output_dir):
    """Convert all MP3 files in input_dir to WAV files in output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    mp3_files = list(input_path.glob("*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {input_dir}")
        return
    
    print(f"Converting {len(mp3_files)} MP3 files from {input_dir}...")
    
    for mp3_file in mp3_files:
        wav_file = output_path / f"{mp3_file.stem}.wav"
        
        try:
            # Use ffmpeg to convert
            subprocess.run([
                'ffmpeg', '-i', str(mp3_file),
                '-acodec', 'pcm_s16le',
                '-ar', '22050',
                '-ac', '1',  # mono
                '-y',  # overwrite
                str(wav_file)
            ], check=True, capture_output=True)
            print(f"  ✓ {mp3_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {mp3_file.name}: {e}")
        except FileNotFoundError:
            print("  ✗ ffmpeg not found. Install with: brew install ffmpeg")
            return

if __name__ == "__main__":
    # Convert ondoku and google translate MP3s to WAV
    convert_mp3_folder("voice_splits/ondoku", "voice_splits/ondoku_wav")
    convert_mp3_folder("voice_splits/google translate", "voice_splits/google_translate_wav")
    convert_mp3_folder("voice_splits/eleven_labs_eiko", "voice_splits/eleven_labs_eiko_wav")
    
    print("\nConversion complete! Now run the comparison:")
    print("python comparison/pitch_accent_compare.py")

