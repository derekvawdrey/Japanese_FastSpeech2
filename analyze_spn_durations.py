import os
import re
import numpy as np

sampling_rate = 22050
hop_length = 256

# Find all TextGrid files
textgrid_dir = "preprocessed_data/kokoro/TextGrid/kokoro"
textgrid_files = [f for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

spn_durations = []
spn_durations_frames = []

print("Analyzing spn durations in TextGrid files...")
print(f"Found {len(textgrid_files)} TextGrid files\n")

for filename in textgrid_files[:100]:  # Sample first 100 files
    filepath = os.path.join(textgrid_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all spn intervals
    pattern = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "spn"'
    matches = re.findall(pattern, content)
    
    for start_str, end_str in matches:
        start = float(start_str)
        end = float(end_str)
        duration_sec = end - start
        duration_frames = (end - start) * sampling_rate / hop_length
        duration_frames_rounded = int(np.round(duration_frames))
        
        spn_durations.append((filename, start, end, duration_sec, duration_frames, duration_frames_rounded))
        spn_durations_frames.append(duration_frames_rounded)

if spn_durations:
    print(f"Found {len(spn_durations)} spn phonemes in sample")
    print(f"\nDuration statistics (in frames after rounding):")
    print(f"  Min: {min(spn_durations_frames)}")
    print(f"  Max: {max(spn_durations_frames)}")
    print(f"  Mean: {np.mean(spn_durations_frames):.2f}")
    print(f"  Median: {np.median(spn_durations_frames):.2f}")
    
    print(f"\nZero-duration spn: {sum(1 for d in spn_durations_frames if d == 0)}")
    print(f"One-frame spn: {sum(1 for d in spn_durations_frames if d == 1)}")
    print(f"Two-frame spn: {sum(1 for d in spn_durations_frames if d == 2)}")
    
    print(f"\nExamples of shortest durations:")
    sorted_durations = sorted(spn_durations, key=lambda x: x[5])[:10]
    for filename, start, end, dur_sec, dur_frames, dur_rounded in sorted_durations:
        print(f"  {filename}: {dur_sec:.4f}s = {dur_frames:.2f} frames -> {dur_rounded} frames (rounded)")
else:
    print("No spn phonemes found in sample")

