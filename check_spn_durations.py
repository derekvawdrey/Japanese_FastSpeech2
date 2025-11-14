import os
import numpy as np
import json

# Load the training data to get phoneme sequences
train_file = "preprocessed_data/kokoro/train.txt"
duration_dir = "preprocessed_data/kokoro/duration"

# Load symbol mapping to find spn index
with open("text/symbols.py", "r", encoding="utf-8") as f:
    symbols_content = f.read()
    # Extract symbols list
    import re
    # Find the symbols definition
    # We'll use text_to_sequence to get the index

from text import text_to_sequence

# Find spn symbol index
spn_symbol_id = None
try:
    # Try to get spn index from symbols
    test_seq = text_to_sequence("{spn}", [])
    # Actually, let's just read the symbols file properly
    with open("text/__init__.py", "r") as f:
        init_content = f.read()
    
    # Import symbols
    import sys
    sys.path.insert(0, ".")
    from text.symbols import symbols
    
    spn_symbol_id = symbols.index("spn") if "spn" in symbols else None
    print(f"spn symbol index: {spn_symbol_id}")
except Exception as e:
    print(f"Error finding spn index: {e}")
    # Fallback: check if spn is in the symbols
    with open("text/symbols.py", "r") as f:
        content = f.read()
        if '"spn"' in content or "'spn'" in content:
            # Count position
            lines = content.split('\n')
            for line in lines:
                if 'spn' in line:
                    print(f"Found spn in: {line}")

# Read training data
zero_duration_spn = []
one_duration_spn = []
total_spn = 0
total_utterances = 0

if os.path.exists(train_file):
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            total_utterances += 1
            parts = line.strip().split("|")
            if len(parts) >= 3:
                basename = parts[0]
                speaker = parts[1]
                phoneme_seq = parts[2]  # e.g., "{a b c spn d e}"
                
                # Check if spn is in the sequence
                if "spn" in phoneme_seq:
                    # Load corresponding duration file
                    duration_file = os.path.join(duration_dir, f"{speaker}-duration-{basename}.npy")
                    if os.path.exists(duration_file):
                        durations = np.load(duration_file)
                        
                        # Parse phoneme sequence to find spn positions
                        # Remove curly braces and split
                        phones = phoneme_seq.strip("{}").split()
                        
                        for i, phone in enumerate(phones):
                            if phone == "spn" and i < len(durations):
                                total_spn += 1
                                dur = durations[i]
                                if dur == 0:
                                    zero_duration_spn.append((basename, speaker, i, dur))
                                elif dur == 1:
                                    one_duration_spn.append((basename, speaker, i, dur))

print(f"\nAnalysis Results:")
print(f"Total utterances checked: {total_utterances}")
print(f"Total spn phonemes found: {total_spn}")
print(f"\nZero-duration spn: {len(zero_duration_spn)}")
print(f"One-frame spn: {len(one_duration_spn)}")

if zero_duration_spn:
    print(f"\nExamples of zero-duration spn:")
    for basename, speaker, pos, dur in zero_duration_spn[:10]:
        print(f"  {speaker}-{basename}, position {pos}, duration: {dur}")
else:
    print("\n[OK] No zero-duration spn found in preprocessed data!")

if one_duration_spn:
    print(f"\nExamples of one-frame spn (first 10):")
    for basename, speaker, pos, dur in one_duration_spn[:10]:
        print(f"  {speaker}-{basename}, position {pos}, duration: {dur}")

# Also check duration statistics
if total_spn > 0:
    all_spn_durations = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                basename = parts[0]
                speaker = parts[1]
                phoneme_seq = parts[2]
                
                if "spn" in phoneme_seq:
                    duration_file = os.path.join(duration_dir, f"{speaker}-duration-{basename}.npy")
                    if os.path.exists(duration_file):
                        durations = np.load(duration_file)
                        phones = phoneme_seq.strip("{}").split()
                        
                        for i, phone in enumerate(phones):
                            if phone == "spn" and i < len(durations):
                                all_spn_durations.append(durations[i])
    
    if all_spn_durations:
        print(f"\nspn Duration Statistics:")
        print(f"  Min: {min(all_spn_durations)}")
        print(f"  Max: {max(all_spn_durations)}")
        print(f"  Mean: {np.mean(all_spn_durations):.2f}")
        print(f"  Median: {np.median(all_spn_durations):.2f}")
        print(f"  Std: {np.std(all_spn_durations):.2f}")

