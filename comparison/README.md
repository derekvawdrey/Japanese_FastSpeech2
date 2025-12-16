# Pitch Accent Comparison Tool

This tool uses the Onsei API to compare different TTS models' pitch accent accuracy against reference Japanese audio samples.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the Onsei API (in one terminal)
./start_api.sh

# 3. Test the API (in another terminal)
python3 test_api.py

# 4. Run the full comparison (takes ~10-15 minutes)
python3 compare_pitch_accent.py

# 5. Generate visualizations
python3 visualize_results.py
```

## Setup Instructions

### 1. Clone the Onsei Repository

```bash
cd comparison
git clone https://github.com/derekvawdrey/OnseiModified.git
```

### 2. Start the Onsei API

You can either use the helper script:

```bash
chmod +x start_api.sh
./start_api.sh
```

Or manually:

```bash
cd OnseiModified
docker build -t onsei .
docker build -f Dockerfile.api -t onsei-api .
docker run -p 8000:8000 onsei-api
```

**Note:** Use `-p 8000:8000` for port mapping (works on all platforms). The `--network=host` option only works properly on Linux.

The API will be available at http://127.0.0.1:8000

### 3. Run the Pitch Accent Comparison

In a new terminal (keep the API running):

```bash
cd comparison
python3 compare_pitch_accent.py
```

## What it Does

The comparison tool:
- Analyzes audio files from multiple TTS models against reference samples
- Computes pitch accent accuracy using Dynamic Time Warping (DTW)
- Generates a comprehensive report ranking the models
- Saves detailed results in both text and JSON formats

## TTS Models Evaluated

- **FastSpeech2 (Mine)** - Your trained model
- **ElevenLabs Eiko** - Commercial TTS service
- **Google Translate** - Google's TTS
- **Ondoku** - Japanese TTS service

## Output Files

After running the comparison, you'll get:
- `pitch_accent_comparison_report.txt` - Human-readable ranking and statistics
- `pitch_accent_comparison_results.json` - Detailed results in JSON format

## Visualizing Results

After running the comparison, generate visual charts:

```bash
python3 visualize_results.py
```

This creates:
- Overall comparison bar charts (distance and score metrics)
- Heatmap showing per-sentence performance
- Box plot showing distance distribution
- Success rate comparison
- Analysis of most discriminative sentences

## Understanding the Results

- **Mean Distance**: Lower is better (0.0 = perfect match)
  - Uses Dynamic Time Warping (DTW) to align and compare pitch patterns
  - Typical values: 0.3-1.5 (lower = more accurate pitch accent)
- **Score**: Similarity percentage (higher is better, 100% = perfect)
  - Calculated as: score = int(1.0 / (distance + 1.0) * 100)
- The tool uses phoneme-based alignment for accurate pitch comparison

## Additional Tools

### API Test Script

Verify the API is working correctly:

```bash
python3 test_api.py
```

This runs a quick test to ensure:
- The Onsei API is accessible
- Audio file comparison works
- Results are returned correctly

### API Usage Examples

Learn how to use the API programmatically:

```bash
python3 example_api_usage.py
```

This demonstrates:
- Basic audio comparison
- Generating comparison graphs
- Analyzing single audio files
- Comparing multiple models for one sentence

## Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install requests matplotlib numpy
```

## Troubleshooting

### API Not Starting

If the API fails to start:
1. Ensure Docker is running: `docker info`
2. Check port 8000 is not in use: `lsof -i :8000`
3. Try rebuilding the images:
   ```bash
   cd OnseiModified
   docker build --no-cache -t onsei .
   docker build --no-cache -f Dockerfile.api -t onsei-api .
   ```

### Comparison Failures

If comparisons fail:
- Ensure all audio files are in WAV format (16kHz mono recommended)
- Check that sentence text exactly matches the audio content
- Some sentences may be too short or complex for automatic phoneme segmentation

### Slow Performance

Each comparison takes 5-30 seconds depending on audio length. For 15 sentences × 4 models = 60 comparisons, expect ~10-20 minutes total.


