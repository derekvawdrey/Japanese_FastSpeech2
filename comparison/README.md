# Mora-Level Pitch Accent Comparison Tool

This tool compares Japanese TTS models' pitch accent accuracy using **mora-level HIGH/LOW pattern analysis** - a linguistically meaningful approach for Japanese pitch accent evaluation.

## Quick Start

```bash
# 1. Start the Onsei API (requires Docker)
./start_api.sh

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the comparison
cd mora_level
python3 mora_accent_comparison_api.py

# 4. Generate visualizations
python3 mora_accent_visualizer.py
```

## Why Mora-Level Analysis?

Japanese pitch accent is fundamentally a **categorical HIGH/LOW distinction** at the mora level, not a continuous F0 curve. This analyzer:

- **Compares H/L patterns**: More linguistically meaningful than F0 distance
- **Detects accent types**: heiban (平板), atamadaka (頭高), nakadaka (中高), odaka (尾高)
- **Provides clear metrics**: Pattern match rate, position accuracy, mora-level accuracy

### Japanese Pitch Accent Types

| Type | Japanese | Pattern | Description |
|------|----------|---------|-------------|
| Heiban (0) | 平板型 | LHH... | No accent drop (flat) |
| Atamadaka (1) | 頭高型 | HLL... | Drop after first mora |
| Nakadaka (n) | 中高型 | LH...HL... | Drop after nth mora |
| Odaka (last) | 尾高型 | LH...HL | Drop after last mora |

## Setup

### 1. Start the Onsei API

The tool requires the Onsei API running in Docker for phoneme segmentation:

```bash
cd OnseiModified
docker build -t onsei .
docker build -f Dockerfile.api -t onsei-api .
docker run -p 8000:8000 onsei-api
```

Or use the helper script:
```bash
./start_api.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** MeCab must be installed before pip install:
- macOS: `brew install mecab mecab-ipadic`
- Ubuntu: `sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8`

Also requires ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`

## Files

```
comparison/
├── mora_level/                      # Mora-level analysis module
│   ├── __init__.py
│   ├── mora_accent_analyzer.py      # Core analyzer
│   ├── mora_accent_comparison_api.py # Main comparison tool
│   └── mora_accent_visualizer.py    # Visualization generator
├── OnseiModified/                   # Onsei API (Docker)
├── README.md
├── requirements.txt
└── start_api.sh
```

## Output

After running the comparison:
- `mora_level/mora_accent_comparison_report.txt` - Text report with rankings
- `mora_level/mora_accent_comparison_results.json` - Detailed JSON results
- `mora_level/mora_accent_graphs/` - Visualization charts

## Understanding Results

| Metric | Description |
|--------|-------------|
| **Accent Position Accuracy** | % with correct accent drop position |
| **Accent Type Accuracy** | % with correct pattern type |
| **Mora Accuracy** | Average % of morae with correct H/L |
| **Pattern Match** | % with exact H/L pattern match |

## TTS Models Evaluated

- FastSpeech2 (yours)
- ElevenLabs Eiko
- Google Translate
- Ondoku
