# FastSpeech 2 - Japanese Text-to-Speech Implementation

A PyTorch implementation of FastSpeech 2, a fast, robust, and controllable neural text-to-speech (TTS) model, specifically adapted for Japanese speech synthesis. This implementation uses the Kokoro dataset and supports multi-speaker TTS with controllable pitch, energy, and duration.

## Overview

FastSpeech 2 is a non-autoregressive TTS model that generates mel-spectrograms directly from text, enabling fast and parallel synthesis. This implementation includes:

- **Japanese language support** with IPA phoneme representation
- **Multi-speaker TTS** with speaker embeddings
- **Controllable synthesis** with adjustable pitch, energy, and duration
- **HiFi-GAN vocoder** for high-quality waveform generation
- **Montreal Forced Aligner (MFA)** integration for phoneme alignment

## Features

- ⚡ **Fast inference**: Non-autoregressive architecture enables parallel generation
- 🎯 **High quality**: Transformer-based encoder-decoder with variance adaptor
- 🎚️ **Controllable**: Adjust pitch, energy, and speaking rate during synthesis
- 👥 **Multi-speaker**: Support for multiple speakers with speaker embeddings
- 🇯🇵 **Japanese optimized**: Custom Japanese text preprocessing with pyopenjtalk

## Project Structure

```
fast-speech-2/
├── audio/                 # Audio processing utilities (STFT, mel-spectrogram)
├── config/               # Configuration files (preprocess, model, train)
│   └── kokoro/          # Kokoro dataset configurations
├── hifigan/              # HiFi-GAN vocoder implementation
├── lexicon/              # Phoneme dictionaries (Japanese MFA dictionary)
├── model/                # FastSpeech 2 model implementation
│   ├── fastspeech2.py   # Main model architecture
│   ├── loss.py          # Loss functions
│   └── modules.py        # Model components (VarianceAdaptor)
├── output/               # Training outputs (checkpoints, logs, results)
├── preprocessed_data/    # Preprocessed dataset (mel, pitch, energy, duration)
├── preprocessor/         # Data preprocessing scripts
├── raw_data/             # Raw audio dataset
├── text/                 # Text processing (cleaners, symbols, phoneme conversion)
├── transformer/          # Transformer encoder/decoder implementation
├── utils/                # Utility functions
├── dataset.py            # PyTorch dataset classes
├── evaluate.py           # Model evaluation script
├── preprocess.py         # Data preprocessing pipeline
├── prepare_align.py      # MFA alignment preparation
├── synthesize.py         # Text-to-speech synthesis script
└── train.py              # Training script
```

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended) or CPU-only setup
- Conda (for environment management)

### Setup
#### Automatic Setup

I have provided a script to setup the environment for both GPU and CPU environments. If you are going to manually setup the environment, you MUST use conda. If you do not use conda you will have issues with MFA (Montreal Force Aligner).

The automatic setup will also install the required models for running the trained model.

#### Manual Setup

For CPU Only installation, the documentation is found in `docs/CPU_INSTALLATION.md`.

For GPU Only installation, the documentation is found in `docs/GPU_INSTALLATION.md`


## Usage

### 1. Data Preparation

Place your audio files and transcripts in the `raw_data/` directory following the expected structure.

Audio files must be in .wav format with a 22050 Hz smaple rate. and the transcripts must be .lab files. Both the .wav and .lab files need to be named the same for each audio/text pair.

### 2. Prepare Alignment

Prepare data for Montreal Forced Aligner (MFA):

```bash
python prepare_align.py config/kokoro/preprocess.yaml
```

### 3. Run MFA Alignment

Use Montreal Forced Aligner to generate phoneme alignments:

```bash
mfa align raw_data/kokoro lexicon/japanese_mfa.dict japanese_mfa preprocessed_data/kokoro/TextGrid
```

### 4. Preprocess Data

Extract features (mel-spectrograms, pitch, energy, duration):

```bash
python preprocess.py config/kokoro/preprocess.yaml
```

This will generate:
- Mel-spectrograms
- Pitch contours
- Energy features
- Duration labels
- Train/validation splits

### 5. Training

Train the FastSpeech 2 model:

```bash
python train.py \
    -p config/kokoro/preprocess.yaml \
    -m config/kokoro/model.yaml \
    -t config/kokoro/train.yaml \
    --restore_step 0
```

To resume from a checkpoint:

```bash
python train.py \
    -p config/kokoro/preprocess.yaml \
    -m config/kokoro/model.yaml \
    -t config/kokoro/train.yaml \
    --restore_step 30000
```

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir output/log/kokoro
```

### 6. Evaluation

Evaluate the model on the validation set:

```bash
python evaluate.py \
    -p config/kokoro/preprocess.yaml \
    -m config/kokoro/model.yaml \
    -t config/kokoro/train.yaml \
    --restore_step 30000
```

### 7. Synthesis

#### Single Sentence Synthesis

Synthesize a single Japanese sentence:

```bash
python synthesize.py \
    --mode single \
    --text "こんにちは、これはテストです。" \
    --restore_step 30000 \
    -p config/kokoro/preprocess.yaml \
    -m config/kokoro/model.yaml \
    -t config/kokoro/train.yaml \
    --speaker_id 0
```

#### Batch Synthesis

Synthesize from a text file (format: `basename|speaker|phonemes|raw_text`):

```bash
python synthesize.py \
    --mode batch \
    --source path/to/text_file.txt \
    --restore_step 30000 \
    -p config/kokoro/preprocess.yaml \
    -m config/kokoro/model.yaml \
    -t config/kokoro/train.yaml
```

#### Controllable Synthesis

Adjust pitch, energy, and duration:

```bash
python synthesize.py \
    --mode single \
    --text "こんにちは" \
    --restore_step 30000 \
    -p config/kokoro/preprocess.yaml \
    -m config/kokoro/model.yaml \
    -t config/kokoro/train.yaml \
    --pitch_control 1.2 \
    --energy_control 1.1 \
    --duration_control 0.9
```

- `--pitch_control`: Higher values = higher pitch (default: 1.0)
- `--energy_control`: Higher values = louder (default: 1.0)
- `--duration_control`: Higher values = slower speech (default: 1.0)

## Configuration

Configuration files are located in `config/kokoro/`:

- **`preprocess.yaml`**: Data preprocessing settings (audio parameters, feature extraction)
- **`model.yaml`**: Model architecture (transformer layers, variance predictors)
- **`train.yaml`**: Training hyperparameters (batch size, learning rate, steps)

See `config/README.md` for detailed explanations of configuration options.

### Key Configuration Parameters

- **`preprocess.yaml`**:
  - `sampling_rate`: Audio sample rate (22050 Hz)
  - `n_mel_channels`: Number of mel-spectrogram channels (80)
  - `pitch.feature`: `phoneme_level` or `frame_level`
  - `energy.feature`: `phoneme_level` or `frame_level`

- **`model.yaml`**:
  - `transformer.encoder_layer`: Number of encoder layers (4)
  - `transformer.decoder_layer`: Number of decoder layers (6)
  - `multi_speaker`: Enable multi-speaker TTS (True/False)
  - `vocoder.model`: Vocoder type (`HiFi-GAN` or `MelGAN`)

- **`train.yaml`**:
  - `optimizer.batch_size`: Training batch size (16)
  - `optimizer.grad_acc_step`: Gradient accumulation steps (1)
  - `step.total_step`: Total training steps (900000)

## Model Architecture

FastSpeech 2 consists of:

1. **Encoder**: Transformer encoder that processes phoneme sequences
2. **Variance Adaptor**: Predicts and adapts pitch, energy, and duration
3. **Decoder**: Transformer decoder that generates mel-spectrograms
4. **PostNet**: Convolutional post-processing network
5. **Speaker Embedding**: Optional speaker embeddings for multi-speaker TTS

The model predicts:
- Mel-spectrograms (main output)
- Pitch contours (phoneme or frame level)
- Energy values (phoneme or frame level)
- Duration labels (phoneme level)

## Requirements

Key dependencies (see `requirements.txt` for complete list):

- PyTorch
- NumPy, SciPy
- librosa (audio processing)
- pyopenjtalk (Japanese text-to-phoneme)
- Montreal Forced Aligner (MFA)
- HiFi-GAN vocoder
- TensorBoard (logging)
- PyYAML (configuration)

## Output

- **Checkpoints**: Saved in `output/ckpt/kokoro/` (every `save_step` steps)
- **Logs**: TensorBoard logs in `output/log/kokoro/`
- **Synthesized audio**: WAV files in `output/result/kokoro/`
- **Mel-spectrogram plots**: PNG files in `output/result/kokoro/`

## Notes

- This implementation uses **phoneme-level** pitch and energy features (as opposed to frame-level in the original paper) for more natural prosody
- Japanese text preprocessing uses **pyopenjtalk** for G2P conversion and maps to IPA phonemes
- The model supports **normalized** pitch and energy values with linear quantization
- HiFi-GAN universal vocoder is used for waveform generation

## License

See `LICENSE` file for details.

## References

- FastSpeech 2: Fast, Robust and Controllable Text to Speech (Ren et al., NeurIPS 2020)
- HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (Kong et al., NeurIPS 2020)
- Montreal Forced Aligner: https://montreal-forced-alignment.readthedocs.io/

## Acknowledgments

This implementation is adapted for Japanese TTS using the Kokoro dataset and includes custom preprocessing for Japanese phoneme representation.
