import argparse
import re

import numpy as np
import pyopenjtalk
import torch
import yaml
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping from OpenJTalk phonemes to IPA (matching training data format)
_OPENJTALK_TO_IPA = {
    "a": "a",
    "i": "i",
    "u": "ɯ",
    "U": "ɯ̥",
    "e": "e",
    "o": "o",
    "N": "ɴ",
    "m": "m",
    "n": "n",
    "p": "p",
    "b": "b",
    "by": "bʲ",
    "f": "ɸ",
    "v": "v",
    "t": "t",
    "ts": "ts",
    "d": "d",
    "s": "s",
    "sh": "ɕ",
    "z": "z",
    "j": "dʑ",
    "k": "k",
    "ky": "c",
    "g": "ɡ",
    "gy": "ɟ",
    "h": "h",
    "hy": "ç",
    "r": "ɾ",
    "ry": "ɾʲ",
    "w": "w",
    "y": "j",
    "ch": "tɕ",
    # "cl" is handled specially in post-processing (gemination marker)
    "spn": "spn",
    "pau": "pau", # This is what the synthesizer uses for pauses
}


def _map_openjtalk_to_ipa(tokens):
    """Convert OpenJTalk phonemes to IPA format matching training data."""
    mapped = []
    for token in tokens:
        ipa = _OPENJTALK_TO_IPA.get(token)
        if ipa is None:
            # If token starts with special IPA characters, keep it as is
            if token and token[0] in {"@", "ɕ", "ɟ", "ɲ"}:
                ipa = token
            else:
                # Unknown token, keep as is for now (will handle "cl" specially)
                ipa = token
        mapped.append(ipa)
    
    # Post-process: handle gemination (small っ) - "cl" followed by consonant becomes long consonant
    # e.g., "cl" + "t" -> "tː", "cl" + "ch" -> "tɕː"
    # Map both OpenJTalk tokens and IPA tokens to long consonants
    gemination_map = {
        # OpenJTalk tokens
        "t": "tː", "k": "kː", "p": "pː", "s": "sː", "ts": "tsː",
        "ch": "tɕː", "sh": "ɕː", "j": "dʑː", "d": "dː", "b": "bː",
        "g": "ɡː", "z": "zː", "r": "ɾː", "m": "mː", "n": "nː",
        "h": "hː", "f": "ɸː", "v": "vː",
        # IPA tokens (already converted)
        "tɕ": "tɕː", "ɕ": "ɕː", "dʑ": "dʑː", "ts": "tsː",
        "ɾ": "ɾː", "ɸ": "ɸː", "ɡ": "ɡː"
    }
    
    result = []
    i = 0
    while i < len(mapped):
        current = mapped[i]
        
        # Handle "cl" (closure/gemination marker) - convert to long consonant
        if current == "cl" and i + 1 < len(mapped):
            next_token = mapped[i + 1]
            # Map the next consonant to its long form (works for both OpenJTalk and IPA)
            long_consonant = gemination_map.get(next_token)
            if long_consonant:
                result.append(long_consonant)
                i += 2  # Skip both "cl" and the consonant
            else:
                # If we can't map it, use glottal stop ʔ (common in training data for gemination)
                result.append("ʔ")
                i += 1
        # Handle consecutive identical vowels -> long vowels
        elif current in {"a", "i", "u", "e", "o", "ɯ", "ɨ"} and i + 1 < len(mapped) and mapped[i + 1] == current:
            long_vowel_map = {
                "a": "aː", "i": "iː", "u": "uː", "e": "eː", "o": "oː",
                "ɯ": "ɯː", "ɨ": "ɨː"
            }
            long_vowel = long_vowel_map.get(current, current + "ː")
            result.append(long_vowel)
            i += 2  # Skip both vowels
        else:
            result.append(current)
            i += 1
    
    return result


def preprocess_japanese(text, preprocess_config):
    """
    Convert Japanese text to phonemes using pyopenjtalk G2P.
    """

    phones = []
    
    # Use pyopenjtalk to get phonemes for the full text
    # pyopenjtalk.g2p returns space-separated phonemes
    openjtalk_phones = pyopenjtalk.g2p(text, join=False)
    
    # Map OpenJTalk phonemes to IPA format
    phones = _map_openjtalk_to_ipa(openjtalk_phones)

    if phones and phones[-1] == "spn":
        phones = phones[:-1]

    if not phones:
        phones = ["spn"]

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "ja":
            texts = np.array([preprocess_japanese(args.text, preprocess_config)])
        else:
            raise ValueError(f"Unsupported language: {preprocess_config['preprocessing']['text']['language']}")
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
