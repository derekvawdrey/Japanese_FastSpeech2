import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

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


def read_lexicon(lex_path):
    """
    Load a pronunciation lexicon into a {word: [phones]} mapping.

    Julius-style dictionaries and MFA dictionaries include a bracketed reading
    token and may store pronunciation variants after a log-likelihood weight
    (e.g., '@-2.45') plus a katakana reading. We skip those auxiliary fields so
    only phoneme symbols remain. The loaded dictionary exposes both the original
    surface key (which may include a part-of-speech suffix) and a convenience key
    containing just the surface form before the first '+'.
    """
    lexicon = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 2:
                continue
            key = parts[0].lower()
            phones = []
            for token in parts[1:]:
                if token.startswith("[") and token.endswith("]"):
                    continue
                if token.startswith("@"):
                    continue
                try:
                    float(token)
                    continue
                except ValueError:
                    pass
                if token == "":
                    continue
                phones.append(token)
            if not phones:
                continue

            if key not in lexicon:
                lexicon[key] = phones

            surface = key.split("+", 1)[0]
            if surface and surface not in lexicon:
                lexicon[surface] = phones
    return lexicon


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
    "cl": "sp",
    "pau": "sp",
    "sil": "sp",
}

_PUNCTUATION_TO_SP = {"。", "、", "，", "．", "！", "？", "…", "・", "「", "」", "『", "』"}

_MFA_G2P_CACHE = {}


def _mfa_g2p_batch(words, model_path):
    """Batch process multiple words through MFA G2P for better performance."""
    if not words:
        return {}
    
    if not shutil.which("mfa"):
        raise RuntimeError("Could not find the `mfa` executable in PATH.")
    
    # Check if model_path is a direct path that exists
    model = Path(model_path)
    if model.exists() and model.is_dir():
        # It's a valid directory path
        model_arg = str(model)
    else:
        # It's likely a model name (e.g., "japanese_mfa"), try to find it or use it directly
        found_model = _find_mfa_g2p_model(model_path)
        if found_model:
            # found_model can be either a path or the model name (if registered with MFA)
            model_arg = found_model
        else:
            # Last resort: try using it as a model name directly (MFA might still recognize it)
            # or it might be a path that we can't verify but MFA can use
            model_arg = model_path
    
    # Filter out words that are already cached
    uncached_words = []
    results = {}
    for word in words:
        cache_key = (model_path, word)
        if cache_key in _MFA_G2P_CACHE:
            results[word] = _MFA_G2P_CACHE[cache_key]
        else:
            uncached_words.append(word)
    
    if not uncached_words:
        return results
    
    # Batch process uncached words
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "g2p_input.txt"
        output_path = tmpdir / "g2p_output.txt"
        input_path.write_text("\n".join(uncached_words), encoding="utf-8")
        
        cmd = [
            "mfa",
            "g2p",
            str(input_path),
            model_arg,
            str(output_path),
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        
        if result.returncode != 0:
            # Clean up UTF drawing characters from error messages
            stderr_clean = ""
            if result.stderr:
                stderr_clean = "".join(
                    c for c in result.stderr if ord(c) < 0x2500 or ord(c) > 0x25FF
                ).strip()
            stdout_clean = ""
            if result.stdout:
                stdout_clean = "".join(
                    c for c in result.stdout if ord(c) < 0x2500 or ord(c) > 0x25FF
                ).strip()
            message = stderr_clean or stdout_clean or "Unknown MFA error"
            raise RuntimeError(message)
        
        if not output_path.exists():
            raise RuntimeError("MFA G2P did not produce an output file.")
        
        # Parse output - each line should be: word\tpronunciation
        output_lines = [
            line.strip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        
        # Map words to pronunciations
        for line in output_lines:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                word = parts[0].strip()
                pronunciation = parts[1].strip()
                tokens = pronunciation.split() if pronunciation else []
                cache_key = (model_path, word)
                _MFA_G2P_CACHE[cache_key] = tokens
                results[word] = tokens
    
    return results


def _map_openjtalk_tokens(tokens):
    mapped = []
    for token in tokens:
        ipa = _OPENJTALK_TO_IPA.get(token)
        if ipa is None:
            ipa = token if token and token[0] in {"@", "ɕ", "ɟ", "ɲ"} else "sp"
        mapped.append(ipa)
    return mapped


def _lexicon_lookup(lexicon, key):
    if not key:
        return None
    if key in lexicon:
        return lexicon[key]
    lowered = key.lower()
    if lowered in lexicon:
        return lexicon[lowered]
    return None


def _find_mfa_g2p_model(model_name):
    """Check if an MFA G2P model exists and return the model identifier (name or path)."""
    # Common locations for MFA models
    possible_locations = [
        Path.home() / "Documents" / "MFA" / "pretrained_models" / "g2p" / model_name,
        Path.home() / ".local" / "share" / "montreal-forced-aligner" / "pretrained_models" / "g2p" / model_name,
    ]
    
    # First check if a directory path exists
    for location in possible_locations:
        if location.exists() and location.is_dir():
            return str(location)
    
    # If not found as a directory, check if MFA recognizes it as a registered model
    try:
        result = subprocess.run(
            ["mfa", "model", "list", "g2p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
        )
        if result.returncode == 0 and model_name in result.stdout:
            # Model is registered with MFA, use the name directly
            return model_name
    except Exception:
        pass
    
    return None


def _mfa_g2p_word(word, model_path):
    cache_key = (model_path, word)
    if cache_key in _MFA_G2P_CACHE:
        return _MFA_G2P_CACHE[cache_key]

    if not shutil.which("mfa"):
        raise RuntimeError("Could not find the `mfa` executable in PATH.")

    # Check if model_path is a direct path that exists
    model = Path(model_path)
    if model.exists() and model.is_dir():
        # It's a valid directory path
        model_arg = str(model)
    else:
        # It's likely a model name (e.g., "japanese_mfa"), try to find it or use it directly
        found_model = _find_mfa_g2p_model(model_path)
        if found_model:
            # found_model can be either a path or the model name (if registered with MFA)
            model_arg = found_model
        else:
            # Last resort: try using it as a model name directly (MFA might still recognize it)
            # or it might be a path that we can't verify but MFA can use
            model_arg = model_path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "g2p_input.txt"
        output_path = tmpdir / "g2p_output.txt"
        input_path.write_text(word + "\n", encoding="utf-8")

        cmd = [
            "mfa",
            "g2p",
            str(input_path),
            model_arg,
            str(output_path),
        ]
        # Capture output but strip UTF drawing characters (progress bars)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',  # Replace undecodable bytes instead of failing
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            # Clean up UTF drawing characters from error messages
            stderr_clean = ""
            if result.stderr:
                stderr_clean = "".join(
                    c for c in result.stderr if ord(c) < 0x2500 or ord(c) > 0x25FF
                ).strip()
            stdout_clean = ""
            if result.stdout:
                stdout_clean = "".join(
                    c for c in result.stdout if ord(c) < 0x2500 or ord(c) > 0x25FF
                ).strip()
            message = stderr_clean or stdout_clean or "Unknown MFA error"
            raise RuntimeError(message)

        if not output_path.exists():
            raise RuntimeError("MFA G2P did not produce an output file.")

        lines = [
            line.strip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    tokens = []
    if lines:
        parts = lines[0].split("\t")
        pronunciation = parts[-1] if parts else ""
        tokens = pronunciation.split()

    _MFA_G2P_CACHE[cache_key] = tokens
    return tokens


def preprocess_japanese(text, preprocess_config):
    lexicon_path = preprocess_config["path"].get("lexicon_path")
    lexicon = read_lexicon(lexicon_path) if lexicon_path else {}
    mfa_model_path = preprocess_config["preprocessing"]["text"].get("mfa_g2p_model_path")

    phones = []
    if lexicon or mfa_model_path:
        nodes = pyopenjtalk.run_frontend(text)
        surfaces = []
        node_list = []
        for node in nodes:
            surface = (node.get("string") or node.get("orig") or "").strip()
            if not surface:
                continue
            
            if surface in _PUNCTUATION_TO_SP:
                phones.append("sp")
                continue
            
            surfaces.append(surface)
            node_list.append(node)
        
        # Batch process all words through MFA G2P if available
        mfa_results = {}
        if mfa_model_path and surfaces:
            try:
                mfa_results = _mfa_g2p_batch(surfaces, mfa_model_path)
            except RuntimeError as err:
                print(f"[WARN] MFA G2P batch processing failed: {err}")
        
        # Process each node with results
        for surface, node in zip(surfaces, node_list):
            # Try MFA G2P first
            if surface in mfa_results:
                mfa_tokens = mfa_results[surface]
                if mfa_tokens:
                    phones.extend(mfa_tokens)
                    continue
            
            # Fall back to lexicon lookup if MFA G2P not available or failed
            entry = _lexicon_lookup(lexicon, surface) if lexicon else None
            if not entry and lexicon:
                for candidate in filter(None, (node.get("orig"), node.get("read"), node.get("pron"))):
                    entry = _lexicon_lookup(lexicon, candidate)
                    if entry:
                        break
            if entry:
                phones.extend(entry)
                continue

            # Final fallback to pyopenjtalk
            fallback_tokens = pyopenjtalk.g2p(surface, join=False)
            phones.extend(_map_openjtalk_tokens(fallback_tokens))
    else:
        fallback_tokens = pyopenjtalk.g2p(text, join=False)
        phones.extend(_map_openjtalk_tokens(fallback_tokens))

    phones = [p for p in phones if p]
    if not phones:
        phones = ["sp"]

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
