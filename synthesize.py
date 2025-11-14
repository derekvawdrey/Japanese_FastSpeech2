import argparse
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


_MFA_G2P_CACHE = {}


def _mfa_g2p_word(word, model_path):
    cache_key = (model_path, word)
    if cache_key in _MFA_G2P_CACHE:
        return _MFA_G2P_CACHE[cache_key]

    if not shutil.which("mfa"):
        raise RuntimeError("Could not find the `mfa` executable in PATH.")

    # Handle model path - could be a name, directory, or zip file
    model = Path(model_path)
    zip_tmpdir = None
    
    # If it's a zip file, extract it to a temp directory
    if model.suffix == ".zip" and model.exists():
        import zipfile
        zip_tmpdir = tempfile.mkdtemp()
        zip_tmpdir = Path(zip_tmpdir)
        with zipfile.ZipFile(model, 'r') as zip_ref:
            zip_ref.extractall(zip_tmpdir)
        # Find the model directory inside (usually the zip contains one folder)
        extracted_dirs = [d for d in zip_tmpdir.iterdir() if d.is_dir()]
        if extracted_dirs:
            model_path = str(extracted_dirs[0])
        else:
            model_path = str(zip_tmpdir)
    elif model.exists() and model.is_dir():
        # It's a directory, use it directly
        model_path = str(model)
    elif model.exists() and model.is_file() and model.suffix != ".zip":
        # It's a file but not a zip, might be a model file
        model_path = str(model)
    else:
        # Assume it's a model name that MFA knows about (e.g., "japanese_mfa")
        # MFA will look it up in its model cache, or it might be a path to a zip
        # that we need to check
        if not model.exists():
            # Try as model name first - MFA will handle it
            model_path = str(model_path)
        else:
            model_path = str(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "g2p_input.txt"
        output_path = tmpdir / "g2p_output.txt"
        input_path.write_text(word + "\n", encoding="utf-8")

        cmd = [
            "mfa",
            "g2p",
            str(input_path),
            model_path,
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "Unknown MFA error"
            raise RuntimeError(f"MFA G2P command failed: {message}\nCommand: {' '.join(cmd)}")

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

    # Clean up extracted zip directory if we created one
    if zip_tmpdir and zip_tmpdir.exists():
        import shutil as shutil_cleanup
        shutil_cleanup.rmtree(zip_tmpdir, ignore_errors=True)

    _MFA_G2P_CACHE[cache_key] = tokens
    return tokens


def preprocess_japanese(text, preprocess_config):
    """
    Convert Japanese text to phonemes using MFA G2P (same as training).
    Segments text into words and calls MFA G2P on each word.
    """
    mfa_model_path = preprocess_config["preprocessing"]["text"].get("mfa_g2p_model_path")
    
    if not mfa_model_path:
        raise RuntimeError("MFA G2P model path not found in config. Please set preprocessing.text.mfa_g2p_model_path")

    # Punctuation that should become spn
    punctuation_to_spn = {"。", "、", "，", "．", "！", "？", "…", "・", "「", "」", "『", "』"}
    
    phones = []
    
    # Use pyopenjtalk to segment text into words
    nodes = pyopenjtalk.run_frontend(text)
    
    # Collect all nodes first
    node_list = []
    for node in nodes:
        surface = (node.get("string") or node.get("orig") or "").strip()
        if not surface:
            continue
        node_list.append(node)
    
    i = 0
    while i < len(node_list):
        node = node_list[i]
        surface = (node.get("string") or node.get("orig") or "").strip()
        
        # Handle punctuation as silence (but not at the very end)
        if surface in punctuation_to_spn:
            # Only add spn if there are more nodes after this (not the last element)
            if i < len(node_list) - 1:
                phones.append("spn")
            i += 1
            continue
        
        # Try to combine with next word if it's a single character particle/auxiliary
        # This helps with compounds like "何か" (nanka)
        # But be careful - only combine very short particles (1-2 chars) that are likely auxiliaries
        combined_surface = surface
        combined_reading = (node.get("read") or node.get("pron") or surface).strip()
        j = i + 1
        
        # Only combine with single-character particles/auxiliaries (not longer words)
        # Common particles: か, が, の, を, に, は, へ, と, で, ば, など
        single_char_particles = {"か", "が", "の", "を", "に", "は", "へ", "と", "で", "ば", "も", "や", "から", "まで", "より"}
        
        while j < len(node_list):
            next_node = node_list[j]
            next_surface = (next_node.get("string") or next_node.get("orig") or "").strip()
            
            # Stop if we hit punctuation
            if next_surface in punctuation_to_spn:
                break
            
            # Only combine if it's a single character AND a known particle
            # Don't combine longer words as this can break compounds
            if len(next_surface) == 1 and next_surface in single_char_particles:
                combined_surface += next_surface
                next_reading = (next_node.get("read") or next_node.get("pron") or next_surface).strip()
                combined_reading += next_reading
                j += 1
            else:
                break
        
        # Try MFA G2P in order of preference:
        # 1. Combined surface form (e.g., "何か") - MFA might handle compounds better
        # 2. Combined reading (e.g., "ナニカ") 
        # 3. Individual word reading
        word_phones = None
        
        # First try combined surface form (works better for compounds)
        if len(combined_surface) > len(surface):
            try:
                word_phones = _mfa_g2p_word(combined_surface, mfa_model_path)
            except RuntimeError:
                pass
        
        # If that didn't work, try combined reading
        if not word_phones and combined_reading and len(combined_reading) > len((node.get("read") or node.get("pron") or surface).strip()):
            try:
                word_phones = _mfa_g2p_word(combined_reading, mfa_model_path)
            except RuntimeError:
                pass
        
        # If still no result, try individual word reading
        if not word_phones:
            individual_reading = (node.get("read") or node.get("pron") or surface).strip()
            try:
                word_phones = _mfa_g2p_word(individual_reading, mfa_model_path)
            except RuntimeError:
                pass
        
        # Last resort: try surface form
        if not word_phones:
            try:
                word_phones = _mfa_g2p_word(surface, mfa_model_path)
            except RuntimeError as e:
                raise RuntimeError(f"MFA G2P failed for word '{combined_surface}' (reading: '{combined_reading}'): {e}")
        
        if word_phones:
            phones.extend(word_phones)
        
        # Move to next unprocessed node
        i = j

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
