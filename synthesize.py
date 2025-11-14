<<<<<<< Updated upstream
=======
import argparse
>>>>>>> Stashed changes
import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence, symbols
import pyopenjtalk
from prepare_tg_accent import pp_symbols
from convert_label import openjtalk2julius

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

<<<<<<< Updated upstream
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
=======
_PUNCTUATION_TO_SP = {"。", "、", "，", "．", "！", "？", "…", "・", "「", "」", "『", "』"}

_MFA_G2P_CACHE = {}


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


def _mfa_g2p_word(word, model_path):
    cache_key = (model_path, word)
    if cache_key in _MFA_G2P_CACHE:
        return _MFA_G2P_CACHE[cache_key]

    if not shutil.which("mfa"):
        raise RuntimeError("Could not find the `mfa` executable in PATH.")

    model = Path(model_path)
    if not model.exists():
        raise RuntimeError(f"MFA G2P model not found at: {model_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "g2p_input.txt"
        output_path = tmpdir / "g2p_output.txt"
        input_path.write_text(word + "\n", encoding="utf-8")

        cmd = [
            "mfa",
            "g2p",
            "--no_progress_bar",
            str(model),
            str(input_path),
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
        for node in nodes:
            surface = (node.get("string") or node.get("orig") or "").strip()
            if not surface:
                continue

            if surface in _PUNCTUATION_TO_SP:
                phones.append("sp")
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
>>>>>>> Stashed changes

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def preprocess_japanese(text:str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phonemes , accents = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']
    return phonemes, accents



def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
    for batch in batchs:
        batch = to_device(batch, device)
        accents = None
        if use_accent:
            accents = batch[-1]
            batch = batch[:-1]
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                accents=accents
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
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "ja":
            phonemes, accents = preprocess_japanese(args.text)
            print(phonemes,accents)
            texts = np.array([[symbol_to_id[t] for t in phonemes]])
            if preprocess_config["preprocessing"]["accent"]["use_accent"]:
                accents = np.array([[accent_to_id[a] for a in accents]])
            else:
                accents = None

        text_lens = np.array([len(texts[0])])
        print(text_lens)
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens),accents)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)