""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"

# Julius-style Japanese phoneme inventory extracted from the BCCWJ 60k lexicon.
_japanese_phonemes = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "silB",
    "silE",
    "sp",
    "t",
    "ts",
    "u",
    "u:",
    "w",
    "y",
    "z",
]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + _japanese_phonemes