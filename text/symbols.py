""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"

# Japanese MFA dictionary phoneme inventory (IPA-based).
_japanese_phonemes = [
    "a", "aː", "b", "bʲ", "bʲː", "bː", "c", "cː", "d", "dz", "dzː", "dʑ", "dʑː",
    "dʲ", "dʲː", "dː", "e", "eː", "h", "hː", "i", "iː", "i̥", "j", "k", "kː",
    "m", "mʲ", "mʲː", "mː", "n", "nː", "o", "oː", "p", "pʲ", "pʲː", "pː", "s",
    "sː", "t", "ts", "tsː", "tɕ", "tɕː", "tʲ", "tʲː", "tː", "v", "vʲ", "w",
    "wː", "z", "ç", "çː", "ŋ", "ɕ", "ɕː", "ɟ", "ɟː", "ɡ", "ɡː", "ɨ", "ɨː",
    "ɨ̥", "ɯ", "ɯː", "ɯ̥", "ɰ̃", "ɲ", "ɲː", "ɴ", "ɴː", "ɸ", "ɸʲ", "ɸʲː", "ɸː",
    "ɾ", "ɾʲ", "ɾʲː", "ɾː", "ʑ", "ʔ"
]


# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + _japanese_phonemes