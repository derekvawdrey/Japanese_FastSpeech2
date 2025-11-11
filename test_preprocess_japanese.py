import unittest

import numpy as np
import pyopenjtalk

from synthesize import preprocess_japanese
from text import text_to_sequence


def _build_minimal_config():
    return {
        "path": {},
        "preprocessing": {
            "text": {
                "text_cleaners": ["basic_cleaners"],
            }
        },
    }


class PreprocessJapaneseTests(unittest.TestCase):
    def test_konnichiwa_sequence_matches_pyopenjtalk_tokens(self):
        text = "お元気ですか"
        config = _build_minimal_config()

        expected_tokens = [
            "sp" if token in {"pau", "sil"} else token
            for token in pyopenjtalk.g2p(text).split(" ")
            if token
        ]
        expected_phones = "{" + " ".join(expected_tokens) + "}"
        expected_sequence = np.array(
            text_to_sequence(
                expected_phones, config["preprocessing"]["text"]["text_cleaners"]
            )
        )

        actual_sequence = preprocess_japanese(text, config)

        np.testing.assert_array_equal(actual_sequence, expected_sequence)


if __name__ == "__main__":
    unittest.main()

