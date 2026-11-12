from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from Manga109.translation import CTranslate2SugoiTranslator, find_sentencepiece_model


class TranslationTests(unittest.TestCase):
    def test_ct2_batch_tokenizes_source_and_decodes_target(self) -> None:
        translator = object.__new__(CTranslate2SugoiTranslator)
        translator.batch_size = 8
        translator.source_tokenizer = Mock()
        translator.source_tokenizer.encode.side_effect = lambda text, out_type: [f"src:{text}"]
        translator.target_tokenizer = Mock()
        translator.target_tokenizer.decode.side_effect = lambda tokens: " ".join(tokens)
        translator.translator = Mock()
        translator.translator.translate_batch.return_value = [
            SimpleNamespace(hypotheses=[["Hello", "there"]]),
            SimpleNamespace(hypotheses=[["Wait"]]),
        ]

        translated = translator.translate_batch(["こんにちは", "待って"])

        self.assertEqual(translated, ["Hello there", "Wait"])
        translator.translator.translate_batch.assert_called_once_with(
            [["src:こんにちは"], ["src:待って"]],
            beam_size=4,
            max_batch_size=8,
            max_decoding_length=256,
        )

    def test_ct2_accepts_string_hypotheses(self) -> None:
        translator = object.__new__(CTranslate2SugoiTranslator)
        translator.batch_size = 1
        translator.source_tokenizer = Mock()
        translator.source_tokenizer.encode.return_value = ["source"]
        translator.translator = Mock()
        translator.translator.translate_batch.return_value = [SimpleNamespace(hypotheses=["Hello"])]
        translator.target_tokenizer = Mock()

        self.assertEqual(translator.translate_batch(["text"]), ["Hello"])
        translator.target_tokenizer.decode.assert_not_called()

    def test_finds_nested_language_specific_sentencepiece_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            spm_dir = model_dir / "spm"
            spm_dir.mkdir()
            source = spm_dir / "spm.ja.nopretok.model"
            source.touch()
            target = spm_dir / "spm.en.nopretok.model"
            target.touch()
            self.assertEqual(find_sentencepiece_model(model_dir, ("spm.ja.nopretok.model",)), source)
            self.assertEqual(find_sentencepiece_model(model_dir, ("spm.en.nopretok.model",)), target)
            self.assertIsNone(find_sentencepiece_model(model_dir, ("missing.spm",)))


if __name__ == "__main__":
    unittest.main()
