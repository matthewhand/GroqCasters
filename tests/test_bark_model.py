import sys
import types
import unittest
from unittest import mock

import numpy as _np  # noqa: F401  (helps IDE resolve local path when numpy absent)
import numpy as np

from bark_model import (  # noqa: E402
    BarkTTSModel,
    MALE_VOICE_PRESET,
    FEMALE_VOICE_PRESET,
)
from tts_model import TTSModel  # noqa: E402


class BarkTTSModelConstructorTest(unittest.TestCase):
    def _make_model(self):
        with mock.patch("bark_model.preload_models") as preload, \
             mock.patch("bark_model.BarkTTSModel._create_voice_prompt", side_effect=[None, "hist"]) as cvp:
            model = BarkTTSModel()
        preload.assert_called_once()
        return model, cvp

    def test_is_abstract_tts_model(self):
        self.assertIsInstance(BarkTTSModel, type)
        self.assertTrue(issubclass(BarkTTSModel, TTSModel))

    def test_constructs_without_weights(self):
        model, cvp = self._make_model()
        self.assertIsNotNone(model)
        self.assertEqual(cvp.call_count, 2)
        cvp.assert_any_call("null")
        cvp.assert_any_call("bark_voice_samples/female.wav")

    def test_preprocess_text_passthrough(self):
        model, _ = self._make_model()
        for text in ("Hello world", "", "Ünïcödé 拼音"):
            self.assertEqual(model.preprocess_text(text), text)

    def test_infer_routes_to_male_voice(self):
        model, _ = self._make_model()
        with mock.patch("bark_model.generate_text_semantic", return_value="sem") as gen, \
             mock.patch("bark_model.semantic_to_waveform", return_value=np.zeros(3)) as stw:
            out = model.infer("hi")
        gen.assert_called_once_with("hi", history_prompt=MALE_VOICE_PRESET, temp=0.7)
        stw.assert_called_once_with("sem")
        self.assertEqual(list(out), [0, 0, 0])

    def test_infer_routes_to_female_voice(self):
        model, _ = self._make_model()
        with mock.patch("bark_model.generate_text_semantic", return_value="sem") as gen, \
             mock.patch("bark_model.semantic_to_waveform", return_value=np.zeros(3)):
            model.infer("hello", speaker="rachel")
        gen.assert_called_once_with("hello", history_prompt="hist", temp=0.7)

    def test_presets_kept(self):
        self.assertEqual(MALE_VOICE_PRESET, "v2/en_speaker_6")
        self.assertEqual(FEMALE_VOICE_PRESET, "v2/en_speaker_9")


if __name__ == "__main__":
    unittest.main()
