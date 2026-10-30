import os
import sys
import types
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class _FakeF5TTS:
    """Recorded stand-in for the official f5_tts.api.F5TTS class."""

    instances = []
    infer_calls = []
    infer_result = (np.array([0.1, 0.2, 0.3]), 24000, None)

    def __init__(self, model=None, ckpt_file=None, vocab_file=None, vocoder_local_path=None, device=None):
        self.init_kwargs = dict(
            model=model, ckpt_file=ckpt_file, vocab_file=vocab_file,
            vocoder_local_path=vocoder_local_path, device=device,
        )
        _FakeF5TTS.instances.append(self)

    def infer(self, ref_file=None, ref_text=None, gen_text=None):
        _FakeF5TTS.infer_calls.append(dict(ref_file=ref_file, ref_text=ref_text, gen_text=gen_text))
        return _FakeF5TTS.infer_result


from f5_tts_model import F5TTSModel  # noqa: E402
import f5_tts_model  # noqa: E402


class F5TTSModelTest(unittest.TestCase):
    def setUp(self):
        _FakeF5TTS.instances.clear()
        _FakeF5TTS.infer_calls.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["F5_TTS_MODEL_PATH"] = self.tmp.name
        self.addCleanup(os.environ.pop, "F5_TTS_MODEL_PATH", None)
        self.ckpt_dir = os.path.join(self.tmp.name, "F5TTS_Base")
        os.makedirs(self.ckpt_dir)
        self.ckpt = os.path.join(self.ckpt_dir, "model_1200000.pt")
        with open(self.ckpt, "wb") as f:
            f.write(b"fake-local-checkpoint")
        self.vocab = os.path.join(self.tmp.name, "vocab.txt")
        with open(self.vocab, "w") as f:
            f.write("fake-vocab-char-map")
        os.environ["F5_TTS_REF_TEXT_MIKE"] = "reference transcript"
        self.addCleanup(os.environ.pop, "F5_TTS_REF_TEXT_MIKE", None)
        self.ref_audio = os.path.join(self.tmp.name, "ref.wav")
        with open(self.ref_audio, "wb") as f:
            f.write(b"fake-audio")
        self.vocoder_dir = os.path.join(self.tmp.name, "vocoder")
        os.makedirs(self.vocoder_dir)
        with open(os.path.join(self.vocoder_dir, "config.yaml"), "w") as f:
            f.write("fake-vocoder-config")
        with open(os.path.join(self.vocoder_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"fake-vocoder-weights")
        os.environ["F5_TTS_VOCODER_PATH"] = self.vocoder_dir
        self.addCleanup(os.environ.pop, "F5_TTS_VOCODER_PATH", None)

    def _make(self, **kwargs):
        with mock.patch.object(f5_tts_model, "F5TTS", _FakeF5TTS):
            return F5TTSModel(**kwargs)

    def test_loads_via_official_api_with_local_ckpt(self):
        model = self._make(
            speaker_model_map={"mike": "F5TTS_Base"},
            vocab_char_map_path=self.vocab,
        )
        self.assertEqual(len(_FakeF5TTS.instances), 1)
        kwargs = _FakeF5TTS.instances[0].init_kwargs
        self.assertEqual(kwargs["model"], "F5TTS_Base")
        self.assertTrue(os.path.isfile(kwargs["ckpt_file"]))
        self.assertEqual(kwargs["ckpt_file"], self.ckpt)
        self.assertEqual(kwargs["vocab_file"], self.vocab)
        self.assertEqual(kwargs["vocoder_local_path"], self.vocoder_dir)
        self.assertTrue(os.path.isdir(kwargs["vocoder_local_path"]))
        self.assertTrue(os.path.isfile(os.path.join(kwargs["vocoder_local_path"], "config.yaml")))
        self.assertTrue(os.path.isfile(os.path.join(kwargs["vocoder_local_path"], "pytorch_model.bin")))
        self.assertIn("mike", model.tts_models)

    def test_infer_routes_to_official_api(self):
        model = self._make(
            speaker_model_map={"mike": "F5TTS_Base"},
            vocab_char_map_path=self.vocab,
        )
        _FakeF5TTS.infer_result = (np.array([0.5]), 24000, None)
        out = model.infer("hello", "mike", ref_audio_path=self.ref_audio)
        self.assertTrue(np.array_equal(out, np.array([0.5])))
        call = _FakeF5TTS.infer_calls[-1]
        self.assertEqual(call["ref_file"], self.ref_audio)
        self.assertEqual(call["ref_text"], "reference transcript")
        self.assertEqual(call["gen_text"], "hello")

    def test_missing_checkpoint_error_no_download(self):
        os.remove(self.ckpt)
        with self.assertRaises(FileNotFoundError):
            self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        self.assertEqual(_FakeF5TTS.instances, [])

    def test_vocoder_env_var_required_fails_before_construction(self):
        os.environ.pop("F5_TTS_VOCODER_PATH")
        try:
            with self.assertRaises(ValueError):
                self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        finally:
            os.environ["F5_TTS_VOCODER_PATH"] = self.vocoder_dir
        self.assertEqual(_FakeF5TTS.instances, [])

    def test_vocoder_missing_path_fails_before_construction(self):
        os.environ["F5_TTS_VOCODER_PATH"] = os.path.join(self.tmp.name, "absent-vocoder")
        with self.assertRaises(FileNotFoundError):
            self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        self.assertEqual(_FakeF5TTS.instances, [])

    def test_infer_requires_existing_ref_audio(self):
        model = self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        with self.assertRaises(FileNotFoundError):
            model.infer("hello", "mike", ref_audio_path=os.path.join(self.tmp.name, "absent.wav"))
        self.assertEqual(_FakeF5TTS.infer_calls, [])

    def test_infer_requires_ref_text_env_not_asr(self):
        model = self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        os.environ.pop("F5_TTS_REF_TEXT_MIKE")
        try:
            with self.assertRaises(ValueError):
                model.infer("hello", "mike", ref_audio_path=self.ref_audio)
        finally:
            os.environ["F5_TTS_REF_TEXT_MIKE"] = "reference transcript"
        self.assertEqual(_FakeF5TTS.infer_calls, [])

    def test_unknown_speaker(self):
        model = self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        with self.assertRaises(ValueError):
            model.infer("hello", "nobody")

    def test_vocab_path_required(self):
        with self.assertRaises(ValueError):
            self._make()

    def test_preprocess_text_passthrough(self):
        model = self._make(speaker_model_map={"mike": "F5TTS_Base"}, vocab_char_map_path=self.vocab)
        self.assertEqual(model.preprocess_text("hi", "mike"), "hi")
        self.assertEqual(model.preprocess_text("hi"), "hi")


if __name__ == "__main__":
    unittest.main()
