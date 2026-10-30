"""Focused regressions for GroqCasters / GroqCastersApp.

No pretrained weights are loaded and no Groq/network calls are made. The TTS
model objects are built with ``__new__`` (plus ``torch.nn.Module.__init__``)
purely to satisfy ``isinstance`` checks and to exercise attribute fallbacks.
"""

import importlib
import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import groqcasters  # noqa: E402
from bark_model import BarkTTSModel  # noqa: E402
from f5_tts_model import F5TTSModel  # noqa: E402
from groqcasters import GroqCasters, GroqCastersApp  # noqa: E402


def _bare_bark_model():
    """BarkTTSModel instance without preload_models() or voice prompts."""
    model = BarkTTSModel.__new__(BarkTTSModel)
    torch.nn.Module.__init__(model)
    return model


def _bare_f5_model():
    """F5TTSModel instance without checkpoints, vocoder, or downloads."""
    model = F5TTSModel.__new__(F5TTSModel)
    torch.nn.Module.__init__(model)
    return model


def _casters_without_init(tts_model):
    """GroqCasters instance without GroqProvider or model construction."""
    casters = GroqCasters.__new__(GroqCasters)
    casters.tts_model = tts_model
    casters.app = GroqCastersApp(tts_model)
    return casters


class PodcastScriptRegressionTest(unittest.TestCase):
    def test_bark_model_alternates_speakers_without_attribute_error(self):
        tts_model = _bare_bark_model()
        self.assertFalse(hasattr(tts_model, "speaker_model_map"))

        casters = _casters_without_init(tts_model)
        dialogue = (
            "Mike: Welcome to the show.\n"
            "Mike: This line repeats the previous speaker.\n"
            "Rachel: Here is a response.\n"
            "Rachel: This one repeats as well.\n"
        )

        with mock.patch.object(
            casters, "_generate_outline", return_value="outline"
        ) as outline, mock.patch.object(
            casters, "_expand_outline", return_value="full script"
        ) as expand, mock.patch.object(
            casters, "_convert_to_dialogue", return_value=dialogue
        ) as convert, self.assertLogs(level="WARNING") as captured:
            result = casters.generate_podcast_script("some input text")

        outline.assert_called_once_with("some input text")
        expand.assert_called_once_with("outline")
        convert.assert_called_once_with("full script")
        self.assertIsNotNone(result)

        self.assertTrue(
            any("using default speakers" in message for message in captured.output),
            captured.output,
        )

        labels = [
            line.split(":", 1)[0].strip()
            for line in result.splitlines()
            if line.strip()
        ]
        self.assertEqual([label.lower() for label in labels],
                         ["mike", "rachel", "mike", "rachel"])
        self.assertEqual(labels, ["Mike", "Rachel", "Mike", "Rachel"])


class AudioFromScriptRegressionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.casters = _casters_without_init(_bare_bark_model())

    def test_mono_segments_are_written_as_2d_float32(self):
        segments = [
            np.linspace(-0.5, 0.5, 100, dtype=np.float64),
            np.linspace(0.5, -0.5, 150, dtype=np.float64),
            np.zeros(80, dtype=np.float64),
        ]
        script = "Mike: one\nRachel: two\nMike: three\n"

        with mock.patch.object(
            self.casters.app, "generate_audio", side_effect=segments
        ) as generate_audio, mock.patch(
            "groqcasters.write_wav"
        ) as write_wav:
            self.casters.generate_audio_from_script(script, self.tmp.name)

        self.assertEqual(generate_audio.call_count, 3)
        self.assertEqual(
            [call.kwargs for call in generate_audio.call_args_list],
            [
                {"text": "one", "speaker": "mike"},
                {"text": "two", "speaker": "rachel"},
                {"text": "three", "speaker": "mike"},
            ],
        )

        write_wav.assert_called_once()
        output_path, sample_rate, audio = write_wav.call_args.args
        self.assertEqual(output_path, os.path.join(self.tmp.name, "full_podcast.wav"))
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(
            audio.shape,
            (max(len(segment) for segment in segments) * len(segments), 1),
        )
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(np.all(audio >= -1.0) and np.all(audio <= 1.0))

    def test_mixed_mono_and_2d_segments_are_normalized(self):
        segments = [
            np.full(100, 0.5, dtype=np.float64),
            np.full((1, 60), 0.25, dtype=np.float32),
        ]
        script = "Mike: one\nRachel: two\n"

        with mock.patch.object(
            self.casters.app, "generate_audio", side_effect=segments
        ), mock.patch("groqcasters.write_wav") as write_wav:
            self.casters.generate_audio_from_script(script, self.tmp.name)

        write_wav.assert_called_once()
        _output_path, sample_rate, audio = write_wav.call_args.args
        self.assertEqual(sample_rate, 24000)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape, (200, 1))
        self.assertEqual(audio.dtype, np.float32)


class F5MissingReferenceRegressionTest(unittest.TestCase):
    def test_missing_reference_returns_none_logs_error_and_skips_infer(self):
        tts_model = _bare_f5_model()
        app = GroqCastersApp(tts_model, speaker_ref_map={})

        with mock.patch.object(
            F5TTSModel, "infer"
        ) as infer, self.assertLogs(level="ERROR") as captured:
            result = app.generate_audio("hello", "mike")

        self.assertIsNone(result)
        infer.assert_not_called()
        self.assertTrue(
            any(
                "No reference audio path configured for speaker 'mike'" in message
                for message in captured.output
            ),
            captured.output,
        )


class ImportRegressionTest(unittest.TestCase):
    def test_reload_emits_no_error_log(self):
        records = []
        handler = logging.Handler()
        handler.emit = records.append

        root = logging.getLogger()
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            reloaded = importlib.reload(groqcasters)
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)

        error_records = [r for r in records if r.levelno >= logging.ERROR]
        self.assertEqual(error_records, [], [r.getMessage() for r in error_records])
        # Reload recreates the classes; confirm the module still exposes them.
        self.assertTrue(callable(reloaded.GroqCasters))
        self.assertTrue(callable(reloaded.GroqCastersApp))


if __name__ == "__main__":
    unittest.main()
