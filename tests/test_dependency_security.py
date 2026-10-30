import re
import unittest
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements.txt"

_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_][A-Za-z0-9._-]*)(\[.*\])?\s*(.*)$")


def _parse_requirements():
    reqs = {}
    for raw in REQUIREMENTS_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        name, _, spec = match.groups()
        reqs[name.lower()] = spec.strip()
    return reqs


class DependencySecurityTest(unittest.TestCase):
    def setUp(self):
        self.reqs = _parse_requirements()

    def test_requirements_file_exists(self):
        self.assertTrue(REQUIREMENTS_FILE.is_file())

    def test_torch_pinned_to_2_6_0(self):
        self.assertEqual(self.reqs.get("torch"), "==2.6.0")

    def test_torchaudio_pinned_to_2_6_0(self):
        self.assertEqual(self.reqs.get("torchaudio"), "==2.6.0")

    def test_numpy_bounded_2_0_to_lt_2_3(self):
        spec = self.reqs.get("numpy")
        self.assertEqual(spec, ">=2.0,<2.3")

    def test_librosa_bounded_0_11_to_lt_0_12(self):
        spec = self.reqs.get("librosa")
        self.assertEqual(spec, ">=0.11,<0.12")

    def test_soundfile_bounded_0_13_to_lt_0_14(self):
        spec = self.reqs.get("soundfile")
        self.assertEqual(spec, ">=0.13,<0.14")

    def test_f5_tts_pinned_1_1_22(self):
        self.assertEqual(self.reqs.get("f5-tts"), "==1.1.22")

    def test_pyarrow_bounded_17_to_lt_21(self):
        spec = self.reqs.get("pyarrow")
        self.assertIsNotNone(spec)
        self.assertIn(">=17", spec)
        self.assertIn("<21", spec)

    def test_langchain_bounded_0_3_1_to_lt_1(self):
        spec = self.reqs.get("langchain")
        self.assertIsNotNone(spec)
        self.assertIn(">=0.3.1", spec)
        self.assertTrue(spec.endswith("<1"))

    def test_langchain_community_bounded_0_3_1_to_lt_0_4(self):
        spec = self.reqs.get("langchain-community")
        self.assertEqual(spec, ">=0.3.1,<0.4")

    def test_transformers_secure_floor(self):
        self.assertEqual(self.reqs.get("transformers"), ">=4.36.0,<5.0")

    def test_suno_bark_pinned_0_1_5(self):
        self.assertEqual(self.reqs.get("suno-bark"), "==0.1.5")
        self.assertNotIn("bark", self.reqs)

    def test_all_packages_resolved_with_specs(self):
        for name in ("torch", "torchaudio", "numpy", "transformers", "suno-bark"):
            self.assertIn(name, self.reqs)


if __name__ == "__main__":
    unittest.main()
