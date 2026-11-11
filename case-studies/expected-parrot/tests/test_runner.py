import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from parrot_baseline import runner
from parrot_baseline.engines import DemoEngine, EngineError
from parrot_baseline.runner import ARTIFACTS, run_study
from parrot_baseline.schemas import Concept, StudyManifest

CONCEPT = Concept("Lasagna Cola", "A savory cola with tomato and basil notes.")
NOW = datetime(2025, 5, 6, 7, 8, 9, tzinfo=timezone.utc)


class RecordingEngine:
    name, model = "recording", "rec-1"

    def __init__(self, reply=None):
        self.prompts, self.reply = [], reply

    def generate(self, concept, persona, questions, prompt, seed):
        self.prompts.append(prompt)
        if self.reply is not None:
            return self.reply
        return DemoEngine().generate(concept, persona, questions, prompt, seed)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "nested" / "study"

    def read(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def test_persists_four_artifacts(self):
        result = run_study(CONCEPT, DemoEngine(), self.out, seed=3, now=NOW)
        self.assertEqual(ARTIFACTS, ("manifest.json", "responses.json", "summary.json", "report.html"))
        for name in ARTIFACTS:
            self.assertTrue((self.out / name).is_file(), name)
        self.assertEqual(set(result.paths), set(ARTIFACTS))

    def test_manifest_contents_round_trip(self):
        run_study(CONCEPT, DemoEngine(), self.out, panel_size=12, seed=3, now=NOW)
        manifest = StudyManifest.from_dict(self.read("manifest.json"))
        self.assertEqual((manifest.engine, manifest.model, manifest.seed), ("demo", "demo-rules-v1", 3))
        self.assertEqual(manifest.engine_config, {"version": "demo-rules-v1"})
        self.assertIn("python", manifest.runtime)
        self.assertIn("parrot_baseline", manifest.runtime)
        self.assertEqual(manifest.created_at, "2025-05-06T07:08:09+00:00")
        self.assertEqual(manifest.concept, CONCEPT)
        self.assertEqual(len(manifest.panel), 12)
        self.assertEqual(len(manifest.questions), 7)
        self.assertEqual(set(manifest.prompts), {p.persona_id for p in manifest.panel})

    def test_responses_keep_raw_output(self):
        run_study(CONCEPT, DemoEngine(), self.out, now=NOW)
        responses = self.read("responses.json")
        self.assertEqual(len(responses), 12)
        self.assertIn('"appeal"', responses[0]["raw_output"])
        self.assertIsNone(responses[0]["error"])

    def test_summary_has_means_distributions_warnings_limitations(self):
        run_study(CONCEPT, DemoEngine(), self.out, now=NOW)
        summary = self.read("summary.json")
        for key in ("means", "distributions", "quotes", "warnings", "limitations", "counts"):
            self.assertIn(key, summary)
        self.assertEqual(set(summary["means"]["by_segment"]), {"adventurous_foodie", "mainstream_drinker", "skeptical_health_conscious"})

    def test_every_prompt_sent_contains_full_concept(self):
        engine = RecordingEngine()
        run_study(CONCEPT, engine, self.out, now=NOW)
        self.assertEqual(len(engine.prompts), 12)
        for prompt in engine.prompts:
            self.assertIn(CONCEPT.description, prompt)
            self.assertIn("improvement", prompt)

    def test_repeatable_with_same_seed_and_clock(self):
        other = self.out.parent / "again"
        run_study(CONCEPT, DemoEngine(), self.out, seed=9, now=NOW, study_id="x")
        run_study(CONCEPT, DemoEngine(), other, seed=9, now=NOW, study_id="x")
        for name in ARTIFACTS:
            self.assertEqual((self.out / name).read_text(), (other / name).read_text(), name)

    def test_unparseable_output_becomes_error_response_not_crash(self):
        result = run_study(CONCEPT, RecordingEngine("I refuse."), self.out, panel_size=3, now=NOW)
        self.assertEqual(result.summary["counts"]["failed"], 3)
        self.assertTrue(all(not r.ok for r in result.responses))
        self.assertEqual(self.read("responses.json")[0]["raw_output"], "I refuse.")
        self.assertTrue(result.summary["warnings"])

    def test_nonempty_destination_is_rejected_without_changes(self):
        self.out.mkdir(parents=True)
        marker = self.out / "keep.txt"
        marker.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            run_study(CONCEPT, DemoEngine(), self.out, now=NOW)
        self.assertEqual(marker.read_text(encoding="utf-8"), "original")
        self.assertEqual(list(self.out.iterdir()), [marker])

    def test_existing_empty_destination_is_supported(self):
        self.out.mkdir(parents=True)
        result = run_study(CONCEPT, DemoEngine(), self.out, now=NOW)
        self.assertEqual(set(result.paths), set(ARTIFACTS))
        self.assertEqual({path.name for path in self.out.iterdir()}, set(ARTIFACTS))

    def test_rename_failure_restores_caller_owned_empty_destination(self):
        self.out.mkdir(parents=True)
        self.out.chmod(0o700)
        original_inode = self.out.stat().st_ino
        original_replace = Path.replace
        replace_calls = 0

        def fail_commit(path, target):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise OSError("rename failed")
            return original_replace(path, target)

        with mock.patch.object(Path, "replace", fail_commit):
            with self.assertRaisesRegex(OSError, "rename failed"):
                run_study(CONCEPT, DemoEngine(), self.out, now=NOW)
        self.assertTrue(self.out.is_dir())
        self.assertEqual(self.out.stat().st_ino, original_inode)
        self.assertEqual(os.stat(self.out).st_mode & 0o777, 0o700)
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(list(self.out.parent.glob(f".{self.out.name}.*")), [])

    def test_artifact_write_failure_leaves_no_partial_destination(self):
        original = runner._write_text_synced
        writes = 0

        def fail_second(path, text):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("disk full")
            return original(path, text)

        with mock.patch.object(runner, "_write_text_synced", fail_second):
            with self.assertRaisesRegex(OSError, "disk full"):
                run_study(CONCEPT, DemoEngine(), self.out, now=NOW)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.out.parent.glob(f".{self.out.name}.*")), [])

    def test_engine_failure_propagates_and_writes_nothing(self):
        class Broken(RecordingEngine):
            def generate(self, *a):
                raise EngineError("executable missing")
        with self.assertRaises(EngineError):
            run_study(CONCEPT, Broken(), self.out, now=NOW)
        self.assertFalse(self.out.exists())

    def test_naive_clock_rejected(self):
        with self.assertRaises(ValueError):
            run_study(CONCEPT, DemoEngine(), self.out, now=datetime(2025, 1, 1))


if __name__ == "__main__":
    unittest.main()
