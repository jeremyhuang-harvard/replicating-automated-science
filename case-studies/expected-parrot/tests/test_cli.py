import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from parrot_baseline import cli
from parrot_baseline.engines import EngineError
from parrot_baseline.runner import ARTIFACTS

REPO = Path(__file__).resolve().parent.parent


def call(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "out"

    def test_demo_command_writes_all_artifacts(self):
        code, stdout, _ = call(["demo", "--output-dir", str(self.out)])
        self.assertEqual(code, 0)
        for name in ARTIFACTS:
            self.assertTrue((self.out / name).is_file(), name)
        self.assertIn("report.html", stdout)
        self.assertIn("not human evidence", stdout.lower())
        self.assertEqual(json.loads((self.out / "manifest.json").read_text())["engine"], "demo")

    def test_run_command_with_demo_engine(self):
        code, _, _ = call(["run", "--concept-title", "Spicy Tea", "--concept-description", "Chili-infused iced tea.",
                           "--engine", "demo", "--output-dir", str(self.out), "--panel-size", "6", "--seed", "11"])
        self.assertEqual(code, 0)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual((manifest["seed"], len(manifest["panel"]), manifest["concept"]["title"]), (11, 6, "Spicy Tea"))

    def test_missing_required_args_exit_2(self):
        code, _, err = call(["run", "--engine", "demo", "--output-dir", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("--concept-title", err)

    def test_invalid_engine_and_panel_size(self):
        base = ["run", "--concept-title", "T", "--concept-description", "D", "--output-dir", str(self.out)]
        self.assertEqual(call(base + ["--engine", "gpt"])[0], 2)
        code, _, err = call(base + ["--engine", "demo", "--panel-size", "0"])
        self.assertEqual(code, 2)
        self.assertIn("panel size", err)

    def test_blank_concept_is_a_usage_error(self):
        code, _, err = call(["run", "--concept-title", " ", "--concept-description", "D", "--engine", "demo", "--output-dir", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("title", err)

    def test_filesystem_error_reported_without_traceback(self):
        with mock.patch.object(cli, "run_study", side_effect=OSError("read-only filesystem")):
            code, _, err = call(["demo", "--output-dir", str(self.out)])
        self.assertEqual(code, 1)
        self.assertIn("read-only filesystem", err)
        self.assertNotIn("Traceback", err)

    def test_mlx_engine_failure_reported_without_traceback(self):
        class Broken:
            name, model = "mlx", "m"
            def __init__(self, **kw): pass
            def generate(self, *a): raise EngineError("mlx_lm.generate was not found")
        with mock.patch.object(cli, "MlxEngine", Broken):
            code, _, err = call(["run", "--concept-title", "T", "--concept-description", "D", "--engine", "mlx", "--output-dir", str(self.out)])
        self.assertEqual(code, 1)
        self.assertIn("mlx_lm.generate was not found", err)
        self.assertNotIn("Traceback", err)

    def test_mlx_model_option_passed_through(self):
        seen = {}
        class Spy:
            name = "mlx"
            def __init__(self, model):
                seen["model"] = model; self.model = model
            def generate(self, *a): raise EngineError("stop")
        with mock.patch.object(cli, "MlxEngine", Spy):
            call(["run", "--concept-title", "T", "--concept-description", "D", "--engine", "mlx", "--model", "my/model", "--output-dir", str(self.out)])
        self.assertEqual(seen["model"], "my/model")

    def test_module_entrypoint_demo(self):
        proc = subprocess.run([sys.executable, "-m", "parrot_baseline.cli", "demo", "--output-dir", str(self.out)],
                              cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((self.out / "report.html").is_file())


if __name__ == "__main__":
    unittest.main()
