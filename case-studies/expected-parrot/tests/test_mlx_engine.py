import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from parrot_baseline.engines import EngineError
from parrot_baseline.mlx_engine import DEFAULT_MODEL, MlxEngine
from parrot_baseline.panel import default_panel
from parrot_baseline.parsing import parse_answers
from parrot_baseline.prompts import build_persona_prompt
from parrot_baseline.runner import run_study
from parrot_baseline.schemas import Concept
from parrot_baseline.survey import default_questions

CONCEPT = Concept("Lasagna Cola", "A savory cola with tomato and basil notes.")
QS = default_questions()
PERSONA = default_panel()[0]
PROMPT = build_persona_prompt(CONCEPT, PERSONA, QS)
ANSWERS = {"appeal": 4, "clarity": 4, "uniqueness": 5, "purchase_intent": 3,
           "first_reaction": "Fun.", "concerns": "Salt.", "improvement": "Sample."}
NOISY = "==========\n```json\n" + json.dumps(ANSWERS) + "\n```\n==========\nPrompt: 200 tokens\nGeneration: 90 tokens\n"


class FakeRun:
    def __init__(self, stdout=NOISY, returncode=0, stderr="", raises=None):
        self.calls, self.stdout, self.returncode, self.stderr, self.raises = [], stdout, returncode, stderr, raises

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def engine(run, **kw):
    return MlxEngine(executable="/fake/mlx_lm.generate", run=run, **kw)


class MlxEngineTests(unittest.TestCase):
    def generate(self, eng, seed=7):
        return eng.generate(CONCEPT, PERSONA, QS, PROMPT, seed)

    def test_identity_and_default_model(self):
        self.assertEqual(DEFAULT_MODEL, "mlx-community/Qwen3-8B-4bit")
        eng = engine(FakeRun())
        self.assertEqual((eng.name, eng.model), ("mlx", DEFAULT_MODEL))

    def test_configuration_records_reproduction_settings(self):
        eng = engine(FakeRun(), max_tokens=123, temperature=0.25, timeout=9,
                     chat_template_config={"enable_thinking": False, "mode": "test"})
        self.assertEqual(eng.configuration(), {
            "model": DEFAULT_MODEL,
            "executable": "/fake/mlx_lm.generate",
            "max_tokens": 123,
            "temperature": 0.25,
            "timeout": 9,
            "chat_template_config": {"enable_thinking": False, "mode": "test"},
        })

    def test_command_is_argv_list_without_shell(self):
        run = FakeRun()
        self.generate(engine(run), seed=7)
        cmd, kwargs = run.calls[0]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "/fake/mlx_lm.generate")
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(cmd[cmd.index("--model") + 1], "mlx-community/Qwen3-8B-4bit")
        self.assertEqual(json.loads(cmd[cmd.index("--chat-template-config") + 1]), {"enable_thinking": False})
        self.assertEqual(cmd[cmd.index("--seed") + 1], "7")
        self.assertIn("--max-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--prompt") + 1], "-")

    def test_subprocess_output_is_captured_in_bounded_files(self):
        run = FakeRun()
        self.generate(engine(run))
        _, kwargs = run.calls[0]
        self.assertNotIn("capture_output", kwargs)
        self.assertTrue(hasattr(kwargs["stdout"], "write"))
        self.assertTrue(hasattr(kwargs["stderr"], "write"))

    def test_prompt_goes_through_stdin_not_argv(self):
        run = FakeRun()
        self.generate(engine(run))
        cmd, kwargs = run.calls[0]
        self.assertEqual(kwargs["input"], PROMPT)
        self.assertNotIn(PROMPT, cmd)
        self.assertTrue(kwargs["text"])
        self.assertIn("timeout", kwargs)

    def test_returns_noisy_stdout_which_parses(self):
        raw = self.generate(engine(FakeRun()))
        self.assertEqual(parse_answers(raw, QS), ANSWERS)

    def test_missing_executable_gives_install_hint(self):
        eng = MlxEngine(run=FakeRun(), which=lambda name: None)
        with self.assertRaises(EngineError) as ctx:
            self.generate(eng)
        self.assertIn("mlx_lm.generate", str(ctx.exception))
        self.assertIn("pip install mlx-lm", str(ctx.exception))

    def test_executable_found_on_path(self):
        run = FakeRun()
        MlxEngine(run=run, which=lambda name: "/bin/" + name).generate(CONCEPT, PERSONA, QS, PROMPT, 1)
        self.assertEqual(run.calls[0][0][0], "/bin/mlx_lm.generate")

    def test_nonzero_exit_reports_stderr_tail(self):
        eng = engine(FakeRun(returncode=1, stderr="Traceback...\nOSError: model not found"))
        with self.assertRaises(EngineError) as ctx:
            self.generate(eng)
        self.assertIn("exit code 1", str(ctx.exception))
        self.assertIn("model not found", str(ctx.exception))

    def test_timeout_reported(self):
        eng = engine(FakeRun(raises=subprocess.TimeoutExpired("mlx", 5)), timeout=5)
        with self.assertRaises(EngineError) as ctx:
            self.generate(eng)
        self.assertIn("timed out", str(ctx.exception))

    def test_os_error_reported(self):
        eng = engine(FakeRun(raises=PermissionError("denied")))
        with self.assertRaises(EngineError) as ctx:
            self.generate(eng)
        self.assertIn("denied", str(ctx.exception))

    def test_study_survives_one_bad_persona_with_useful_error(self):
        replies = iter([NOISY, "Sorry, I can't help.", NOISY])
        run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, next(replies), "")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_study(CONCEPT, engine(run), tmp, panel_size=3)
        self.assertEqual([r.ok for r in result.responses], [True, False, True])
        self.assertIn("no JSON object found", result.responses[1].error)
        self.assertIn("Sorry", result.responses[1].error)

    def test_real_subprocess_with_stub_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "mlx_lm.generate"
            stub.write_text(
                f"#!{sys.executable}\nimport sys\nprompt = sys.stdin.read()\n"
                "assert 'Lasagna Cola' in prompt and '--prompt' in sys.argv\n"
                f"print({NOISY!r})\n"
            )
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
            raw = MlxEngine(executable=str(stub)).generate(CONCEPT, PERSONA, QS, PROMPT, 1)
        self.assertEqual(parse_answers(raw, QS), ANSWERS)


if __name__ == "__main__":
    unittest.main()
