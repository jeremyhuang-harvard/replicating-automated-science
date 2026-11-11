"""Optional engine that shells out (argv list, never a shell) to the installed `mlx_lm.generate`."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from typing import BinaryIO, Callable

from .engines import EngineError
from .schemas import Concept, Persona, Question

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
EXECUTABLE = "mlx_lm.generate"
INSTALL_HINT = "Install it with `pip install mlx-lm` (Apple silicon) or use `--engine demo`."
MAX_CAPTURE_BYTES = 1024 * 1024


class MlxEngine:
    name = "mlx"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        executable: str | None = None,
        max_tokens: int = 700,
        temperature: float = 0.7,
        timeout: float = 300.0,
        chat_template_config: dict[str, object] | None = None,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.model = model
        self._executable, self._which = executable, which
        self.max_tokens, self.temperature, self.timeout, self._run = max_tokens, temperature, timeout, run
        self.chat_template_config = dict(chat_template_config or {"enable_thinking": False})

    def _resolve(self) -> str:
        path = self._executable or self._which(EXECUTABLE)
        if not path:
            raise EngineError(f"`{EXECUTABLE}` was not found on PATH. {INSTALL_HINT}")
        return path

    def configuration(self) -> dict[str, object]:
        return {
            "model": self.model,
            "executable": self._resolve(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "chat_template_config": dict(self.chat_template_config),
        }

    def generate(self, concept: Concept, persona: Persona, questions: list[Question], prompt: str, seed: int) -> str:
        cmd = [
            self._resolve(),
            "--model", self.model,
            "--prompt", "-",  # prompt is read from stdin: no argv length limits or leading-dash surprises
            "--max-tokens", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--seed", str(seed),
            "--chat-template-config", json.dumps(self.chat_template_config),
            "--verbose", "False",
        ]
        try:
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                proc = self._run(cmd, input=prompt, stdout=stdout_file, stderr=stderr_file,
                                 text=True, timeout=self.timeout, check=False)
                stdout = proc.stdout if isinstance(proc.stdout, str) else _read_bounded(stdout_file, MAX_CAPTURE_BYTES)
                stderr = proc.stderr if isinstance(proc.stderr, str) else _read_bounded(stderr_file, MAX_CAPTURE_BYTES)
        except subprocess.TimeoutExpired:
            raise EngineError(f"{EXECUTABLE} timed out after {self.timeout:g}s for persona {persona.persona_id}") from None
        except OSError as exc:
            raise EngineError(f"could not run {EXECUTABLE}: {exc}") from exc
        if proc.returncode != 0:
            tail = (stderr or "").strip()[-500:]
            raise EngineError(f"{EXECUTABLE} failed with exit code {proc.returncode} for persona {persona.persona_id}: {tail}")
        return stdout


def _read_bounded(file: BinaryIO, limit: int) -> str:
    file.seek(0)
    data = file.read(limit + 1)
    if len(data) > limit:
        raise EngineError(f"{EXECUTABLE} output exceeded {limit} bytes")
    return data.decode("utf-8", "replace")
