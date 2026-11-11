"""Local providers used by the replication harness."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from .models import ChatRequest, ChatResponse

ProcessRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run_process(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )


class MLXCLIProvider:
    """Provider adapter for a locally installed ``mlx_lm.generate`` executable."""

    name = "mlx-cli"

    def __init__(
        self,
        *,
        executable: str = "mlx_lm.generate",
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 0,
        process_runner: ProcessRunner = _run_process,
    ) -> None:
        self.executable = executable
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.process_runner = process_runner

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "executable": self.executable,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "chat_template_config": {"enable_thinking": False},
        }

    def complete(self, request: ChatRequest) -> ChatResponse:
        system_prompt = "\n\n".join(
            message.content for message in request.messages if message.role == "system"
        )
        user_prompt = "\n\n".join(
            message.content for message in request.messages if message.role != "system"
        )
        argv = [
            self.executable,
            "--model",
            request.model,
            "--system-prompt",
            system_prompt,
            "--prompt",
            user_prompt,
            "--max-tokens",
            str(self.max_tokens),
            "--temp",
            str(self.temperature),
            "--seed",
            str(self.seed),
            "--chat-template-config",
            json.dumps({"enable_thinking": False}, separators=(",", ":")),
            "--verbose",
            "False",
        ]
        result = self.process_runner(argv)
        raw_response = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        if result.returncode != 0:
            return ChatResponse(
                provider=self.name,
                model=request.model,
                text=None,
                raw_response=raw_response,
                error=f"mlx_lm.generate exited with status {result.returncode}",
            )
        return ChatResponse(
            provider=self.name,
            model=request.model,
            text=result.stdout.strip(),
            raw_response=raw_response,
        )


class MLXInProcessProvider:
    """Persistent MLX adapter that loads one local model for many calls."""

    name = "mlx-in-process"

    def __init__(
        self,
        *,
        model_id: str,
        model: Any,
        tokenizer: Any,
        generate_fn: Callable[..., str],
        sampler_factory: Callable[..., Any],
        seed_fn: Callable[[int], None],
        max_tokens: int = 192,
    ) -> None:
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn
        self.sampler_factory = sampler_factory
        self.seed_fn = seed_fn
        self.max_tokens = max_tokens

    @classmethod
    def load(cls, model_id: str, *, max_tokens: int = 192) -> "MLXInProcessProvider":
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = load(model_id)
        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            generate_fn=generate,
            sampler_factory=make_sampler,
            seed_fn=mx.random.seed,
            max_tokens=max_tokens,
        )

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "chat_template_config": {"enable_thinking": False},
            "persistent_model": True,
        }

    def complete(self, request: ChatRequest) -> ChatResponse:
        if request.model != self.model_id:
            return ChatResponse(self.name, request.model, None, {}, "request model does not match loaded model")
        seed = int(request.metadata.get("seed", 0))
        try:
            self.seed_fn(seed)
            messages = [message.to_dict() for message in request.messages]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            sampler = self.sampler_factory(request.temperature)
            max_tokens = int(request.metadata.get("max_tokens", self.max_tokens))
            text = self.generate_fn(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            raw_response = {"text": text, "seed": seed, "temperature": request.temperature}
            return ChatResponse(self.name, request.model, text.strip(), raw_response)
        except Exception as exc:
            return ChatResponse(
                self.name,
                request.model,
                None,
                {"seed": seed, "temperature": request.temperature, "exception": repr(exc)},
                f"MLX generation failed: {exc}",
            )


class FrontierMockProvider:
    """Deterministic typed responses for bounded frontier-runner verification."""

    name = "frontier-mock"

    @property
    def configuration(self) -> dict[str, object]:
        return {"script": "frontier-mock-v1", "thinking_enabled": False}

    def complete(self, request: ChatRequest) -> ChatResponse:
        task = request.metadata.get("task")
        if task == "subject-turn":
            agent = str(request.metadata["agent"])
            turn = int(request.metadata["turn"])
            text = f"{agent} public mock statement for turn {turn + 1}."
            return ChatResponse(self.name, request.model, text, {"task": task, "text": text})
        study_id = str(request.metadata.get("study_id"))
        scripted = {
            "frontier-mug-bargaining": {"deal": 1, "price": 10},
            "frontier-bail-setting": {"bail_amount": 50000},
            "frontier-lawyer-interview": {"hire": 1, "rating": 75},
            "frontier-art-auction": {"final_price": 100, "winner": "Bidder A"},
        }
        values = scripted[study_id]
        payload = {
            "outcomes": values,
            "evidence": {name: f"mock evidence for {name}" for name in values},
        }
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return ChatResponse(self.name, request.model, text, {"task": task, "text": text})


class DeterministicMockProvider:
    """A no-network scripted provider for exercising the full local pipeline."""

    name = "deterministic-mock"

    @property
    def configuration(self) -> dict[str, object]:
        return {"script": "mug-bargaining-v1", "temperature": 0.0}

    _SCRIPT = {
        ("buyer", 0): "I can offer $45 for the mug.",
        ("seller", 1): "I can sell it for $60.",
        ("buyer", 2): "Agreed. I will pay $60 for the mug.",
    }

    def complete(self, request: ChatRequest) -> ChatResponse:
        agent = str(request.metadata.get("agent", "unknown"))
        turn = int(request.metadata.get("turn", 0))
        text = self._SCRIPT.get((agent, turn), "No further statement.")
        raw_response = {"agent": agent, "turn": turn, "text": text}
        return ChatResponse(
            provider=self.name,
            model=request.model,
            text=text,
            raw_response=raw_response,
        )
