import subprocess
from collections.abc import Sequence

from assr_replication.models import ChatMessage, ChatRequest
from assr_replication.providers import MLXCLIProvider, MLXInProcessProvider


def test_mlx_provider_uses_safe_argv_and_preserves_process_result() -> None:
    seen: list[list[str]] = []

    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="local answer\n", stderr="timing info\n")

    provider = MLXCLIProvider(
        executable="/opt/tools/mlx_lm.generate",
        max_tokens=32,
        temperature=0.2,
        seed=11,
        process_runner=fake_runner,
    )
    request = ChatRequest(
        model="mlx-community/Qwen3-8B-4bit",
        messages=(
            ChatMessage(role="system", content="Keep the reserve private."),
            ChatMessage(role="user", content="Offer on the mug."),
        ),
    )

    response = provider.complete(request)

    assert seen == [[
        "/opt/tools/mlx_lm.generate",
        "--model", "mlx-community/Qwen3-8B-4bit",
        "--system-prompt", "Keep the reserve private.",
        "--prompt", "Offer on the mug.",
        "--max-tokens", "32",
        "--temp", "0.2",
        "--seed", "11",
        "--chat-template-config", '{"enable_thinking":false}',
        "--verbose", "False",
    ]]
    assert response.text == "local answer"
    assert response.raw_response == {
        "stdout": "local answer\n",
        "stderr": "timing info\n",
        "returncode": 0,
    }
    assert response.error is None


def test_mlx_provider_preserves_failed_process_output() -> None:
    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 2, stdout="partial", stderr="model load failed")

    provider = MLXCLIProvider(process_runner=fake_runner)
    response = provider.complete(
        ChatRequest(
            model="local/model",
            messages=(ChatMessage(role="user", content="hello"),),
        )
    )

    assert response.text is None
    assert response.error == "mlx_lm.generate exited with status 2"
    assert response.raw_response == {
        "stdout": "partial",
        "stderr": "model load failed",
        "returncode": 2,
    }


def test_in_process_provider_uses_request_temperature_and_recorded_seed() -> None:
    seen: dict[str, object] = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["template_kwargs"] = kwargs
            return "rendered prompt"

    def generate(model, tokenizer, prompt, **kwargs):
        seen["prompt"] = prompt
        seen["generate_kwargs"] = kwargs
        return "answer"

    def make_sampler(temp):
        seen["temp"] = temp
        return f"sampler-{temp}"

    def seed(value):
        seen["seed"] = value

    provider = MLXInProcessProvider(
        model_id="local/model",
        model=object(),
        tokenizer=Tokenizer(),
        generate_fn=generate,
        sampler_factory=make_sampler,
        seed_fn=seed,
        max_tokens=77,
    )
    response = provider.complete(ChatRequest(
        model="local/model",
        messages=(ChatMessage(role="system", content="system"), ChatMessage(role="user", content="user")),
        metadata={"seed": 123},
        temperature=0.3,
    ))

    assert response.text == "answer"
    assert seen["seed"] == 123
    assert seen["temp"] == 0.3
    assert seen["template_kwargs"] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert seen["generate_kwargs"] == {"max_tokens": 77, "sampler": "sampler-0.3", "verbose": False}
