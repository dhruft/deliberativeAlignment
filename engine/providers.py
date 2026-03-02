"""Model providers for running rollouts."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelResponse:
    """Response from a model."""
    content: str
    thinking: Optional[str] = None  # Chain of thought if available
    raw_response: Optional[dict] = None


class ModelProvider(ABC):
    """Base class for model providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass

    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this provider exposes chain-of-thought."""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return ModelResponse(
            content=response.choices[0].message.content,
            thinking=None,
            raw_response=response.model_dump(),
        )

    @property
    def supports_thinking(self) -> bool:
        return False


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""

    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        super().__init__(model_name)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = ""
        thinking = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                thinking = block.thinking

        return ModelResponse(
            content=content,
            thinking=thinking,
            raw_response=response.model_dump(),
        )

    @property
    def supports_thinking(self) -> bool:
        # Extended thinking available on some models
        return "claude-3" in self.model_name


class TogetherProvider(ModelProvider):
    """Together AI provider (OpenAI-compatible API)."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        super().__init__(model_name)
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return ModelResponse(
            content=response.choices[0].message.content,
            thinking=None,
            raw_response=response.model_dump(),
        )

    @property
    def supports_thinking(self) -> bool:
        return False


class LocalTransformersProvider(ModelProvider):
    """Local HuggingFace transformers provider (runs model on-device via MPS/CPU)."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Path to a local model directory in HuggingFace format.
        """
        super().__init__(model_name)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers and torch required: pip install transformers torch")

        import torch

        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        print(f"Loading model from {model_name} on {self._device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self._model.eval()
        print("Model loaded.")

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        import torch

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        input_ids = self._tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return ModelResponse(content=content, thinking=None)

    @property
    def supports_thinking(self) -> bool:
        return False


class MLXProvider(ModelProvider):
    """Local Apple Silicon inference via mlx-lm (recommended for Mac M-series)."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: Path to a local model directory (HuggingFace format).
                        mlx-lm will auto-quantize to 4-bit on first load.
        """
        super().__init__(model_name)
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx-lm required: pip install mlx-lm")

        print(f"Loading model from {model_name}...")
        from mlx_lm import load
        self._model, self._tokenizer = load(model_name)
        print("Model loaded.")

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        from mlx_lm import generate

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        prompt = self._tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        from mlx_lm.sample_utils import make_sampler

        content = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )

        return ModelResponse(content=content, thinking=None)

    @property
    def supports_thinking(self) -> bool:
        return False


PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "together": TogetherProvider,
    "local": LocalTransformersProvider,
    "mlx": MLXProvider,
}


def get_provider(provider_name: str, model_name: str) -> ModelProvider:
    """Get a provider instance by name."""
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(PROVIDER_REGISTRY.keys())}")
    return PROVIDER_REGISTRY[provider_name](model_name)
