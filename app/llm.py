# LLM provider abstraction.

# Defines a minimal provider interface and a concrete Ollama implementation,
# selected at runtime via ``settings.llm_provider``. Keeping generation behind an
# interface lets the rest of the app stay agnostic to the backend.


from abc import ABC, abstractmethod

import httpx

from app.config import settings


class LLMProvider(ABC):
    """Interface for text-generation backends."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a completion for ``prompt`` under an optional system prompt."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Generation via a local Ollama server's ``/api/generate`` endpoint."""

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate", json=payload, timeout=120.0
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Is it running "
                f"(`ollama serve`) and is model {self.model} pulled? ({exc})"
            ) from exc
        return response.json()["response"]


def get_llm_provider() -> LLMProvider:
    """Return the provider configured by ``settings.llm_provider``."""
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


llm = get_llm_provider()
