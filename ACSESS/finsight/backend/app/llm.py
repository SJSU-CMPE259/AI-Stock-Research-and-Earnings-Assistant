"""LLM wrapper supporting Anthropic, OpenAI, and remote (vLLM/ollama) providers."""
import logging
import time
from typing import Optional

import requests

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    """Provider-agnostic LLM client."""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.remote_url = settings.remote_llm_url

        if self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        elif self.provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=settings.openai_api_key)
        elif self.provider == "remote":
            self.session = requests.Session()
            self.session.headers.update({"Content-Type": "application/json"})
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion using the configured provider.

        Args:
            system: System prompt
            messages: List of messages ({"role": "user"|"assistant", "content": "..."})
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        start = time.time()

        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            text = response.content[0].text
            latency_ms = (time.time() - start) * 1000
            logger.info(
                f"Anthropic completion | model={self.model} | "
                f"tokens_in={response.usage.input_tokens} | "
                f"tokens_out={response.usage.output_tokens} | "
                f"latency={latency_ms:.0f}ms"
            )
            return text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            text = response.choices[0].message.content
            latency_ms = (time.time() - start) * 1000
            logger.info(
                f"OpenAI completion | model={self.model} | "
                f"tokens_in={response.usage.prompt_tokens} | "
                f"tokens_out={response.usage.completion_tokens} | "
                f"latency={latency_ms:.0f}ms"
            )
            return text

        elif self.provider == "remote":
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            try:
                response = self.session.post(
                    f"{self.remote_url}/chat/completions",
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                latency_ms = (time.time() - start) * 1000
                logger.info(
                    f"Remote LLM completion | url={self.remote_url} | "
                    f"model={self.model} | latency={latency_ms:.0f}ms"
                )
                return text
            except requests.RequestException as e:
                logger.error(f"Remote LLM error: {e}")
                raise RuntimeError(
                    f"Failed to call remote LLM at {self.remote_url}: {e}"
                ) from e

        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def get_llm() -> LLMClient:
    """Get or create the LLM client singleton."""
    if not hasattr(get_llm, "_client"):
        get_llm._client = LLMClient()
    return get_llm._client
