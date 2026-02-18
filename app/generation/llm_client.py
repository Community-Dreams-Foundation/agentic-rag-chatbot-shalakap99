"""
LLM Client — talks to locally running Ollama.

No API key needed. Ollama serves llama3.2 over a local HTTP endpoint.
Default model is llama3.2 (what's already pulled on this machine).
"""

from __future__ import annotations
import os
import requests
from dataclasses import dataclass

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")
TIMEOUT_S       = 120


@dataclass
class LLMResponse:
    text:               str
    model:              str
    prompt_tokens:      int = 0
    completion_tokens:  int = 0


class OllamaClient:
    """
    Thin wrapper around Ollama's /api/chat endpoint.
    Uses non-streaming mode for simplicity.
    """

    def __init__(
        self,
        model:    str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model    = model
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally available models."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def chat(
        self,
        messages:    list[dict],
        temperature: float = 0.1,
        max_tokens:  int   = 1024,
    ) -> LLMResponse:
        """
        Send a chat request to Ollama.

        Args:
            messages:    [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 0.1 keeps answers factual and grounded
            max_tokens:  max tokens in the response

        Returns:
            LLMResponse with .text containing the answer.

        Raises:
            RuntimeError if Ollama is not reachable or times out.
        """
        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=TIMEOUT_S,
            )
            r.raise_for_status()
            data  = r.json()
            reply = data.get("message", {}).get("content", "").strip()
            return LLMResponse(
                text=reply,
                model=self.model,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Is it running?\n"
                "  Start it: ollama serve\n"
                f"  Expected: {self.base_url}"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timed out after {TIMEOUT_S}s. "
                "Try a smaller model or check system resources."
            )


# ---------------------------------------------------------------------------
# Run directly: python -m app.generation.llm_client
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich import print as rprint

    client = OllamaClient()

    if not client.is_available():
        rprint("[red]✗ Ollama not running. Start with: ollama serve[/red]")
    else:
        models = client.list_models()
        rprint(f"[green]✓ Ollama running[/green]")
        rprint(f"  Available models: {models}")
        rprint(f"  Using: {client.model}\n")

        rprint("[bold]Sending test message…[/bold]")
        resp = client.chat([
            {"role": "user", "content": "In one sentence, what is a transformer in ML?"}
        ])
        rprint(f"[green]Response:[/green] {resp.text}")
        rprint(f"[dim]Tokens: {resp.prompt_tokens} prompt, "
               f"{resp.completion_tokens} completion[/dim]")
