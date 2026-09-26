import json
import urllib.request
import urllib.error
from typing import Optional

from .config import LocalLLMConfig


class LocalLLMRuntime:
    """
    HTTP client for the local llama.cpp server.

    The llama.cpp server runs separately, so the Python backend does not
    need llama-cpp-python or its native dependencies.
    """

    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self._available = False

    @property
    def loaded(self) -> bool:
        return self._available

    def load(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("Local LLM is disabled.")

        self._check_health()
        self._available = True

    def unload(self) -> None:
        # The actual model belongs to llama-server, not this Python process.
        self._available = False

    def _base_url(self) -> str:
        return self.config.server_url.rstrip("/")

    def _check_health(self) -> None:
        url = f"{self._base_url()}/health"

        try:
            with urllib.request.urlopen(
                url,
                timeout=3,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Local LLM health check failed: HTTP {response.status}"
                    )

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Local LLM server is not running at "
                f"{self._base_url()}. "
                "Start llama-server first."
            ) from exc

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> str:

        if not self.loaded:
            self.load()

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        user_content = prompt

        if not enable_thinking:
            user_content = "/no_think\n" + user_content

        messages.append({
            "role": "user",
            "content": user_content,
        })

        payload = {
            "messages": messages,
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else self.config.max_tokens
            ),
            "temperature": (
                temperature
                if temperature is not None
                else self.config.temperature
            ),
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self._base_url()}/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout,
            ) as response:

                raw = response.read().decode("utf-8")
                result = json.loads(raw)

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Local LLM HTTP error {exc.code}: {body}"
            ) from exc

        except urllib.error.URLError as exc:
            self._available = False
            raise RuntimeError(
                "Could not connect to local LLM server."
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Local LLM returned invalid JSON."
            ) from exc

        choices = result.get("choices", [])

        if not choices:
            raise RuntimeError(
                "Local LLM returned no choices."
            )

        message = choices[0].get("message", {})
        content = message.get("content")

        if content is None:
            content = ""

        content = content.strip()

        if not content:
            raise RuntimeError(
                "Local LLM returned an empty answer."
            )

        return content