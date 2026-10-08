import json
import urllib.request
import urllib.error
from typing import Any, Optional

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

    def supports_tool_calls(self) -> bool:
        """Return whether the loaded llama-server exposes a tool-use template."""
        try:
            with urllib.request.urlopen(
                f"{self._base_url()}/props",
                timeout=3,
            ) as response:
                props = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            return False

        caps = props.get("chat_template_caps")
        if isinstance(caps, dict):
            return bool(caps.get("supports_tool_calls"))

        # Older llama.cpp builds may expose a separate tool-use template
        # property instead of chat_template_caps.
        tool_template = props.get("chat_template_tool_use")
        return isinstance(tool_template, str) and bool(tool_template.strip())

    @staticmethod
    def _repair_mojibake(value: str) -> str:
        """Repair text that was accidentally decoded as Latin-1/CP1252.

        Correct UTF-8 scientific text is returned unchanged. Repair is attempted
        only when characteristic mojibake markers are present.
        """
        if not isinstance(value, str):
            return value

        text = value
        markers = ("Ã", "Â", "â", "Î", "Ï")

        for _ in range(2):
            if not any(marker in text for marker in markers):
                break

            repaired = None
            bad_before = sum(text.count(marker) for marker in markers)

            for encoding in ("cp1252", "latin1"):
                try:
                    candidate = text.encode(encoding).decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue

                bad_after = sum(candidate.count(marker) for marker in markers)
                if bad_after < bad_before:
                    repaired = candidate
                    break

            if repaired is None or repaired == text:
                break
            text = repaired

        replacements = {
            "â†’": "→",
            "â†": "←",
            "â‡’": "⇒",
            "â‰ˆ": "≈",
            "âˆž": "∞",
            "Â·": "·",
            "Â²": "²",
            "Â³": "³",
            "Â¹": "¹",
            "â‰¤": "≤",
            "â‰¥": "≥",
            # Common Windows/ANSI corruption of UTF-8 subscript digits.
            "â‚€": "₀",
            "â‚": "₁",
            "â‚‚": "₂",
            "â‚ƒ": "₃",
            "â‚„": "₄",
            "â‚…": "₅",
            "â‚†": "₆",
            "â‚‡": "₇",
            "â‚ˆ": "₈",
            "â‚‰": "₉",
        }
        for broken, correct in replacements.items():
            text = text.replace(broken, correct)

        return text

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = False,
        response_format: Optional[dict[str, Any]] = None,
        request_timeout: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> str:

        if not self.loaded:
            self.load()

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Thinking mode is controlled explicitly through
        # chat_template_kwargs below. Do not also inject /no_think into
        # the user prompt: Qwen's template can treat that literal control
        # token as part of the user's request and change the response
        # behavior. Direct llama.cpp requests use the same message content
        # with chat_template_kwargs.enable_thinking=False.
        user_content = prompt

        messages.append({
            "role": "user",
            "content": user_content,
        })

        effective_max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.config.max_tokens
        )

        payload = {
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": (
                temperature
                if temperature is not None
                else self.config.temperature
            ),
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            },
        }

        if response_format is not None:
            payload["response_format"] = response_format

        if tools is not None:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # Always retain the configured transport fallback when the caller
        # does not supply a provider-level completion bound. This prevents
        # llama.cpp from interpreting an omitted value as unbounded output.

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
                timeout=(
                    request_timeout
                    if request_timeout is not None
                    else self.config.request_timeout
                ),
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

        content = self._repair_mojibake(content).strip()

        if not content:
            raise RuntimeError(
                "Local LLM returned an empty answer."
            )

        return content