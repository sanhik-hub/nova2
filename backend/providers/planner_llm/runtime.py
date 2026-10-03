import json
import urllib.error
import urllib.request
from typing import Optional

from .config import PlannerLLMConfig


class PlannerLLMRuntime:
    def __init__(self, config: Optional[PlannerLLMConfig] = None):
        self.config = config or PlannerLLMConfig()
        self._available = False

    def load(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("Planner LLM is disabled.")
        self._check_health()
        self._available = True

    def _base_url(self) -> str:
        return self.config.server_url.rstrip("/")

    def _check_health(self) -> None:
        try:
            with urllib.request.urlopen(
                f"{self._base_url()}/health",
                timeout=3,
            ) as response:
                if response.status != 200:
                    raise RuntimeError("Planner health check failed.")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Planner LLM server is not running at {self._base_url()}."
            ) from exc

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
        structured_json: bool = True,
    ) -> str:
        if not self._available:
            self.load()

        # llama.cpp handles JSON-object mode reliably here, but the full
        # response schema can trigger constrained-decoding stalls on the
        # lightweight planner model. Keep the semantic schema in the
        # reasoning service and enforce its shape after parsing there.
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        # JSON constrained decoding is useful for larger structured planner
        # tasks, but it adds unnecessary decoding overhead for tiny metadata
        # plans. Guided Attempt can request compact text mode and parse its two
        # scalar values locally.
        if structured_json:
            payload["response_format"] = {"type": "json_object"}

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        request = urllib.request.Request(
            f"{self._base_url()}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Planner LLM HTTP error {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')}"
            ) from exc
        except urllib.error.URLError as exc:
            self._available = False
            raise RuntimeError("Could not connect to planner LLM.") from exc

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError("Planner LLM returned no choices.")

        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Planner LLM returned an empty plan.")
        return content
