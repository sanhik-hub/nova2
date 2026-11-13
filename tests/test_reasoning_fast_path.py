import json
import unittest
from unittest.mock import MagicMock, patch

from backend.providers.local_llm.runtime import LocalLLMRuntime
from backend.services.reasoning_service import ReasoningService


class ReasoningFastPathTests(unittest.TestCase):
    def setUp(self):
        self.service = ReasoningService(router=MagicMock())

    def test_extracts_complete_arithmetic_request(self):
        self.assertEqual(
            self.service._extract_numeric_calculation("Calculate 2 + 2."),
            "2 + 2",
        )
        self.assertEqual(
            self.service._extract_numeric_calculation("What is 2 * (3 + 4)?"),
            "2 * (3 + 4)",
        )

    def test_does_not_extract_arithmetic_from_stem_word_problem(self):
        problem = (
            "A projectile is launched at 30 m/s at 45 degrees. "
            "Find its range."
        )
        self.assertIsNone(self.service._extract_numeric_calculation(problem))

    def test_result_only_arithmetic_uses_wolfram_without_generation(self):
        self.service.router.calculate.return_value = {
            "success": True,
            "result": "4",
        }

        result = self.service.solve("Calculate 2 + 2.", explanation=False)

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "4")
        self.assertEqual(result["engine"], "wolfram")
        self.service.router.calculate.assert_called_once()
        self.service.router.generate.assert_not_called()
        self.service.router.supports_tool_calls.assert_not_called()


class NativeToolResponseTests(unittest.TestCase):
    def test_runtime_returns_native_tool_arguments_as_json(self):
        runtime = LocalLLMRuntime()
        runtime._available = True

        response_payload = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "wolfram",
                            "arguments": json.dumps({
                                "operation": "calculate",
                                "arguments": {"expression": "2 + 2"},
                            }),
                        },
                    }],
                },
            }],
        }
        http_response = MagicMock()
        http_response.__enter__.return_value.read.return_value = (
            json.dumps(response_payload).encode("utf-8")
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "wolfram",
                "parameters": {"type": "object"},
            },
        }]

        with patch(
            "backend.providers.local_llm.runtime.urllib.request.urlopen",
            return_value=http_response,
        ):
            result = runtime.generate(
                prompt="Calculate 2 + 2.",
                tools=tools,
                tool_choice="required",
            )

        self.assertEqual(
            json.loads(result),
            {
                "operation": "calculate",
                "arguments": {"expression": "2 + 2"},
            },
        )


if __name__ == "__main__":
    unittest.main()
