import atexit
import json
import os
import shutil
import threading
from dataclasses import dataclass
from typing import Any, Optional

from wolframclient.evaluation import WolframLanguageSession


@dataclass
class WolframResult:
    success: bool
    result: Any = None
    error: Optional[str] = None
    raw_output: Optional[str] = None


class WolframEngine:
    """
    Interface between Python and the Wolfram Language Engine.

    A single persistent WolframLanguageSession is reused for all
    calculations made by this backend process.

    Python
        ↓
    WolframLanguageSession
        ↓
    Wolfram Language Engine
        ↓
    Python
    """

    DEFAULT_PATH = (
        r"C:\Program Files\Wolfram Research"
        r"\WolframScript\wolframscript.exe"
    )

    def __init__(
        self,
        executable: Optional[str] = None,
        timeout: int = 30,
    ):
        self.timeout = timeout

        self.executable = (
            executable
            or shutil.which("wolframscript")
            or self.DEFAULT_PATH
        )

        if not os.path.isfile(self.executable):
            raise FileNotFoundError(
                "WolframScript executable was not found.\n"
                f"Expected: {self.executable}"
            )

        self._session: Optional[WolframLanguageSession] = None
        self._session_lock = threading.RLock()

        atexit.register(self.close)

    def _get_session(self) -> WolframLanguageSession:
        """
        Return the persistent Wolfram session, creating it lazily.

        The session is created only when the first calculation is made.
        """

        with self._session_lock:
            if self._session is None:
                self._session = WolframLanguageSession()

            return self._session

    def close(self) -> None:
        """
        Terminate the persistent Wolfram session.
        """

        with self._session_lock:
            if self._session is not None:
                try:
                    self._session.terminate()
                except Exception:
                    pass
                finally:
                    self._session = None

    def restart(self) -> None:
        """Reset the persistent session; the next evaluation starts a fresh kernel."""
        self.close()

    def status(self) -> dict[str, Any]:
        """Report session state without starting a kernel just to inspect it."""
        with self._session_lock:
            session = self._session
            if session is None:
                return {
                    "session_exists": False,
                    "session_started": False,
                }

            try:
                started = bool(session.started)
            except Exception:
                started = False

            return {
                "session_exists": True,
                "session_started": started,
            }

    def _evaluate(self, expression: str) -> Any:
        """
        Evaluate an expression using the persistent Wolfram session.
        """

        with self._session_lock:
            session = self._get_session()
            return session.evaluate(expression)

    @staticmethod
    def _result_to_string(result: Any) -> str:
        """
        Convert a Wolfram result to a stable InputForm string.
        """

        try:
            session_expression = (
                f"ToString[{result}, InputForm]"
            )

            # This helper is intentionally not used for the main
            # calculation path because `result` may already be a
            # Python representation returned by wolframclient.
            return str(result)

        except Exception:
            return str(result)

    def calculate(self, expression: str) -> WolframResult:
        """
        Evaluate arbitrary Wolfram Language code using the persistent
        Wolfram session.
        """

        if not expression or not expression.strip():
            return WolframResult(
                success=False,
                error="Expression is empty.",
            )

        wrapper = f'''
Module[
    {{result, success}},

    result = Quiet[
        Check[
            ToExpression[
                {json.dumps(expression)}
            ],
            $Failed
        ]
    ];

    success = result =!= $Failed;

    If[
        success,
        <|
            "success" -> True,
            "result" -> ToString[result, InputForm]
        |>,
        <|
            "success" -> False,
            "error" -> "Wolfram Engine could not evaluate the expression."
        |>
    ]
]
'''

        try:
            data = self._evaluate(wrapper)

        except Exception as exc:
            return WolframResult(
                success=False,
                error=f"Wolfram evaluation failed: {exc}",
            )

        if not isinstance(data, dict):
            return WolframResult(
                success=False,
                error="Wolfram returned an unexpected response.",
                raw_output=str(data),
            )

        if data.get("success") is True:
            return WolframResult(
                success=True,
                result=data.get("result"),
                raw_output=str(data),
            )

        return WolframResult(
            success=False,
            error=data.get(
                "error",
                "Unknown Wolfram error.",
            ),
            raw_output=str(data),
        )

    def solve(
        self,
        equation: str,
        variable: str = "x",
    ) -> WolframResult:
        """
        Convenience method for Solve.
        """

        expression = (
            f"Solve[{equation}, {variable}]"
        )

        return self.calculate(expression)

    def simplify(self, expression: str) -> WolframResult:
        return self.calculate(
            f"Simplify[{expression}]"
        )

    def expand(self, expression: str) -> WolframResult:
        return self.calculate(
            f"Expand[{expression}]"
        )

    def factor(self, expression: str) -> WolframResult:
        return self.calculate(
            f"Factor[{expression}]"
        )

    def differentiate(
        self,
        expression: str,
        variable: str = "x",
    ) -> WolframResult:
        return self.calculate(
            f"D[{expression}, {variable}]"
        )

    def integrate(
        self,
        expression: str,
        variable: str = "x",
    ) -> WolframResult:
        return self.calculate(
            f"Integrate[{expression}, {variable}]"
        )

    def limit(
        self,
        expression: str,
        variable: str,
        value: str,
    ) -> WolframResult:
        return self.calculate(
            f"Limit[{expression}, {variable} -> {value}]"
        )

    def plot(
        self,
        expression: str,
        variable: str = "x",
        xmin: float = -10,
        xmax: float = 10,
        points: int = 200,
    ) -> WolframResult:

        if not expression or not expression.strip():
            return WolframResult(
                success=False,
                error="Expression is empty.",
            )

        escaped_expression = (
            expression
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", "\\r")
            .replace("\n", "\\n")
        )

        escaped_variable = (
            variable
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        wrapper = f'''
Module[
    {{expr, var, data}},

    Check[

        expr = ToExpression["{escaped_expression}"];
        var = ToExpression["{escaped_variable}"];

        data = Table[
            {{
                Round[N[x, 12], 10^-10],
                Round[N[expr /. var -> x, 12], 10^-10]
            }},
            {{
                x,
                {xmin},
                {xmax},
                ({xmax} - ({xmin})) / {points}
            }}
        ];

        <|
            "success" -> True,
            "points" -> data
        |>,

        <|
            "success" -> False,
            "error" -> ToString[$MessageList, InputForm]
        |>

    ]
]
'''

        try:
            data = self._evaluate(wrapper)

        except Exception as exc:
            return WolframResult(
                success=False,
                error=f"Wolfram plotting failed: {exc}",
            )

        if not isinstance(data, dict):
            return WolframResult(
                success=False,
                error="Wolfram returned an unexpected plotting response.",
                raw_output=str(data),
            )

        if not data.get("success"):
            return WolframResult(
                success=False,
                error=data.get(
                    "error",
                    "Wolfram plotting failed.",
                ),
                raw_output=str(data),
            )

        return WolframResult(
            success=True,
            result=data.get("points", []),
            raw_output=str(data),
        )
