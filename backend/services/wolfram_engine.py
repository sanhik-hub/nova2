import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WolframResult:
    success: bool
    result: Any = None
    error: Optional[str] = None
    raw_output: Optional[str] = None


class WolframEngine:
    """
    Interface between Python and Wolfram Language Engine.

    Python
        ↓
    wolframscript
        ↓
    Wolfram Language Engine
        ↓
    JSON
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

    def _build_wrapper(self, expression: str) -> str:
        """
        Wrap the user's Wolfram Language expression and force
        a JSON-safe response.
        """

        # Escape the expression so it can safely be placed inside
        # a Wolfram string.
        escaped = (
            expression
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", "\\r")
            .replace("\n", "\\n")
        )

        wrapper = f'''
Module[
    {{input, result, success, error}},

    input = "{escaped}";

    result = Quiet[
        Check[
            ToExpression[input],
            $Failed
        ]
    ];

    success = result =!= $Failed;

    If[
        success,

        ExportString[
            <|
                "success" -> True,
                "input" -> input,
                "result" -> ToString[result, InputForm]
            |>,
            "JSON"
        ],

        ExportString[
            <|
                "success" -> False,
                "input" -> input,
                "error" -> "Wolfram Engine could not evaluate the expression."
            |>,
            "JSON"
        ]
    ]
]
'''

        return wrapper

    def calculate(self, expression: str) -> WolframResult:
        """
        Evaluate arbitrary Wolfram Language code.
        """

        if not expression or not expression.strip():
            return WolframResult(
                success=False,
                error="Expression is empty."
            )

        wrapper = self._build_wrapper(expression)

        try:
            process = subprocess.run(
                [
                    self.executable,
                    "-code",
                    wrapper,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                ),
            )

        except subprocess.TimeoutExpired:
            return WolframResult(
                success=False,
                error=(
                    f"Wolfram Engine timed out after "
                    f"{self.timeout} seconds."
                ),
            )

        except Exception as exc:
            return WolframResult(
                success=False,
                error=f"Failed to start Wolfram Engine: {exc}"
            )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if not stdout:
            return WolframResult(
                success=False,
                error=stderr or "Wolfram Engine returned no output.",
                raw_output=stdout,
            )

        try:
            data: Any = json.loads(stdout)

        except json.JSONDecodeError:
            return WolframResult(
                success=False,
                error="Wolfram returned invalid JSON.",
                raw_output=stdout,
            )

        if data.get("success") is True:
            return WolframResult(
                success=True,
                result=data.get("result"),
                raw_output=stdout,
            )

        return WolframResult(
            success=False,
            error=data.get(
                "error",
                stderr or "Unknown Wolfram error."
            ),
            raw_output=stdout,
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
        {{expr, var, data, step, x}},

        Check[

            expr = ToExpression["{escaped_expression}"];
            var = ToExpression["{escaped_variable}"];

            step = ({xmax} - ({xmin})) / {points};

            data = Table[
                {{
                    Round[N[x, 12], 10^-10],
                    Round[N[expr /. var -> x, 12], 10^-10]
                }},
                {{
                    x,
                    {xmin},
                    {xmax},
                    step
                }}
            ];

            ExportString[
                <|
                    "success" -> True,
                    "points" -> data
                |>,
                "JSON"
            ],

            ExportString[
                <|
                    "success" -> False,
                    "error" -> ToString[$MessageList, InputForm]
                |>,
                "JSON"
            ]

        ]
    ]
    '''

        try:
            process = subprocess.run(
                [
                    self.executable,
                    "-code",
                    wrapper,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                ),
            )

        except subprocess.TimeoutExpired:
            return WolframResult(
                success=False,
                error=f"Wolfram Engine timed out after {self.timeout} seconds.",
            )

        except Exception as exc:
            return WolframResult(
                success=False,
                error=f"Failed to start Wolfram Engine: {exc}",
            )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if not stdout:
            return WolframResult(
                success=False,
                error=stderr or "Wolfram Engine returned no output.",
            )

        try:
            data = json.loads(stdout)

        except json.JSONDecodeError:
            return WolframResult(
                success=False,
                error="Wolfram returned invalid JSON.",
                raw_output=stdout,
            )

        if not data.get("success"):
            return WolframResult(
                success=False,
                error=data.get(
                    "error",
                    stderr or "Wolfram plotting failed.",
                ),
                raw_output=stdout,
            )

        return WolframResult(
            success=True,
            result=data.get("points", []),
            raw_output=stdout,
        )