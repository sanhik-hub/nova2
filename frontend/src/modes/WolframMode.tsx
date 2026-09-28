import { useMemo, useState } from "react";
import {
  computationRequest,
  type ComputationResponse,
} from "../services/api";
import { LoadingState } from "../components/LoadingState";
import { ComputationResult } from "../components/ComputationResult";

const computationOperations = [
  { value: "solve", label: "Solve" },
  { value: "simplify", label: "Simplify" },
  { value: "factor", label: "Factor" },
  { value: "expand", label: "Expand" },
  { value: "differentiate", label: "Differentiate" },
  { value: "integrate", label: "Integrate" },
  { value: "limit", label: "Limit" },
  { value: "plot", label: "Plot" },
] as const;

export function WolframMode() {
  const [operation, setOperation] = useState("solve");
  const [expression, setExpression] = useState("");
  const [variable, setVariable] = useState("x");
  const [limitValue, setLimitValue] = useState("0");
  const [xmin, setXmin] = useState("-10");
  const [xmax, setXmax] = useState("10");
  const [response, setResponse] =
    useState<ComputationResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const selectedOperation = useMemo(
    () =>
      computationOperations.find(
        (item) => item.value === operation,
      ),
    [operation],
  );

  async function compute() {
    if (!expression.trim() || loading) {
      return;
    }

    setLoading(true);
    setResponse(null);

    try {
      let result: ComputationResponse;

      switch (operation) {
        case "solve":
          result = await computationRequest("/api/solve", {
            expression: expression.trim(),
            variable: variable.trim() || "x",
            provider: "wolfram",
          });
          break;

        case "simplify":
          result = await computationRequest("/api/simplify", {
            expression: expression.trim(),
            provider: "wolfram",
          });
          break;

        case "factor":
          result = await computationRequest("/api/factor", {
            expression: expression.trim(),
            provider: "wolfram",
          });
          break;

        case "expand":
          result = await computationRequest("/api/expand", {
            expression: expression.trim(),
            provider: "wolfram",
          });
          break;

        case "differentiate":
          result = await computationRequest("/api/differentiate", {
            expression: expression.trim(),
            variable: variable.trim() || "x",
            provider: "wolfram",
          });
          break;

        case "integrate":
          result = await computationRequest("/api/integrate", {
            expression: expression.trim(),
            variable: variable.trim() || "x",
            provider: "wolfram",
          });
          break;

        case "limit":
          result = await computationRequest("/api/limit", {
            expression: expression.trim(),
            variable: variable.trim() || "x",
            value: limitValue.trim() || "0",
            provider: "wolfram",
          });
          break;

        case "plot":
          result = await computationRequest("/api/plot", {
            expression: expression.trim(),
            variable: variable.trim() || "x",
            xmin: Number(xmin),
            xmax: Number(xmax),
            provider: "wolfram",
          });
          break;

        default:
          throw new Error("Unsupported operation.");
      }

      setResponse(result);
    } catch (error) {
      setResponse({
        success: false,
        operation,
        input: expression,
        result: null,
        error:
          error instanceof Error
            ? error.message
            : "Computation failed.",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="workspace">
      <div className="page-heading">
        <div>
          <div className="eyebrow">PURE WOLFRAM</div>
          <h1>Exact computation</h1>
          <p>
            Run a direct symbolic computation without the reasoning layer.
          </p>
        </div>
      </div>

      <div className="card computation-card">
        <div className="form-grid">
          <div>
            <label className="field-label" htmlFor="operation">
              Operation
            </label>

            <select
              id="operation"
              className="select-input"
              value={operation}
              onChange={(event) => {
                setOperation(event.target.value);
                setResponse(null);
              }}
            >
              {computationOperations.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          <div className="wide-field">
            <label
              className="field-label"
              htmlFor="wolfram-expression"
            >
              Expression
            </label>

            <input
              id="wolfram-expression"
              className="text-input"
              value={expression}
              onChange={(event) =>
                setExpression(event.target.value)
              }
              placeholder={
                operation === "solve"
                  ? "x^2 - 5*x + 6 == 0"
                  : "x^2 + 3*x"
              }
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  void compute();
                }
              }}
            />
          </div>

          {operation !== "simplify" &&
            operation !== "factor" &&
            operation !== "expand" && (
              <div>
                <label className="field-label" htmlFor="variable">
                  Variable
                </label>

                <input
                  id="variable"
                  className="text-input"
                  value={variable}
                  onChange={(event) =>
                    setVariable(event.target.value)
                  }
                  placeholder="x"
                />
              </div>
            )}

          {operation === "limit" && (
            <div>
              <label
                className="field-label"
                htmlFor="limit-value"
              >
                Approaches
              </label>

              <input
                id="limit-value"
                className="text-input"
                value={limitValue}
                onChange={(event) =>
                  setLimitValue(event.target.value)
                }
                placeholder="0"
              />
            </div>
          )}

          {operation === "plot" && (
            <>
              <div>
                <label className="field-label" htmlFor="xmin">
                  Minimum x
                </label>

                <input
                  id="xmin"
                  className="text-input"
                  type="number"
                  value={xmin}
                  onChange={(event) =>
                    setXmin(event.target.value)
                  }
                />
              </div>

              <div>
                <label className="field-label" htmlFor="xmax">
                  Maximum x
                </label>

                <input
                  id="xmax"
                  className="text-input"
                  type="number"
                  value={xmax}
                  onChange={(event) =>
                    setXmax(event.target.value)
                  }
                />
              </div>
            </>
          )}
        </div>

        <div className="input-footer">
          <span className="operation-description">
            {selectedOperation?.label ?? "Computation"}
          </span>

          <button
            className="primary-button"
            type="button"
            onClick={() => void compute()}
            disabled={!expression.trim() || loading}
          >
            {loading ? "Computing..." : "Compute"}
          </button>
        </div>
      </div>

      {loading && (
        <LoadingState text="Running the exact computation..." />
      )}

      {!loading && response && (
        <ComputationResult response={response} />
      )}
    </section>
  );
}
