
import type { ComputationResponse } from "../services/api";
import WorkspaceSolutionText from "./WorkspaceSolutionText";

function extractText(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value
      .map(extractText)
      .filter(Boolean)
      .join("\n");
  }

  if (typeof value === "object") {
    return Object.entries(
      value as Record<string, unknown>,
    )
      .map(([key, item]) => {
        const text = extractText(item);

        return text
          ? `${key.replace(/_/g, " ")}: ${text}`
          : "";
      })
      .filter(Boolean)
      .join("\n");
  }

  return String(value);
}

export function ComputationResult({
  response,
}: {
  response: ComputationResponse;
}) {
  if (!response.success) {
    return (
      <section className="result-card card error-card">
        <div className="result-heading">
          <span className="result-icon error">!</span>

          <div>
            <div className="eyebrow">RESULT</div>
            <h2>Computation failed</h2>
          </div>
        </div>

        <p className="error-message">
          {response.error ||
            "The computation could not be completed."}
        </p>
      </section>
    );
  }

  const result = extractText(response.result);

  return (
    <section className="result-card card">
      <div className="result-heading">
        <span className="result-icon success">✓</span>

        <div>
          <div className="eyebrow">SOLUTION</div>
          <h2>{response.operation}</h2>
        </div>
      </div>

      <div className="answer-block">
        <div className="answer-label">Result</div>

        <div className="answer-content">
          {result ? (
            <WorkspaceSolutionText>
              {result}
            </WorkspaceSolutionText>
          ) : (
            "No result returned."
          )}
        </div>
      </div>

      {response.plots &&
        response.plots.length > 0 && (
          <div className="steps-section">
            <div className="answer-label">
              Plot data
            </div>

            <div className="steps-content">
              <WorkspaceSolutionText>
                {extractText(response.plots)}
              </WorkspaceSolutionText>
            </div>
          </div>
        )}
    </section>
  );
}