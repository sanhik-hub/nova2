import type { ReasonResponse } from "../services/api";
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

  return "";
}

function extractStepContent(step: unknown): string {
  if (
    typeof step === "object" &&
    step !== null &&
    "content" in step
  ) {
    return extractText(
      (step as { content?: unknown }).content,
    );
  }

  return extractText(step);
}

export function ReasonResult({
  response,
}: {
  response: ReasonResponse;
}) {
  if (!response.success) {
    return (
      <section className="result-card card error-card">
        <div className="result-heading">
          <span className="result-icon error">!</span>

          <div>
            <div className="eyebrow">RESULT</div>
            <h2>Unable to solve this question</h2>
          </div>
        </div>

        <p className="error-message">
          {response.error ||
            "The question could not be interpreted."}
        </p>
      </section>
    );
  }

  const answer = extractText(response.answer);

  const steps = (response.steps ?? [])
    .map(extractStepContent)
    .map((step) => step.trim())
    .filter(Boolean);

  return (
    <section className="result-card card">
      <div className="result-heading">
        <span className="result-icon success">✓</span>

        <div>
          <div className="eyebrow">SOLUTION</div>
          <h2>Result</h2>
        </div>
      </div>

      <div className="answer-block">
        <div className="answer-label">Answer</div>

        <div className="answer-content">
          {answer ? (
            <WorkspaceSolutionText>
              {answer}
            </WorkspaceSolutionText>
          ) : (
            "No result returned."
          )}
        </div>
      </div>

      {steps.length > 0 && (
        <div className="steps-section">
          <div className="answer-label">Steps</div>

          <div className="steps-content">
            {steps.map((step, index) => (
              <WorkspaceSolutionText
                key={index}
              >
                {step}
              </WorkspaceSolutionText>
            ))}
          </div>
        </div>
      )}

      {response.elapsed_ms !== undefined && (
        <div className="result-meta">
          <span>{response.elapsed_ms} ms</span>

          {response.cached && (
            <span>Cached result</span>
          )}
        </div>
      )}
    </section>
  );
}
