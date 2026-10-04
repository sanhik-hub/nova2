import type { ReasonResponse } from "../services/api";
import WorkspaceSolutionText from "./WorkspaceSolutionText";

function formatElapsed(elapsedMs: number): string {
  const totalSeconds = Math.max(0, elapsedMs) / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return minutes > 0
    ? `${minutes}m ${seconds.toFixed(1).padStart(4, "0")}s`
    : `${seconds.toFixed(1)}s`;
}

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

  if (response.result_only) {
    return (
      <section className="result-card card result-only-card">
        <div className="answer-content">
          {answer ? (
            <WorkspaceSolutionText>{answer}</WorkspaceSolutionText>
          ) : (
            "No result returned."
          )}
        </div>
      </section>
    );
  }

  return (
    <section className="result-card card">
      <div className="result-heading">
        <span className="result-icon success">?</span>

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

      {response.elapsed_ms !== undefined && (
        <div className="result-meta">
          <strong>Elapsed: {formatElapsed(response.elapsed_ms)}</strong>

          {response.cached && (
            <span>Cached result</span>
          )}
        </div>
      )}
    </section>
  );
}
