import { useState } from "react";
import { reasonQuestion, type ReasonResponse } from "../services/api";
import { LoadingState } from "../components/LoadingState";
import { ReasonResult } from "../components/ReasonResult";

type ExplanationPreference = "with" | "without";

export function StandardMode() {
  const [question, setQuestion] = useState("");
  const [explanation, setExplanation] =
    useState<ExplanationPreference>("with");
  const [response, setResponse] = useState<ReasonResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function solve() {
    const trimmed = question.trim();

    if (!trimmed || loading) {
      return;
    }

    setLoading(true);
    setResponse(null);

    try {
      const result = await reasonQuestion(
        trimmed,
        explanation === "with",
      );

      setResponse(result);
    } catch (error) {
      setResponse({
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unable to solve this question.",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="workspace">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STANDARD MODE</div>
          <h1>Solve a mathematical problem</h1>
          <p>
            Ask a question in natural language or enter a mathematical
            expression.
          </p>
        </div>
      </div>

      <div className="workspace-grid">
        <div className="input-card card">
          <label className="field-label" htmlFor="standard-question">
            Question
          </label>

          <textarea
            id="standard-question"
            className="question-input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (
                (event.ctrlKey || event.metaKey) &&
                event.key === "Enter"
              ) {
                void solve();
              }
            }}
            placeholder="For example: Differentiate x^2 + 3x"
            rows={7}
          />

          <div className="input-footer">
            <span className="shortcut">Ctrl + Enter to solve</span>

            <button
              className="primary-button"
              type="button"
              onClick={() => void solve()}
              disabled={!question.trim() || loading}
            >
              {loading ? "Solving..." : "Solve"}
            </button>
          </div>
        </div>

        <div className="settings-card card">
          <div className="field-label">Explanation</div>

          <button
            className={
              explanation === "with"
                ? "choice-button selected"
                : "choice-button"
            }
            type="button"
            onClick={() => setExplanation("with")}
          >
            <span className="choice-radio">
              {explanation === "with" ? "●" : "○"}
            </span>

            <span>
              <strong>With explanation</strong>
              <small>Show reasoning and relevant steps.</small>
            </span>
          </button>

          <button
            className={
              explanation === "without"
                ? "choice-button selected"
                : "choice-button"
            }
            type="button"
            onClick={() => setExplanation("without")}
          >
            <span className="choice-radio">
              {explanation === "without" ? "●" : "○"}
            </span>

            <span>
              <strong>Without explanation</strong>
              <small>Prefer a concise computational result.</small>
            </span>
          </button>

          <div className="info-box">
            Explanation preference is sent directly with the request.
          </div>
        </div>
      </div>

      {loading && (
        <LoadingState text="Working through the problem..." />
      )}

      {!loading && response && (
        <ReasonResult response={response} />
      )}
    </section>
  );
}
