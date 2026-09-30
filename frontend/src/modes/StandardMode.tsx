import { useState } from "react";
import { reasonQuestion, type ExplanationPreference, type ReasonResponse } from "../services/api";
import { LoadingState } from "../components/LoadingState";
import { ReasonResult } from "../components/ReasonResult";

export function StandardMode() {
  const [question, setQuestion] = useState("");
  const [explanation, setExplanation] =
    useState<ExplanationPreference>("detailed");
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
        explanation,
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

          <button className={explanation === "off" ? "choice-button selected" : "choice-button"} type="button" onClick={() => setExplanation("off")}>
            <span className="choice-radio">{explanation === "off" ? "●" : "○"}</span>
            <span><strong>Off</strong><small>Return the result without an explanation.</small></span>
          </button>

          <button className={explanation === "concise" ? "choice-button selected" : "choice-button"} type="button" onClick={() => setExplanation("concise")}>
            <span className="choice-radio">{explanation === "concise" ? "●" : "○"}</span>
            <span><strong>Concise</strong><small>Very short explanation with the important mathematics.</small></span>
          </button>

          <button className={explanation === "detailed" ? "choice-button selected" : "choice-button"} type="button" onClick={() => setExplanation("detailed")}>
            <span className="choice-radio">{explanation === "detailed" ? "●" : "○"}</span>
            <span><strong>Detailed</strong><small>Complete explanation and derivation.</small></span>
          </button>

          <div className="info-box">
            Concise keeps the mathematical expressions while minimizing text. Detailed gives the complete solution.
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
