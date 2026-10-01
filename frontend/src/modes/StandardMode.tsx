import { useState } from "react";
import {
  reasonQuestion,
  startGuidedAttempt,
  type ExplanationPreference,
  type GuidedStep,
  type ReasonResponse,
} from "../services/api";
import { LoadingState } from "../components/LoadingState";
import { ReasonResult } from "../components/ReasonResult";

type TaskMode = "solution" | "guided";

type GuidedState = {
  query: string;
  steps: GuidedStep[];
  current: number;
  answers: Array<number | null>;
  score: number;
  finished: boolean;
  fullSolution: ReasonResponse | null;
};

export function StandardMode() {
  const [question, setQuestion] = useState("");
  const [taskMode, setTaskMode] = useState<TaskMode>("solution");
  const [explanation, setExplanation] =
    useState<ExplanationPreference>("detailed");
  const [includeHints, setIncludeHints] = useState(false);
  const [includeFullSolution, setIncludeFullSolution] = useState(false);
  const [response, setResponse] = useState<ReasonResponse | null>(null);
  const [guided, setGuided] = useState<GuidedState | null>(null);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [guidedLoading, setGuidedLoading] = useState(false);

  async function solve() {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setResponse(null);
    setGuided(null);

    try {
      const result = await reasonQuestion(trimmed, explanation);
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

  async function beginGuidedAttempt() {
    const trimmed = question.trim();
    if (!trimmed || guidedLoading) return;

    setGuidedLoading(true);
    setResponse(null);
    setGuided(null);
    setSelectedOption(null);

    try {
      const result = await startGuidedAttempt(trimmed, includeHints, includeFullSolution);

      if (!result.success || !result.steps?.length) {
        setResponse({
          success: false,
          error: result.error || "Unable to generate the guided attempt.",
        });
        return;
      }

      setGuided({
        query: trimmed,
        steps: result.steps,
        current: 0,
        answers: Array(result.steps.length).fill(null),
        score: 0,
        finished: false,
        fullSolution: null,
      });
    } catch (error) {
      setResponse({
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unable to generate the guided attempt.",
      });
    } finally {
      setGuidedLoading(false);
    }
  }

  async function finishGuided() {
    if (!guided) return;

    if (!includeFullSolution) {
      setGuided({ ...guided, finished: true });
      return;
    }

    setGuided({ ...guided, finished: true, fullSolution: null });
    setLoading(true);

    try {
      const result = await reasonQuestion(guided.query, "detailed");
      setGuided((current) =>
        current ? { ...current, fullSolution: result } : current,
      );
    } catch (error) {
      setGuided((current) =>
        current
          ? {
              ...current,
              fullSolution: {
                success: false,
                error:
                  error instanceof Error
                    ? error.message
                    : "Unable to generate the full solution.",
              },
            }
          : current,
      );
    } finally {
      setLoading(false);
    }
  }

  function chooseOption(index: number) {
    if (!guided || selectedOption !== null) return;
    setSelectedOption(index);

    const currentStep = guided.steps[guided.current];
    if (index === currentStep.correct_index) {
      setGuided({
        ...guided,
        answers: guided.answers.map((answer, i) =>
          i === guided.current ? index : answer,
        ),
        score: guided.score + 1,
      });
    } else {
      setGuided({
        ...guided,
        answers: guided.answers.map((answer, i) =>
          i === guided.current ? index : answer,
        ),
      });
    }
  }

  function nextGuidedStep() {
    if (!guided || selectedOption === null) return;

    if (guided.current >= guided.steps.length - 1) {
      void finishGuided();
      return;
    }

    setGuided({ ...guided, current: guided.current + 1 });
    setSelectedOption(null);
  }

  function resetGuided() {
    setGuided(null);
    setSelectedOption(null);
    setResponse(null);
  }

  const currentStep = guided && !guided.finished
    ? guided.steps[guided.current]
    : null;

  return (
    <section className="workspace">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STANDARD MODE</div>
          <h1>Solve a mathematical problem</h1>
          <p>
            Enter a question, then choose whether you want a complete solution
            or a guided step-by-step attempt.
          </p>
        </div>
      </div>

      {!guided && (
        <>
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
                    if (taskMode === "solution") void solve();
                    else void beginGuidedAttempt();
                  }
                }}
                placeholder="For example: Derive the formula for induced emf of a rotating rod"
                rows={7}
              />

              <div className="input-footer">
                <span className="shortcut">Ctrl + Enter to continue</span>
                <button
                  className="primary-button"
                  type="button"
                  onClick={() =>
                    taskMode === "solution"
                      ? void solve()
                      : void beginGuidedAttempt()
                  }
                  disabled={!question.trim() || loading || guidedLoading}
                >
                  {loading || guidedLoading
                    ? "Working..."
                    : taskMode === "solution"
                      ? "Solve"
                      : "Begin Guided Attempt"}
                </button>
              </div>
            </div>

            <div className="settings-card card">
              <div className="field-label">Choose what to do</div>

              <button
                className={
                  taskMode === "solution"
                    ? "choice-button selected"
                    : "choice-button"
                }
                type="button"
                onClick={() => setTaskMode("solution")}
              >
                <span className="choice-radio">
                  {taskMode === "solution" ? "●" : "○"}
                </span>
                <span>
                  <strong>Solution</strong>
                  <small>
                    Solve the question normally with the selected explanation
                    level.
                  </small>
                </span>
              </button>

              <button
                className={
                  taskMode === "guided"
                    ? "choice-button selected"
                    : "choice-button"
                }
                type="button"
                onClick={() => setTaskMode("guided")}
              >
                <span className="choice-radio">
                  {taskMode === "guided" ? "●" : "○"}
                </span>
                <span>
                  <strong>Guided Attempt</strong>
                  <small>
                    Work through progressive multiple-choice steps and receive
                    a score at the end.
                  </small>
                </span>
              </button>

              {taskMode === "solution" ? (
                <>
                  <div className="field-label guided-option-heading">
                    Explanation
                  </div>

                  {(["off", "concise", "detailed"] as ExplanationPreference[]).map(
                    (value) => (
                      <button
                        key={value}
                        className={
                          explanation === value
                            ? "choice-button selected"
                            : "choice-button"
                        }
                        type="button"
                        onClick={() => setExplanation(value)}
                      >
                        <span className="choice-radio">
                          {explanation === value ? "●" : "○"}
                        </span>
                        <span>
                          <strong>
                            {value.charAt(0).toUpperCase() + value.slice(1)}
                          </strong>
                          <small>
                            {value === "off"
                              ? "Return the result without an explanation."
                              : value === "concise"
                                ? "Short explanation with the important mathematics."
                                : "Complete explanation and derivation."}
                          </small>
                        </span>
                      </button>
                    ),
                  )}
                </>
              ) : (
                <>
                  <div className="field-label guided-option-heading">
                    Guided Attempt Options
                  </div>

                  <button
                    className={
                      includeHints
                        ? "choice-button selected"
                        : "choice-button"
                    }
                    type="button"
                    onClick={() => setIncludeHints((value) => !value)}
                  >
                    <span className="choice-radio">
                      {includeHints ? "●" : "○"}
                    </span>
                    <span>
                      <strong>Include hints</strong>
                      <small>Allow a hint to be revealed for each step.</small>
                    </span>
                  </button>

                  <button
                    className={
                      includeFullSolution
                        ? "choice-button selected"
                        : "choice-button"
                    }
                    type="button"
                    onClick={() => setIncludeFullSolution((value) => !value)}
                  >
                    <span className="choice-radio">
                      {includeFullSolution ? "●" : "○"}
                    </span>
                    <span>
                      <strong>Full solution at the end</strong>
                      <small>
                        Generate the complete solution after the final step.
                      </small>
                    </span>
                  </button>
                </>
              )}
            </div>
          </div>
        </>
      )}

      {(loading || guidedLoading) && (
        <LoadingState
          text={
            guidedLoading
              ? "Building the guided attempt..."
              : "Working through the problem..."
          }
        />
      )}

      {!guidedLoading && guided && currentStep && (
        <>
          <div className="progress-card card">
            <div className="progress-header">
              <span>Guided Attempt</span>
              <span>
                Step {guided.current + 1} of {guided.steps.length}
              </span>
            </div>

            <div className="progress-line">
              {guided.steps.map((_, index) => (
                <span key={index} style={{ display: "contents" }}>
                  <span
                    className={
                      index <= guided.current
                        ? "progress-dot active"
                        : "progress-dot"
                    }
                  >
                    {index + 1}
                  </span>
                  {index < guided.steps.length - 1 && (
                    <span className="progress-connector" />
                  )}
                </span>
              ))}
            </div>
          </div>

          <section className="card guided-step-card">
            <div className="eyebrow">STEP {guided.current + 1}</div>
            <h2>{currentStep.question}</h2>

            <div className="guided-options">
              {currentStep.options.map((option, index) => {
                const selected = selectedOption === index;
                const correct = selectedOption !== null &&
                  index === currentStep.correct_index;

                return (
                  <button
                    key={index}
                    className={[
                      "guided-option",
                      selected ? "selected" : "",
                      correct ? "correct" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    type="button"
                    onClick={() => chooseOption(index)}
                    disabled={selectedOption !== null}
                  >
                    <span className="guided-option-letter">
                      {String.fromCharCode(65 + index)}
                    </span>
                    <span>{option.text}</span>
                  </button>
                );
              })}
            </div>

            {selectedOption !== null && (
              <div
                className={
                  selectedOption === currentStep.correct_index
                    ? "guided-feedback correct"
                    : "guided-feedback"
                }
              >
                <strong>
                  {selectedOption === currentStep.correct_index
                    ? "Correct"
                    : "Not quite"}
                </strong>
              </div>
            )}

            {includeHints && selectedOption === null && currentStep.hint && (
              <details className="hint-box">
                <summary>Show Hint</summary>
                <p>{currentStep.hint}</p>
              </details>
            )}

            <div className="input-footer">
              <button
                className="secondary-button"
                type="button"
                onClick={resetGuided}
              >
                Exit Attempt
              </button>
              <button
                className="primary-button"
                type="button"
                onClick={nextGuidedStep}
                disabled={selectedOption === null}
              >
                {guided.current === guided.steps.length - 1
                  ? "Finish Attempt"
                  : "Next Step"}
              </button>
            </div>
          </section>
        </>
      )}

      {!guidedLoading && guided?.finished && (
        <>
          <section className="card guided-score-card">
            <div className="eyebrow">ATTEMPT COMPLETE</div>
            <h2>Score</h2>
            <div className="score-value">
              {guided.score} / {guided.steps.length}
            </div>
            <p>
              {guided.score === guided.steps.length
                ? "All steps were answered correctly."
                : "Review the steps you missed and try the problem again."}
            </p>
            <button
              className="secondary-button"
              type="button"
              onClick={resetGuided}
            >
              Try Another Attempt
            </button>
          </section>

          {includeFullSolution && guided.fullSolution && (
            <ReasonResult response={guided.fullSolution} />
          )}

          {includeFullSolution && !guided.fullSolution && loading && (
            <LoadingState text="Generating the complete solution..." />
          )}
        </>
      )}

      {!guided && !loading && !guidedLoading && response && (
        <ReasonResult response={response} />
      )}
    </section>
  );
}
