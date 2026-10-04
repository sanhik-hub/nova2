import { useEffect, useState } from "react";
import {
  reasonQuestion,
  startGuidedAttempt,
  type ExplanationPreference,
  type GuidedStep,
  type ReasonResponse,
  getLocalSettings,
  saveLocalItem,
  updateLocalItem,
  answerFollowUp,
  type FollowUpTurn,
} from "../services/api";
import { LoadingState } from "../components/LoadingState";
import { ReasonResult } from "../components/ReasonResult";
import MathText from "../components/MathText";
import WorkspaceSolutionText from "../components/WorkspaceSolutionText";

type TaskMode = "solution" | "guided";

type GuidedState = {
  query: string;
  steps: GuidedStep[];
  current: number;
  answers: Array<number | null>;
  score: number;
  finished: boolean;
  fullSolution: ReasonResponse | null;
  fullSolutionLoading: boolean;
  mcqElapsedMs: number;
  hintElapsedMs: number;
  finalSolutionElapsedMs: number | null;
};

export function StandardMode() {
  const [question, setQuestion] = useState("");
  const [taskMode, setTaskMode] = useState<TaskMode>("solution");
  const [explanation, setExplanation] =
    useState<ExplanationPreference>("detailed");
  const [includeHints, setIncludeHints] = useState(false);
  const [includeFullSolution, setIncludeFullSolution] = useState(true);
  const [response, setResponse] = useState<ReasonResponse | null>(null);
  const [guided, setGuided] = useState<GuidedState | null>(null);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [guidedLoading, setGuidedLoading] = useState(false);
  const [saveByDefault, setSaveByDefault] = useState(false);
  const [saveFollowupsByDefault, setSaveFollowupsByDefault] = useState(false);
  const [savedSolution, setSavedSolution] = useState(false);
  const [savedSolutionId, setSavedSolutionId] = useState<number | null>(null);
  const [followUpText, setFollowUpText] = useState("");
  const [followUps, setFollowUps] = useState<FollowUpTurn[]>([]);
  const [followUpLoading, setFollowUpLoading] = useState(false);

  const [savedGuided, setSavedGuided] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [cachedGuided, setCachedGuided] = useState<{
    query: string;
    includeHints: boolean;
    includeFullSolution: boolean;
    steps: GuidedStep[];
    fullSolution?: ReasonResponse | null;
  } | null>(null);

  const guidedCacheKey = "prismoffline-guided-attempt";

  useEffect(() => {
    void getLocalSettings()
       .then((settings) => {
        setSaveByDefault(settings.save_by_default);
        setSaveFollowupsByDefault(settings.save_followups_by_default);
      })
      .catch(() => setSaveByDefault(false));

    try {
      const raw = sessionStorage.getItem(guidedCacheKey);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (
        parsed &&
        typeof parsed.query === "string" &&
        Array.isArray(parsed.steps) &&
        parsed.steps.length > 0
      ) {
        setCachedGuided(parsed);
      }
    } catch {
      sessionStorage.removeItem(guidedCacheKey);
    }
  }, []);

  async function solve() {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setResponse(null);
    setGuided(null);
    setFollowUpText("");
    setFollowUps([]);
    setSavedSolutionId(null);

    try {
      const result = await reasonQuestion(trimmed, explanation);
      setResponse(result);
      setSavedSolution(false);
      setSaveError("");

      if (result.success && saveByDefault) {
        try {
          const saved = await saveLocalItem(
            "solution",
            trimmed.slice(0, 120),
            trimmed,
            {
              answer: typeof result.answer === "string" ? result.answer : String(result.answer ?? ""),
              response: result,
              explanation,
              follow_ups: [],
            },
          );
          setSavedSolution(true);
          setSavedSolutionId(saved.id);
        } catch (error) {
          setSaveError(error instanceof Error ? error.message : "Could not save the result.");
        }
      }
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

    const cached =
      cachedGuided &&
      cachedGuided.query === trimmed &&
      cachedGuided.includeHints === includeHints &&
      cachedGuided.includeFullSolution === includeFullSolution
        ? cachedGuided
        : null;

    try {
      const result = cached
        ? {
            success: true,
            query: cached.query,
            steps: cached.steps,
            include_hints: cached.includeHints,
            include_full_solution: cached.includeFullSolution,
            full_solution:
              typeof cached.fullSolution?.answer === "string"
                ? cached.fullSolution.answer
                : null,
            solution_generation_elapsed_ms:
              cached.fullSolution?.elapsed_ms ?? 0,
            elapsed_ms: 0,
            mcq_elapsed_ms: 0,
            hint_elapsed_ms: 0,
          }
        : await startGuidedAttempt(trimmed, includeHints, includeFullSolution);

      if (!result.success || !result.steps?.length) {
        setResponse({
          success: false,
          error: result.error || "Unable to generate the guided attempt.",
        });
        return;
      }

      const generatedSteps = result.steps;
      const fullSolution =
        typeof result.full_solution === "string" && result.full_solution.trim()
          ? {
              success: true,
              answer: result.full_solution,
              engine: "local_llm",
              tool_calls: [],
              cached: false,
              elapsed_ms: result.solution_generation_elapsed_ms ?? 0,
            }
          : null;

      if (!cached) {
        const cacheEntry = {
          query: trimmed,
          includeHints,
          includeFullSolution,
          steps: generatedSteps,
          fullSolution,
        };
        try {
          sessionStorage.setItem(guidedCacheKey, JSON.stringify(cacheEntry));
          setCachedGuided(cacheEntry);
        } catch {
          // Session caching is an optimization; a full attempt still works
          // when storage is unavailable or full.
        }
      }

      setGuided({
        query: trimmed,
        steps: generatedSteps,
        current: 0,
        answers: Array(result.steps.length).fill(null),
        score: 0,
        finished: false,
        fullSolution: cached?.fullSolution ?? fullSolution,
        fullSolutionLoading: false,
        mcqElapsedMs: result.mcq_elapsed_ms ?? result.elapsed_ms ?? 0,
        hintElapsedMs: result.hint_elapsed_ms ?? 0,
        finalSolutionElapsedMs:
          cached?.fullSolution?.elapsed_ms ??
          result.solution_generation_elapsed_ms ??
          null,
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

  function finishGuided() {
    if (!guided) return;

    const score = guided.answers.reduce<number>((total, answer, index) => {
      if (answer === null) return total;
      return total + (answer === guided.steps[index].correct_index ? 1 : 0);
    }, 0);

    const finishedGuided = {
      ...guided,
      finished: true,
      score,
    };
    setGuided(finishedGuided);
    setSavedGuided(false);
    setSaveError("");

    if (saveByDefault) {
      void saveLocalItem(
        "guided_attempt",
        guided.query.slice(0, 120),
        guided.query,
        {
          steps: guided.steps,
          answers: finishedGuided.answers,
          score,
          total_steps: guided.steps.length,
          include_hints: includeHints,
          include_full_solution: includeFullSolution,
          full_solution: finishedGuided.fullSolution?.answer ?? null,
          mcq_elapsed_ms: guided.mcqElapsedMs,
          hint_elapsed_ms: guided.hintElapsedMs,
          final_solution_elapsed_ms: guided.finalSolutionElapsedMs,
        },
      )
        .then(() => setSavedGuided(true))
        .catch((error) =>
          setSaveError(error instanceof Error ? error.message : "Could not save the attempt."),
        );
    }
  }

  async function saveSolutionManually() {
    if (!response?.success || savedSolution) return;
    setSaveError("");
    try {
      const saved = await saveLocalItem(
        "solution",
        question.trim().slice(0, 120),
        question.trim(),
        {
          answer: typeof response.answer === "string" ? response.answer : String(response.answer ?? ""),
          response,
          explanation,
          follow_ups: followUps,
        },
      );
      setSavedSolution(true);
      setSavedSolutionId(saved.id);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not save the result.");
    }
  }

  async function saveGuidedManually() {
    if (!guided?.finished || savedGuided) return;
    setSaveError("");
    try {
      await saveLocalItem(
        "guided_attempt",
        guided.query.slice(0, 120),
        guided.query,
        {
          steps: guided.steps,
          answers: guided.answers,
          score: guided.score,
          total_steps: guided.steps.length,
          include_hints: includeHints,
          include_full_solution: includeFullSolution,
          full_solution: guided.fullSolution?.answer ?? null,
          mcq_elapsed_ms: guided.mcqElapsedMs,
          hint_elapsed_ms: guided.hintElapsedMs,
          final_solution_elapsed_ms: guided.finalSolutionElapsedMs,
        },
      );
      setSavedGuided(true);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Could not save the attempt.");
    }
  }

  async function submitFollowUp() {
    const trimmed = followUpText.trim();
    if (!response?.success || !trimmed || followUpLoading) return;

    const originalAnswer =
      typeof response.answer === "string"
        ? response.answer
        : String(response.answer ?? "");

    setFollowUpLoading(true);
    setSaveError("");

    try {
      const result = await answerFollowUp(
        question.trim(),
        originalAnswer,
        followUps,
        trimmed,
      );

      if (!result.success) {
        setSaveError(result.error || "Could not answer the follow-up.");
        return;
      }

      const answer =
        typeof result.answer === "string"
          ? result.answer
          : String(result.answer ?? "");

      const nextHistory = [...followUps, { question: trimmed, answer }];
      setFollowUps(nextHistory);
      setFollowUpText("");

      if (saveFollowupsByDefault) {
        const payload = {
          answer: originalAnswer,
          response,
          explanation,
          follow_ups: nextHistory,
        };

        try {
          if (savedSolutionId !== null) {
            await updateLocalItem(
              savedSolutionId,
              question.trim().slice(0, 120),
              question.trim(),
              payload,
            );
          } else {
            const saved = await saveLocalItem(
              "solution",
              question.trim().slice(0, 120),
              question.trim(),
              payload,
            );
            setSavedSolutionId(saved.id);
          }
          setSavedSolution(true);
        } catch (error) {
          setSaveError(
            error instanceof Error
              ? error.message
              : "The follow-up was answered but could not be saved.",
          );
        }
      } else if (savedSolutionId !== null) {
        try {
          const saved = await updateLocalItem(
            savedSolutionId,
            question.trim().slice(0, 120),
            question.trim(),
            {
              answer: originalAnswer,
              response,
              explanation,
              follow_ups: nextHistory,
            },
          );
          setSavedSolution(true);
          setSavedSolutionId(saved.id);
        } catch {
          // Keep the live session usable even if a previously saved item
          // cannot be updated.
        }
      }
    } catch (error) {
      setSaveError(
        error instanceof Error
          ? error.message
          : "Could not answer the follow-up.",
      );
    } finally {
      setFollowUpLoading(false);
    }
  }

  function chooseOption(index: number) {
    if (!guided || selectedOption !== null) return;
    setSelectedOption(index);

    // Record the choice only. Correct/incorrect feedback is intentionally
    // withheld until the entire attempt is submitted.
    setGuided({
      ...guided,
      answers: guided.answers.map((answer, i) =>
        i === guided.current ? index : answer,
      ),
    });
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
          <h1>Ask a question or solve a problem</h1>
          <p>
            Ask a question or solve a problem in mathematics, physics, chemistry, or science. Choose a complete solution or a guided step-by-step attempt.
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
                placeholder="For example: What is Newton's second law?"
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
            <h2><MathText>{currentStep.question}</MathText></h2>

            <div className="guided-options">
              {currentStep.options.map((option, index) => {
                const selected = selectedOption === index;

                return (
                  <button
                    key={index}
                    className={[
                      "guided-option",
                      selected ? "selected" : "",
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
                    <span><MathText>{option.text}</MathText></span>
                  </button>
                );
              })}
            </div>

            {includeHints && selectedOption === null && currentStep.hint && (
              <details className="hint-box">
                <summary>Show Hint</summary>
                <p><MathText>{currentStep.hint}</MathText></p>
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

      {saveError && (
        <div className="card save-error-card">{saveError}</div>
      )}

      {!guided && !loading && !guidedLoading && response?.success && (
        <section className="card save-result-card">
          <span>
            {savedSolution
              ? "Saved locally to your account."
              : "Keep this question and result in your local account."}
          </span>
          <button
            className={savedSolution ? "secondary-button" : "primary-button"}
            type="button"
            onClick={() => void saveSolutionManually()}
            disabled={savedSolution}
          >
            {savedSolution ? "Saved" : "Save"}
          </button>
        </section>
      )}

      {!guidedLoading && guided?.finished && (
        <>
          <section className="card guided-timing-card">
            <div className="eyebrow">TIME ELAPSED</div>
            <div className="guided-timing-grid">
              <div>
                <strong>MCQs</strong>
                <span>{(guided.mcqElapsedMs / 1000).toFixed(2)} s</span>
              </div>
              {includeHints && (
                <div>
                  <strong>Hints</strong>
                  <span>{(guided.hintElapsedMs / 1000).toFixed(2)} s</span>
                  <small>Generated with the MCQ request</small>
                </div>
              )}
              {includeFullSolution && (
                <div>
                  <strong>Final solution</strong>
                  <span>
                    {guided.finalSolutionElapsedMs === null
                      ? "Still generating..."
                      : (guided.finalSolutionElapsedMs / 1000).toFixed(2) + " s"}
                  </span>
                </div>
              )}
            </div>
          </section>

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
            <div className="guided-result-actions">
              <button
                className={savedGuided ? "secondary-button" : "primary-button"}
                type="button"
                onClick={() => void saveGuidedManually()}
                disabled={savedGuided}
              >
                {savedGuided ? "Saved" : "Save Attempt"}
              </button>
              <button
                className="primary-button"
                type="button"
                onClick={() => {
                  const saved = guided;
                  if (!saved) return;
                  setGuided({
                    ...saved,
                    current: 0,
                    answers: Array(saved.steps.length).fill(null),
                    score: 0,
                    finished: false,
                    fullSolution: cachedGuided?.fullSolution ?? null,
                    fullSolutionLoading:
                      includeFullSolution && !cachedGuided?.fullSolution,
                    finalSolutionElapsedMs:
                      cachedGuided?.fullSolution?.final_solution_elapsed_ms ??
                      cachedGuided?.fullSolution?.elapsed_ms ??
                      null,
                  });
                  setSelectedOption(null);
                }}
              >
                Reattempt
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={resetGuided}
              >
                Exit Attempt
              </button>
            </div>
          </section>

          <section className="card guided-review-card">
            <div className="eyebrow">RESULTS</div>
            <h2>Step Results</h2>

            <div className="guided-review-list">
              {guided.steps.map((step, index) => {
                const answer = guided.answers[index];
                const correct = answer === step.correct_index;

                return (
                  <div
                    key={index}
                    className={
                      correct
                        ? "guided-review-item correct"
                        : "guided-review-item incorrect"
                    }
                  >
                    <div>
                      <strong>Step {index + 1}</strong>
                      <span>{correct ? "Correct" : "Incorrect"}</span>
                    </div>
                    {!correct && answer !== null && (
                      <small>
                        Correct answer: {step.options[step.correct_index].text}
                      </small>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {includeFullSolution && guided.fullSolution && (
            <ReasonResult response={guided.fullSolution} />
          )}

          {includeFullSolution && guided.fullSolutionLoading && (
            <LoadingState text="Finishing the complete solution..." />
          )}
        </>
      )}

      {!guided && !loading && !guidedLoading && response && (
        <>
          <ReasonResult response={response} />

          {response.success && (
            <section className="card follow-up-card">
              <div className="eyebrow">FOLLOW UP</div>
              <h2>Continue this question</h2>
              <p>
                Ask as many consecutive follow-up questions as you need. Previous
                answers in this question session are kept as context.
              </p>

              {followUps.length > 0 && (
                <div className="follow-up-history">
                  {followUps.map((turn, index) => (
                    <article className="follow-up-entry" key={index}>
                      <div className="follow-up-question">
                        <div className="eyebrow">FOLLOW-UP {index + 1}</div>
                        <MathText>{turn.question}</MathText>
                      </div>
                      <div className="follow-up-answer">
                        <div className="answer-label">Answer</div>
                        <WorkspaceSolutionText>{turn.answer}</WorkspaceSolutionText>
                      </div>
                    </article>
                  ))}
                </div>
              )}

              <textarea
                className="question-input follow-up-input"
                value={followUpText}
                onChange={(event) => setFollowUpText(event.target.value)}
                onKeyDown={(event) => {
                  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                    void submitFollowUp();
                  }
                }}
                placeholder="Ask a follow-up about this question..."
                rows={4}
                disabled={followUpLoading}
              />

              <div className="input-footer">
                <span className="shortcut">Ctrl + Enter to continue</span>
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => void submitFollowUp()}
                  disabled={!followUpText.trim() || followUpLoading}
                >
                  {followUpLoading ? "Working..." : "Ask Follow-up"}
                </button>
              </div>
            </section>
          )}
        </>
      )}
    </section>
  );
}
