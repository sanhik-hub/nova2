import { useState } from "react";

export function GuidedAttemptMode() {
  const [question, setQuestion] = useState("");
  const [attempt, setAttempt] = useState("");
  const [started, setStarted] = useState(false);
  const [hintVisible, setHintVisible] = useState(false);

  function start() {
    if (!question.trim()) {
      return;
    }

    setStarted(true);
    setHintVisible(false);
  }

  return (
    <section className="workspace">
      <div className="page-heading">
        <div>
          <div className="eyebrow">GUIDED ATTEMPT</div>
          <h1>Work through the problem</h1>
          <p>
            Build your solution step by step and receive feedback
            along the way.
          </p>
        </div>
      </div>

      {!started ? (
        <div className="card guided-start-card">
          <label
            className="field-label"
            htmlFor="guided-question"
          >
            Question
          </label>

          <textarea
            id="guided-question"
            className="question-input"
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="Enter the problem you want to attempt."
            rows={7}
          />

          <div className="input-footer">
            <span className="shortcut">
              Work one step at a time
            </span>

            <button
              className="primary-button"
              type="button"
              onClick={start}
              disabled={!question.trim()}
            >
              Begin Attempt
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="progress-card card">
            <div className="progress-header">
              <span>Attempt progress</span>
              <span>Step 1 of 4</span>
            </div>

            <div className="progress-line">
              <span className="progress-dot active">1</span>
              <span className="progress-connector" />
              <span className="progress-dot">2</span>
              <span className="progress-connector" />
              <span className="progress-dot">3</span>
              <span className="progress-connector" />
              <span className="progress-dot">4</span>
            </div>
          </div>

          <div className="guided-grid">
            <section className="card guided-question-card">
              <div className="field-label">Question</div>
              <div className="problem-display">
                {question}
              </div>
            </section>

            <section className="card">
              <label className="field-label" htmlFor="attempt">
                Your attempt
              </label>

              <textarea
                id="attempt"
                className="question-input"
                value={attempt}
                onChange={(event) =>
                  setAttempt(event.target.value)
                }
                placeholder="Enter your next step or calculation."
                rows={7}
              />

              <div className="input-footer">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() =>
                    setHintVisible((visible) => !visible)
                  }
                >
                  {hintVisible ? "Hide Hint" : "Hint"}
                </button>

                <button
                  className="primary-button"
                  type="button"
                  disabled={!attempt.trim()}
                  onClick={() => {}}
                >
                  Submit Attempt
                </button>
              </div>
            </section>
          </div>

          {hintVisible && (
            <section className="card feedback-card">
              <div className="eyebrow">HINT</div>
              <h2>
                Think about the type of problem first.
              </h2>
              <p>
                Identify the mathematical concept and the method
                you would use before carrying out the next
                calculation.
              </p>
            </section>
          )}

          <section className="card feedback-card muted-card">
            <div className="eyebrow">FEEDBACK</div>
            <p>
              Guided attempt evaluation will connect to the guided
              backend once its API contract is implemented. Your
              attempt is not being silently evaluated by a
              placeholder.
            </p>
          </section>
        </>
      )}
    </section>
  );
}
