import { useState } from "react";
import "./App.css";
import { StandardMode } from "./modes/StandardMode";
import { WolframMode } from "./modes/WolframMode";
import { GuidedAttemptMode } from "./modes/GuidedAttemptMode";

type Mode = "standard" | "wolfram" | "guided";

function App() {
  const [mode, setMode] = useState<Mode>("standard");

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>

          <div>
            <div className="brand-name">PrismOffline</div>
            <div className="brand-subtitle">
              Mathematical problem solving
            </div>
          </div>
        </div>

        <div
          className="mode-switcher"
          role="tablist"
          aria-label="Problem solving mode"
        >
          <button
            className={
              mode === "standard"
                ? "mode-button active"
                : "mode-button"
            }
            onClick={() => setMode("standard")}
            type="button"
          >
            Standard
          </button>

          <button
            className={
              mode === "wolfram"
                ? "mode-button active"
                : "mode-button"
            }
            onClick={() => setMode("wolfram")}
            type="button"
          >
            Pure Wolfram
          </button>

          <button
            className={
              mode === "guided"
                ? "mode-button active"
                : "mode-button"
            }
            onClick={() => setMode("guided")}
            type="button"
          >
            Guided Attempt
          </button>
        </div>
      </header>

      <main className="main-content">
        {mode === "standard" && <StandardMode />}
        {mode === "wolfram" && <WolframMode />}
        {mode === "guided" && <GuidedAttemptMode />}
      </main>

      <footer className="footer">
        <span>PrismOffline</span>
        <span>Local mathematical workspace</span>
      </footer>
    </div>
  );
}

export default App;
