import { useState } from "react";
import "./App.css";
import { StandardMode } from "./modes/StandardMode";
import { GraphicalMode } from "./modes/GraphicalMode";

type Mode = "standard" | "graphical";

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

        <div className="mode-switcher" role="tablist" aria-label="Problem solving mode">
          <button
            className={mode === "standard" ? "mode-button active" : "mode-button"}
            onClick={() => setMode("standard")}
            type="button"
          >
            Standard
          </button>
          <button
            className={mode === "graphical" ? "mode-button active" : "mode-button"}
            onClick={() => setMode("graphical")}
            type="button"
          >
            Graphical
          </button>
        </div>     </header>

      <main className="main-content">
        {mode === "standard" ? <StandardMode /> : <GraphicalMode />}
      </main>

      <footer className="footer">
        <span>PrismOffline</span>
        <span>Offline mathematical workspace</span>
      </footer>
    </div>
  );
}

export default App;
