import { useState } from "react";
import "./App.css";
import { StandardMode } from "./modes/StandardMode";

type Mode = "standard";

function App() {

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
          <button className="mode-button active" type="button">
            Standard
          </button>
        </div>     </header>

      <main className="main-content">
        <StandardMode />
      </main>

      <footer className="footer">
        <span>PrismOffline</span>
        <span>Offline mathematical workspace</span>
      </footer>
    </div>
  );
}

export default App;
