import { useEffect, useState } from "react";
import "./App.css";
import { StandardMode } from "./modes/StandardMode";
import { GraphicalMode } from "./modes/GraphicalMode";
import { AuthView } from "./components/AuthView";
import { AccountView } from "./components/AccountView";
import { getCurrentLocalUser, logoutLocalAccount, type LocalUser } from "./services/api";

type Mode = "standard" | "graphical" | "account";

function App() {
  const [mode, setMode] = useState<Mode>("standard");
  const [user, setUser] = useState<LocalUser | null>(null);
  const [authChecking, setAuthChecking] = useState(true);\n  const [savedGuidedAttempt, setSavedGuidedAttempt] = useState<import("./services/api").SavedItem | null>(null);

  useEffect(() => {
    const token = localStorage.getItem("prismoffline_auth_token");
    if (!token) {
      setAuthChecking(false);
      return;
    }

    getCurrentLocalUser()
      .then(setUser)
      .catch(() => localStorage.removeItem("prismoffline_auth_token"))
      .finally(() => setAuthChecking(false));
  }, []);

  async function logout() {
    await logoutLocalAccount();
    setUser(null);
    setMode("standard");
  }

  if (authChecking) {
    return (
      <div className="app-shell">
        <main className="main-content">
          <section className="card auth-loading">Loading local account...</section>
        </main>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="app-shell">
        <main className="main-content">
          <AuthView onAuthenticated={(nextUser) => setUser(nextUser)} />
        </main>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>

          <div>
            <div className="brand-name">PrismOffline</div>
            <div className="brand-subtitle">
              STEM problem solving
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
          <button
            className={mode === "account" ? "mode-button active" : "mode-button"}
            onClick={() => setMode("account")}
            type="button"
          >
            Account
          </button>
        </div>
        <div className="account-chip">
          <span>{user.username}</span>
          <button className="account-logout-button" type="button" onClick={() => void logout()}>
            Log Out
          </button>
        </div>
      </header>

      <main className="main-content">
        {mode === "standard" ? (
          <StandardMode savedGuidedAttempt={savedGuidedAttempt} onSavedGuidedOpened={() => setSavedGuidedAttempt(null)} />
        ) : mode === "graphical" ? (
          <GraphicalMode />
        ) : (
          <AccountView user={user} onLogout={() => void logout()} onReattemptGuided={(item) => { setSavedGuidedAttempt(item); setMode("standard"); }} />
        )}
      </main>

      <footer className="footer">
        <span>PrismOffline</span>
        <span>Offline STEM workspace</span>
      </footer>
    </div>
  );
}

export default App;
