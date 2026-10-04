import { useEffect, useState } from "react";
import {
  deleteLocalItem,
  getLocalSettings,
  listLocalItems,
  updateLocalSettings,
  type LocalSettings,
  type LocalUser,
  type SavedItem,
} from "../services/api";
import MathText from "./MathText";

type AccountViewProps = {
  user: LocalUser;
  onLogout: () => void;
};

export function AccountView({ user, onLogout }: AccountViewProps) {
  const [settings, setSettings] = useState<LocalSettings>({ save_by_default: false });
  const [items, setItems] = useState<SavedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const [nextSettings, nextItems] = await Promise.all([
        getLocalSettings(),
        listLocalItems(),
      ]);
      setSettings(nextSettings);
      setItems(nextItems);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load local account data.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function toggleDefaultSave() {
    const next = { save_by_default: !settings.save_by_default };
    setSettings(next);
    try {
      const saved = await updateLocalSettings(next);
      setSettings(saved);
    } catch (err) {
      setSettings(settings);
      setError(err instanceof Error ? err.message : "Could not update settings.");
    }
  }

  async function remove(id: number) {
    try {
      await deleteLocalItem(id);
      setItems((current) => current.filter((item) => item.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete the saved item.");
    }
  }

  return (
    <section className="account-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">LOCAL ACCOUNT</div>
          <h1>{user.username}</h1>
          <p>Account data, saving preferences, and your saved STEM work.</p>
        </div>
      </div>

      <div className="account-grid">
        <section className="card account-settings-card">
          <div className="eyebrow">SETTINGS</div>
          <h2>Storage</h2>

          <button
            className={settings.save_by_default ? "choice-button selected" : "choice-button"}
            type="button"
            onClick={() => void toggleDefaultSave()}
          >
            <span className="choice-radio">{settings.save_by_default ? "●" : "○"}</span>
            <span>
              <strong>Save results by default</strong>
              <small>
                Automatically save complete solutions and finished Guided Attempts,
                including questions, options, answers, score, and full solution.
              </small>
            </span>
          </button>

          <div className="info-box">
            When this is off, every completed result has a Save button so you can
            choose exactly what to keep.
          </div>

          <button className="secondary-button account-logout" type="button" onClick={onLogout}>
            Log Out
          </button>
        </section>

        <section className="card saved-items-card">
          <div className="saved-header">
            <div>
              <div className="eyebrow">SAVED WORK</div>
              <h2>Questions and results</h2>
            </div>
            <button className="secondary-button" type="button" onClick={() => void refresh()}>
              Refresh
            </button>
          </div>

          {error && <div className="auth-error">{error}</div>}
          {loading ? (
            <p className="saved-empty">Loading local storage...</p>
          ) : items.length === 0 ? (
            <p className="saved-empty">
              Nothing saved yet. Solve a question and use Save, or enable Save results by default.
            </p>
          ) : (
            <div className="saved-list">
              {items.map((item) => (
                <details className="saved-item" key={item.id}>
                  <summary>
                    <span>
                      <strong>{item.title}</strong>
                      <small>{item.kind === "guided_attempt" ? "Guided Attempt" : "Solution"} · {new Date(item.created_at).toLocaleString()}</small>
                    </span>
                  </summary>
                  <div className="saved-question"><MathText>{item.question}</MathText></div>
                  {item.kind === "solution" && typeof item.payload.answer === "string" && (
                    <div className="saved-answer"><MathText>{item.payload.answer}</MathText></div>
                  )}
                  {item.kind === "guided_attempt" && (
                    <div className="saved-guided-summary">
                      <strong>
                        Score: {String(item.payload.score ?? 0)} / {String(item.payload.total_steps ?? 0)}
                      </strong>
                      {typeof item.payload.full_solution === "string" && (
                        <div className="saved-answer"><MathText>{item.payload.full_solution}</MathText></div>
                      )}
                    </div>
                  )}
                  <button className="secondary-button saved-delete" type="button" onClick={() => void remove(item.id)}>
                    Delete
                  </button>
                </details>
              ))}
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
