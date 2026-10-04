import { FormEvent, useState } from "react";
import {
  loginLocalAccount,
  registerLocalAccount,
  type LocalUser,
} from "../services/api";

type AuthViewProps = {
  onAuthenticated: (user: LocalUser, token: string) => void;
};

export function AuthView({ onAuthenticated }: AuthViewProps) {
  const [registering, setRegistering] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setError("");

    if (registering && password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setBusy(true);
    try {
      const result = registering
        ? await registerLocalAccount(username, password)
        : await loginLocalAccount(username, password);

      localStorage.setItem("prismoffline_auth_token", result.token);
      onAuthenticated(result.user, result.token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="auth-shell">
      <div className="auth-card card">
        <div className="eyebrow">LOCAL ACCOUNT</div>
        <h1>{registering ? "Create your account" : "Welcome back"}</h1>
        <p>
          Your account and saved STEM work stay on this computer. No cloud
          account is required.
        </p>

        <form onSubmit={submit} className="auth-form">
          <label className="field-label" htmlFor="auth-username">Username</label>
          <input
            id="auth-username"
            className="text-input"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            required
          />

          <label className="field-label" htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            className="text-input"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete={registering ? "new-password" : "current-password"}
            required
          />

          {registering && (
            <>
              <label className="field-label" htmlFor="auth-confirm">Confirm password</label>
              <input
                id="auth-confirm"
                className="text-input"
                type="password"
                value={confirm}
                onChange={(event) => setConfirm(event.target.value)}
                autoComplete="new-password"
                required
              />
            </>
          )}

          {error && <div className="auth-error">{error}</div>}

          <button className="primary-button auth-submit" type="submit" disabled={busy}>
            {busy ? "Please wait..." : registering ? "Create Account" : "Log In"}
          </button>
        </form>

        <button
          className="secondary-button auth-switch"
          type="button"
          onClick={() => {
            setRegistering((value) => !value);
            setError("");
          }}
        >
          {registering ? "Already have an account? Log in" : "Create a local account"}
        </button>
      </div>
    </section>
  );
}
