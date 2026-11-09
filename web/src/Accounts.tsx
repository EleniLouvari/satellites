import { useEffect, useState } from "react";
import { api, date, type Account } from "./api";
import { Button, Empty, Heading, Notice, Password, Status } from "./ui";
export default function Accounts() {
  const [accounts, setAccounts] = useState<Account[]>([]),
    [editing, setEditing] = useState<Account | null | undefined>(undefined),
    [username, setUsername] = useState(""),
    [label, setLabel] = useState(""),
    [password, setPassword] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function refresh() {
    try {
      setAccounts(await api<Account[]>("/accounts"));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);
  async function action(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function edit(account: Account | null) {
    setEditing(account);
    setUsername(account?.username ?? "");
    setLabel(account?.label ?? "");
    setPassword("");
  }
  return (
    <>
      <Heading
        title="Copernicus accounts"
        subtitle="Your connections to satellite imagery. Available across your analyses."
      >
        <Button primary onClick={() => edit(null)}>
          + Add account
        </Button>
      </Heading>
      <Notice message={error} />
      {editing !== undefined && (
        <form
          className="panel pad sectiongap"
          onSubmit={(e) => {
            e.preventDefault();
            void action(async () => {
              await api(
                editing ? `/accounts/${editing.eo_user_id}` : "/accounts",
                editing ? "PATCH" : "POST",
                editing
                  ? { label, ...(password ? { password } : {}) }
                  : { username, label, password },
              );
              setEditing(undefined);
              setPassword("");
            });
          }}
        >
          <h2>{editing ? "Edit account" : "Add a Copernicus account"}</h2>
          <p className="muted">
            Passwords are encrypted on the server and never shown after saving.
          </p>
          <div className="formgrid sectiongap">
            <label>
              Label
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                maxLength={120}
              />
            </label>
            <label>
              Copernicus username
              <input
                required
                disabled={!!editing}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="off"
              />
            </label>
            <Password
              label={editing ? "Replacement password (optional)" : "Password"}
              required={!editing}
              value={password}
              onChange={setPassword}
            />
          </div>
          <div className="actions sectiongap">
            <Button primary type="submit" disabled={busy}>
              Save account
            </Button>
            <Button onClick={() => setEditing(undefined)}>Close</Button>
          </div>
        </form>
      )}
      <div className="accountgrid sectiongap">
        {accounts.map((account, index) => (
          <article className="accountcard" key={account.eo_user_id}>
            <header>
              <span className="avatar">
                {String(index + 1).padStart(2, "0")}
              </span>
              <Status value={account.test_status} />
            </header>
            <h2>{account.label || "Copernicus account"}</h2>
            <p>{account.username}</p>
            <small>Last tested: {date(account.tested_at)}</small>
            <div className="actions sectiongap">
              <Button
                disabled={busy}
                onClick={() =>
                  void action(() =>
                    api(`/accounts/${account.eo_user_id}/test`, "POST"),
                  )
                }
              >
                Test connection
              </Button>
              <Button onClick={() => edit(account)}>Edit</Button>
              <Button
                disabled={busy}
                onClick={() => {
                  if (
                    window.confirm(
                      "Delete this saved account? Referenced accounts cannot be deleted.",
                    )
                  )
                    void action(() =>
                      api(`/accounts/${account.eo_user_id}`, "DELETE"),
                    );
                }}
              >
                Delete
              </Button>
            </div>
          </article>
        ))}
      </div>
      {!accounts.length && (
        <Empty>
          <h2>Your imagery connections start here.</h2>
          <p>Add a Copernicus account to configure your first analysis.</p>
        </Empty>
      )}
      <p className="footnote">
        Connection tests check authentication only. They do not submit
        processing jobs or guarantee available quota.
      </p>
    </>
  );
}
