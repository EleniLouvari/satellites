import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Button, Notice } from "./ui";

/** Edit an analysis label independently of its processing configuration. */
export default function RenameAction<T>({
  id,
  name: current,
  path,
  status,
  onRenamed,
  noun = "analysis",
  maxLength = 120,
}: {
  id: string;
  name: string;
  path: string;
  status: string;
  onRenamed: (row: T) => void;
  noun?: string;
  maxLength?: number;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open) dialog.current?.showModal();
  }, [open]);

  /** Save only the name and keep the dialog open when the request fails. */
  async function save() {
    setBusy(true);
    setError("");
    try {
      const renamed = await api<T>(path, "PATCH", {
        name: name.trim(),
      });
      onRenamed(renamed);
      setOpen(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Button
        disabled={busy || status === "deleting"}
        onClick={() => {
          setName(current);
          setError("");
          setOpen(true);
        }}
      >
        Rename
      </Button>
      {open && (
        <dialog
          ref={dialog}
          className="confirm-dialog"
          aria-labelledby={`rename-${id}`}
          onCancel={(e) => {
            if (busy) e.preventDefault();
            else setOpen(false);
          }}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!busy && name.trim()) void save();
            }}
          >
            <h2 id={`rename-${id}`}>Rename {noun}</h2>
            <label>
              {noun[0].toUpperCase() + noun.slice(1)} name
              <input
                autoFocus
                className="name-input"
                required
                maxLength={maxLength}
                value={name}
                disabled={busy}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <Notice message={error} />
            <div className="actions sectiongap">
              <Button disabled={busy} onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button
                primary
                type="submit"
                disabled={busy || !name.trim() || status === "deleting"}
              >
                {busy ? "Saving…" : "Save name"}
              </Button>
            </div>
          </form>
        </dialog>
      )}
    </>
  );
}
