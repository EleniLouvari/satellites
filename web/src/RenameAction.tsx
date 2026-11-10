import { useEffect, useRef, useState } from "react";
import { api, type Run } from "./api";
import { Button, Notice } from "./ui";

/** Edit an analysis label independently of its processing configuration. */
export default function RenameAction({
  run,
  onRenamed,
}: {
  run: Run;
  onRenamed: (run: Run) => void;
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
      const renamed = await api<Run>(`/runs/${run.run_id}/name`, "PATCH", {
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
        disabled={busy || run.status === "deleting"}
        onClick={() => {
          setName(run.name);
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
          aria-labelledby={`rename-${run.run_id}`}
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
            <h2 id={`rename-${run.run_id}`}>Rename analysis</h2>
            <label>
              Analysis name
              <input
                autoFocus
                required
                maxLength={120}
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
                disabled={busy || !name.trim() || run.status === "deleting"}
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
