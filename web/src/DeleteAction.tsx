import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Button, Notice } from "./ui";

/** Confirm the precise scope before an authenticated irreversible deletion. */
export default function DeleteAction({
  kind,
  id,
  name,
  status,
  onDeleted,
}: {
  kind: "run" | "eda";
  id: string;
  name: string;
  status: string;
  onDeleted: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const disabled = ![
    "draft",
    "ready",
    "complete",
    "failed",
    "interrupted",
    "cancelled",
    "deleting",
  ].includes(status);
  useEffect(() => {
    if (open) dialog.current?.showModal();
  }, [open]);
  async function remove() {
    setBusy(true);
    setError("");
    try {
      await api(`/${kind === "run" ? "runs" : "eda"}/${id}`, "DELETE");
      setOpen(false);
      onDeleted();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <span
        title={
          disabled
            ? "Wait for active work to finish before deleting"
            : "Permanently delete this analysis"
        }
      >
        <Button
          danger
          disabled={disabled || busy}
          onClick={() => setOpen(true)}
        >
          {status === "deleting" ? "Retry deletion" : "Delete"}
        </Button>
      </span>
      {open && (
        <dialog
          ref={dialog}
          className="confirm-dialog"
          aria-labelledby={`delete-${id}`}
          onCancel={(e) => {
            if (busy) e.preventDefault();
            else setOpen(false);
          }}
        >
          <span className="eyebrow">Permanent deletion</span>
          <h2 id={`delete-${id}`}>
            Delete {kind === "run" ? "run" : "EDA analysis"}?
          </h2>
          <p>
            <strong>{name}</strong>
          </p>
          <p>
            {kind === "run"
              ? "This removes the run, all its EDA analyses, results, logs, cached files and database history. Its tracked remote openEO jobs and results are also deleted, including retry attempts."
              : "This removes this EDA report, charts, tables, ZIP, logs and saved configuration. The parcel run and other EDA analyses are kept."}
          </p>
          {kind === "run" && (
            <p>
              Uploaded parcels are removed only when no other run uses them.
              Your Copernicus accounts are kept.
            </p>
          )}
          <p>This cannot be undone.</p>
          <Notice message={error} />
          {busy && (
            <p role="status">
              Cleaning up… Remote cleanup can take several minutes. This window
              will close when deletion finishes.
            </p>
          )}
          <div className="actions sectiongap">
            <button
              className="btn"
              autoFocus
              disabled={busy}
              onClick={() => setOpen(false)}
            >
              Keep {kind === "run" ? "run" : "analysis"}
            </button>
            <Button danger disabled={busy} onClick={() => void remove()}>
              {busy ? "Deleting…" : "Delete permanently"}
            </Button>
          </div>
        </dialog>
      )}
    </>
  );
}
