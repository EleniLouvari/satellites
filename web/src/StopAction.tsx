import { useEffect, useRef, useState } from "react";
import { api, type Run } from "./api";
import { Button, Notice } from "./ui";

/** Stop local processing while retaining the original resumable analysis. */
export default function StopAction({ run, onRequested }: {
  run: Run;
  onRequested: (run: Run) => void;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open) dialog.current?.showModal();
  }, [open]);
  useEffect(() => {
    if (run.status !== "running" || run.stop_requested) setOpen(false);
  }, [run.status, run.stop_requested]);
  if (run.status !== "running") return null;

  async function stop() {
    setBusy(true);
    setError("");
    try {
      onRequested(await api<Run>(`/runs/${run.run_id}/stop`, "POST"));
      setOpen(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return <>
    <Button danger disabled={busy || run.stop_requested} onClick={() => setOpen(true)}>
      {run.stop_requested ? "Stopping…" : "Stop analysis"}
    </Button>
    {open && <dialog ref={dialog} className="confirm-dialog" aria-labelledby={`stop-${run.run_id}`}
      onCancel={(event) => { if (busy) event.preventDefault(); else setOpen(false); }}>
      <span className="eyebrow">Keep completed work</span>
      <h2 id={`stop-${run.run_id}`}>Stop analysis?</h2>
      <p><strong>{run.name}</strong></p>
      <p>Local processing will stop. Completed checkpoints will be kept so you can resume later.
        The unfinished stage may need to run again.</p>
      <p>Already-submitted openEO jobs may continue remotely. Other analyses and the website remain available.</p>
      <Notice message={error} />
      <div className="actions sectiongap">
        <button className="btn" autoFocus disabled={busy} onClick={() => setOpen(false)}>Keep running</button>
        <Button danger disabled={busy} onClick={() => void stop()}>{busy ? "Requesting stop…" : "Stop analysis"}</Button>
      </div>
    </dialog>}
  </>;
}
