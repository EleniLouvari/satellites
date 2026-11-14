/** Isolated browser fixture: all requests are mocked and no real run is accessed. */
import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import StopAction from "../src/StopAction";
import { type Run } from "../src/api";
import "../src/style.css";

const initial = { run_id: "ui-test", name: "Synthetic parcel analysis", status: "running" } as Run;
let requests = 0;
window.fetch = async (input, init) => {
  if (String(input) !== "/api/runs/ui-test/stop" || init?.method !== "POST") throw new Error("Unexpected fixture request");
  requests += 1;
  return new Response(JSON.stringify({ ...initial, stop_requested: true }), { status: 202 });
};
function Preview() {
  const [run, setRun] = useState(initial);
  return <main style={{ padding: 40 }}>
    <h1>Stop analysis UI test</h1>
    <p>This fixture uses synthetic state and mocked requests.</p>
    <StopAction run={run} onRequested={setRun} />
    <p role="status">State: {run.status}; requests: {requests}</p>
    {run.stop_requested && run.status === "running" && <button className="btn" onClick={() => setRun({ ...run, status: "interrupted", stop_requested: false })}>Simulate worker exit</button>}
    {run.status === "interrupted" && <button className="btn primary" onClick={() => setRun(initial)}>Resume analysis</button>}
  </main>;
}
createRoot(document.getElementById("root")!).render(<Preview />);
