# Stop analysis implementation plan

**Goal:** Add an owned, resumable Stop analysis action without stopping the queue service or deleting data.

**Design:** Store an idempotent stop request in the active run/resume job's existing JSON payload. Keep its running claim and account reservations until local child processes have exited. Fence task publication after a stop request. The supervising worker stops the isolated task process tree, then existing reconciliation marks the analysis interrupted. Expose the pending request in run responses, show a confirmation dialog and Stopping state, and reuse Resume analysis. Already submitted remote openEO jobs are not cancelled. No schema migration or deployment of the active service is part of this change.

**Checks:** Ownership, CSRF, state validation, repeat requests, completion races, claim fencing, descendant termination, service survival, preserved checkpoints, resume availability, and frontend build. Use synthetic processes and disposable test state only.

- [x] Add API and worker regression tests and confirm initial failures.
- [x] Implement persisted stop request, publication fencing, and isolated process termination.
- [x] Add UI confirmation and pending state; document local versus remote stopping.
- [x] Run regression and UI checks and request independent review.

## Verification

The final parcel-statistics/web regression run passed 423 cases; five database
fixtures hit the registration rate limit accumulated by repeated test runs. Those
five passed against a fresh disposable database (428 cases verified in total).
The synthetic downloader fixture was updated to accept the scheduler's existing
`batch_workers` keyword. Automatic temporal block sizing now remains accepted
with the current 512-row default; tests explicitly request automatic sizing.

All 10 frontend tests and the production frontend build passed. Render tests cover
running, stopping and terminal button states. Browser automation was unavailable,
so the confirmation dialog has not been visually checked. A mocked manual fixture
is available at `web/tests/stop-preview.html` through the development server.

Real synthetic process tests verified Windows Job Object termination and Linux
process-group termination, including descendants surviving an early parent exit.
Unrelated processes and completed checkpoint files remained intact. Review found
and resolved the early-exit race and queue/analysis lock-order inversion; the
follow-up review reported no remaining material findings.

Changed-file Ruff, docstrings and whitespace checks passed. The repository quality
notebook generated Radon reports; new stop functions stay below its complexity
threshold. Bandit reported existing constant SQL interpolation and exception-handler
warnings in `runs.py`/`tasks.py`, with no findings in the new process-control modules.
The broad auto-fixing code-check script was not run against unrelated source files.

No production process, database, or run files were changed. The API, worker and
frontend must be rebuilt/restarted together before the live UI can use Stop.
