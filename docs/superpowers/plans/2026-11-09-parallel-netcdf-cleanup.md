# Parallel NetCDF cleanup

**Goal:** Honor `batch_workers` with `remove_nc_after_completion=True` in direct,
partition and shared web scheduling. This implements the behavior requested in
the discussion: each worker saves both batch tables before deleting its rasters.

**Approach:** Reuse the existing checkpoint lifecycle in spawned processes. Direct
calls use up to `batch_workers` processes per partition. Shared cleanup uses one
run-wide pool of that size, fed by download threads with bounded submissions.
Keep single-worker execution synchronous and retain existing cache signatures.
Return batch IDs from cleanup workers, loading tables only for final assembly.

**Constraints:** Preserve unrelated working-tree changes and existing run data.
Do not start remote jobs or deploy/restart the running web application.

- [x] Add real-process regression tests in `test_parallel_cleanup.py`: four workers
  rendezvous, including across four partitions fed by one remote account slot;
  check correct statistics, cleanup, failed-save recovery and cached resume.
- [x] Verify the regression fails against sequential cleanup.
- [x] Update `core/base.py` to use the existing process pool with checkpoint-only
  workers when cleanup is enabled, keeping deterministic output ordering.
- [x] Update `batch_scheduler.py` to feed a bounded process pool while downloads
  continue; wait for every checkpoint before partition publication and propagate
  local failures. Use spawn to avoid forking a multithreaded NetCDF process.
- [x] Run cleanup/scheduler suites, then the complete repository test suite;
  inspect scoped lint and complexity results and report unrelated failures.
- [x] Update scheduling and NetCDF documentation with actual worker limits,
  storage/memory tradeoffs and failure semantics.

**Review focus:** Single worker; one account with several local workers; cached
and newly downloaded batches; worker/write failures; no partial publication or
deleted recovery inputs; no retained DataFrames in checkpoint futures.

## Verification record

- Four-process rendezvous failed against sequential cleanup, then passed with
  real spawned processes and real NetCDF/Parquet persistence. Successful tests
  process eight batches in two waves and verify exactly four distinct workers.
- Empty/cached-only cleanup initially started redundant pools; regression tests
  now confirm assembly does not spawn any processes.
- Initial cleanup/scheduler suite: 32 passed. Complete repository suite in
  `.venv-checks`: 659 passed, 19 disposable-database tests skipped, two dependency
  warnings. The first full attempt under the default Python lacked FastAPI;
  the first complete-environment run exposed a CLI mock signature needing the
  new argument, corrected before the successful full run.
- Independent review found a direct-Python compatibility issue with textual
  `batch_workers="4"`. Reproduced it, passed the validated extractor's integer
  to the scheduler, then reran all six focused regressions successfully.
- Scoped Ruff and whitespace checks pass. Executed the code-quality notebook
  cells without modifying the notebook. Modified production modules retain
  maintainability grade A; batch-statistics complexity decreased from 14 to 11.
- Existing run outputs and deployed containers were not changed. Rebuild and
  restart the web worker to use the new implementation for subsequent runs.
