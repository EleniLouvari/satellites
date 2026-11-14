# Temporal chunk processing implementation plan

**Goal:** Reduce repeated NetCDF chunk access during temporal filling while preserving all scientific outputs and caches.

**Design:** Keep the existing checkpoint layout and process spatial blocks aligned to its chunks. Automatic blocks target 64 MiB of float32 data across all dates. The explicit row setting remains an upper limit; larger row counts round down to whole chunk rows. Small strips finish one spatial chunk before moving on. Split wide rasters into column blocks. Temporarily size the cleaned and temporal-mask caches to the accessed chunk set, capped at 64 MiB each, then restore them even on exceptions. No production container or run files are changed.

**Validation:** Numerical equivalence, provenance, coverage guards, uneven edges, configured rows, bounded automatic blocks, cache restoration, existing checkpoint reuse, and an isolated reproducible before/after benchmark. Memory targets exclude working copies; benchmarks do not predict production speedups.

- [x] Add failing block-alignment, numerical-equivalence and cache-lifecycle tests.
- [x] Implement block planning and bounded temporary caching in `core/streamed_raster.py`.
- [x] Run regression tests and benchmark against the previous full-width strip implementation.
- [x] Update configuration descriptions and NetCDF documentation; review and verify.

## Results

Parcel-statistics and web suites: 393 passed, 19 disposable-database tests skipped.
Changed-source Ruff, pydocstyle and Bandit checks passed; Radon reports generated.
Independent read-only review found no material issues.

An isolated Docker container (one CPU, 2 GiB memory, no network) processed identical
synthetic 135-date, one-band, 256 x 512 checkpoints on a Windows bind mount.
Alternating legacy/aligned/aligned/legacy runs, including flush/close, took
11.142/1.602/1.651/9.947 seconds. Mean speedup was 6.48x, with identical SHA-256
hashes of values, temporal provenance and null-count changes. Logical reads fell
from 1.438 GB to 0.075 GB per run. These are process I/O counters, not physical
disk traffic, and this temporal-stage measurement does not predict end-to-end
production speed. The benchmark script and raw results are retained locally in
`outputs/diagnostics/temporal_chunk_benchmark.py` and
`outputs/diagnostics/temporal-chunk-results-1/results.json`.

No production runs or containers were changed. Deployment requires rebuilding
the application image and using it in a subsequent worker process.
