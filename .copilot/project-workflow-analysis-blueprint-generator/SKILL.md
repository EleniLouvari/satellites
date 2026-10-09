---
name: project-workflow-analysis-blueprint-generator
description: Use this skill when documenting end-to-end workflow behavior, including entry points, sequence of steps, branching, failure handling, retries, outputs, and operational checkpoints.
---

# Project Workflow Analysis Blueprint Generator

## Purpose

Use this skill to document how the system actually behaves from start to finish.

Best fit:
- explain an end-to-end workflow
- map CLI/API/job execution
- document branch conditions
- clarify failure handling and partial-success rules
- make operational flows reviewable

---

## Workflow

1. Identify the triggering entry point.
2. Describe setup and validation.
3. Trace the normal path in order.
4. Capture major branches and mode switches.
5. Record side effects at each step.
6. Document failure paths and retry/skip behavior.
7. Record final outputs and success criteria.

---

## What to Capture

### Entry points
- CLI commands
- API endpoints
- scheduled jobs
- background workers
- scripts

### Execution stages
- setup
- discovery/search
- download or input acquisition
- processing/transformation
- persistence or upload
- metadata generation
- registration/reporting
- cleanup

### Failure behavior
- fail-fast cases
- recoverable cases
- partial-success cases
- logging and metrics
- final status markers

---

## Output

Produce:
- a numbered workflow
- decision points
- failure branches
- output inventory
- a concise “how to debug this flow” section
