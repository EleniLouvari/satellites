---
name: pytest-coverage
description: Use this skill when increasing pytest coverage, adding targeted regression tests, or identifying untested branches in a Python project.
---

# Pytest Coverage

## Purpose

Use this skill to raise meaningful coverage, not cosmetic coverage.

Best fit:
- add missing tests
- identify uncovered branches
- add regression tests for past bugs
- strengthen edge-case coverage
- preserve existing test style

---

## Workflow

1. Determine the target module or function.
2. Review existing tests before writing new ones.
3. Identify uncovered branches, error paths, and boundary cases.
4. Add the smallest useful set of tests.
5. Prefer tests that validate behavior, outputs, and side effects.
6. Re-check that tests align with project conventions.

---

## Coverage Priorities

Cover these first:
- public API behavior
- failure paths
- branch logic
- file/path edge cases
- empty input handling
- `None` and missing-key cases
- retries, skips, and partial-success flows
- output schema and metadata integrity

---

## Test Design Rules

- Follow the repository’s current pytest style.
- Reuse existing fixtures and helpers when available.
- Prefer one assertion theme per test.
- Name tests by behavior, not implementation detail.
- For bug fixes, add a regression test that fails before the patch.
- Avoid brittle tests that mirror implementation line-by-line.

---

## For Operational Pipelines

When the code touches:
- files
- cloud storage
- subprocesses
- databases
- catalog APIs
- raster outputs

then validate:
- ordering of side effects
- emitted paths and names
- expected status transitions
- pass/fail semantics
- idempotency where relevant

---

## Output

Return:
- the proposed new tests
- the specific branches or behaviors they cover
- any meaningful gaps that still remain
