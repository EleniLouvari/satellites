---
name: ruff-recursive-fix
description: Use this skill when a Python repository needs iterative Ruff cleanup, especially for lint fixes, complexity reduction, imports, and small behavior-preserving refactors.
---

# Ruff Recursive Fix

## Purpose

Use this skill when the task is to clean Python code with Ruff while keeping behavior unchanged.

Best fit:
- fix Ruff findings
- apply safe autofixes first
- reduce complexity warnings
- clean imports, dead code, redundant branches, and style issues
- convert repetitive cleanup into reviewable small patches

Do not use this skill for feature work or broad redesign.

---

## Workflow

1. Identify the narrowest scope possible.
2. Run Ruff or reason from the reported Ruff findings.
3. Apply safe fixes first.
4. Re-run mentally or via tooling and inspect the remaining findings.
5. For non-autofix issues, prefer minimal patches.
6. Preserve public interfaces, side effects, log messages, and output structure.
7. Group related fixes together; avoid mixing unrelated changes in one patch.

---

## Rules

- Prefer behavior-preserving edits.
- Do not rewrite whole files for style.
- Keep exception semantics unchanged unless the finding is a true bug.
- For complexity warnings, extract small private helpers instead of redesigning flows.
- For import cleanup, avoid removing imports that are used dynamically, conditionally, or by type-checking patterns.
- For unreachable-code reports, respect Python function boundaries.
- When the code is operational, preserve logging and failure markers.

---

## Preferred Fix Patterns

### Complexity
- flatten nested conditionals with guard clauses
- extract repeated blocks into private helpers
- centralize repeated validation
- replace copy-pasted branches with a small parameterized helper

### Style and correctness
- remove unused variables only after checking for debugging or sentinel usage
- replace redundant `else` after `return`
- simplify boolean expressions only if semantics remain identical
- normalize imports without changing lazy-import behavior

### Safety
- keep file I/O ordering stable
- keep side effects in the same sequence
- avoid changing return shapes

---

## Output

When using this skill, return:
- the minimal patch
- a brief note on which Ruff categories were addressed
- any findings intentionally left unchanged because they were risky
