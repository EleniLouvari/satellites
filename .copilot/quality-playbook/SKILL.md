---
name: quality-playbook
description: Use this skill when defining a repository quality strategy: review checklists, test priorities, release confidence criteria, regression protections, and failure analysis workflows.
---

# Quality Playbook

## Purpose

Use this skill to create or improve a practical quality framework for a software project.

Best fit:
- define release gates
- document review checklists
- identify quality risks
- map tests to business-critical workflows
- propose QA additions for confidence in outputs

---

## Core Principle

Quality is not just test count. It is confidence that the system:
- behaves correctly
- fails safely
- preserves invariants
- surfaces diagnostics
- remains maintainable under change

---

## Workflow

1. Identify the system’s critical workflows.
2. List the highest-cost failures.
3. Define observable quality signals.
4. Map each risk to one or more checks.
5. Separate fast checks from deeper validation.
6. Define release-readiness criteria.
7. Record known gaps and deferred risks.

---

## What to Document

### Critical workflows
- main execution path
- high-value data transformations
- external integrations
- registration/persistence steps
- cleanup and retry behavior

### Quality layers
- static analysis
- unit tests
- integration tests
- end-to-end validation
- runtime monitoring
- manual review points

### Release criteria
- what must pass
- what may warn
- what blocks release
- what requires manual sign-off

---

## Review Checklist Template

Check:
- input validation
- failure handling
- logging quality
- side-effect ordering
- correctness of outputs
- schema conformance
- idempotency
- performance regressions
- operational rollback or retry clarity

---

## Output

When using this skill, produce:
- a concise quality playbook
- prioritized test additions
- release confidence criteria
- a short “known risks and gaps” section
