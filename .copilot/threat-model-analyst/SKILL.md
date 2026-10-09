---
name: threat-model-analyst
description: Use this skill when producing a repository or system threat model using assets, trust boundaries, attack surfaces, STRIDE-style categories, and prioritized mitigations.
---

# Threat Model Analyst

## Purpose

Use this skill for system-level security analysis rather than line-by-line code scanning.

Best fit:
- threat model a service or pipeline
- document trust boundaries
- identify abuse cases
- reason about storage, credentials, APIs, and data movement
- prioritize architectural mitigations

---

## Workflow

1. Define the system scope.
2. Identify assets that matter.
3. Draw the main components and data flows.
4. Mark trust boundaries.
5. Enumerate attack surfaces.
6. Evaluate threats using STRIDE-style reasoning.
7. Prioritize by likelihood and impact.
8. Propose mitigations and residual risks.

---

## What to Model

### Assets
- credentials and tokens
- source data
- generated artifacts
- metadata and registrations
- logs and monitoring data
- infrastructure control paths

### Components
- clients and operators
- pipeline entry points
- processors and workers
- file storage
- cloud storage
- databases
- catalogs and APIs
- auth systems

### Trust boundaries
- user to application
- application to cloud services
- internal service-to-service boundaries
- local file system to remote storage
- build environment to runtime environment

---

## Threat Categories

Use at least these lenses:
- spoofing
- tampering
- repudiation
- information disclosure
- denial of service
- privilege escalation
- misuse and abuse scenarios

---

## Output

Produce:
- system summary
- assets
- trust boundaries
- top threats
- prioritized mitigations
- residual risks
