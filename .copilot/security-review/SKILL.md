---
name: security-review
description: Use this skill when reviewing code or architecture for security issues, including secrets exposure, injection risks, unsafe file handling, auth mistakes, and insecure integrations.
---

# Security Review

## Purpose

Use this skill for a focused code or system security review.

Best fit:
- audit a repository or module
- review auth and authorization logic
- inspect secrets handling
- trace user input to dangerous sinks
- check command execution, SQL, file access, cloud credentials, and API exposure

---

## Scope

Inspect:
- code paths that accept external input
- filesystem operations
- subprocess execution
- credential loading
- HTTP requests and callbacks
- storage and database access
- access-control decisions
- logging of sensitive data

---

## Workflow

1. Define scope.
2. Find trust boundaries and external inputs.
3. Trace input through validation, transformation, and sinks.
4. Review credential and secret handling.
5. Review file, process, and network operations.
6. Check authorization and tenancy assumptions.
7. Prioritize issues by exploitability and impact.
8. Recommend concrete remediations.

---

## High-Priority Checks

- command injection
- path traversal
- SSRF-like request misuse
- unsafe deserialization
- SQL/NoSQL injection
- insecure temp-file handling
- hardcoded secrets
- overbroad cloud permissions
- weak token validation
- missing authorization checks
- sensitive data in logs
- unsafe archive extraction

---

## Review Rules

- Be evidence-based.
- Distinguish confirmed issue, plausible risk, and hardening suggestion.
- Show the entry point, sink, and missing control.
- Prefer precise remediations over vague advice.
- Note where impact depends on deployment assumptions.

---

## Output

Return a structured report with:
- finding title
- severity
- affected file or component
- exploit path
- why it matters
- recommended fix
