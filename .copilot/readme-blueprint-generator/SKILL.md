---
name: readme-blueprint-generator
description: Use this skill when generating or upgrading a repository README with architecture, setup, workflow, commands, outputs, testing, and contributor guidance.
---

# README Blueprint Generator

## Purpose

Use this skill to produce a useful README for developers and operators.

Best fit:
- create a first README
- modernize an outdated README
- align docs with the current repository structure
- document setup, commands, outputs, and testing

---

## Workflow

1. Identify the repository purpose.
2. Extract the main workflows and commands.
3. Document prerequisites and environment expectations.
4. Summarize architecture at a high level.
5. Describe outputs and folder structure.
6. Add development and testing guidance.
7. Add troubleshooting and common failure notes if valuable.

---

## Recommended README Sections

- project overview
- key capabilities
- repository layout
- prerequisites
- configuration
- how to run
- workflow summary
- outputs produced
- testing and linting
- troubleshooting
- contribution notes

---

## Rules

- Optimize for a new engineer joining the project.
- Prefer concrete commands over vague prose.
- Document outputs and side effects clearly.
- Keep examples realistic and current.
- Do not invent commands or env vars.

---

## Output

Return a complete `README.md` draft ready for review.
