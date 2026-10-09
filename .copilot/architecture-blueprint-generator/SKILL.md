---
name: architecture-blueprint-generator
description: Use this skill when documenting repository architecture, major modules, control flow, dependencies, and design constraints in a developer-friendly blueprint.
---

# Architecture Blueprint Generator

## Purpose

Use this skill to create architecture documentation that helps developers understand how a codebase is structured.

Best fit:
- explain repository architecture
- document module responsibilities
- map dependencies and data flow
- support onboarding
- capture design constraints and extension points

---

## Workflow

1. Identify the main entry points.
2. Map the core modules and their responsibilities.
3. Trace the primary data and control flows.
4. Separate orchestration code from domain logic and utilities.
5. Note external integrations and persistence layers.
6. Document cross-cutting concerns.
7. Summarize constraints and extension points.

---

## What to Include

### Structural view
- entry points
- major packages
- orchestration layer
- domain logic
- integration layer
- QA/validation layer
- config and environment handling

### Behavioral view
- what happens first
- decision points
- side effects
- outputs produced
- error handling patterns

### Constraints
- performance-sensitive paths
- correctness-sensitive outputs
- external protocol contracts
- compatibility or version pinning issues

---

## Output

Return:
- one-page architecture summary
- module map
- primary workflow
- extension points
- key risks or complexity hotspots
