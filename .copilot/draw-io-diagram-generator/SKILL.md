---
name: draw-io-diagram-generator
description: Use this skill when creating or updating draw.io diagrams for architecture, workflows, data flow, sequence diagrams, or component relationships.
---

# Draw.io Diagram Generator

## Purpose

Use this skill when the user wants a diagram that can live in the repository as a `.drawio` artifact.

Best fit:
- architecture diagrams
- workflow diagrams
- service interaction diagrams
- data flow diagrams
- deployment views
- sequence-style operational flows

---

## Workflow

1. Determine the diagram goal.
2. Identify the core nodes and relationships.
3. Choose a layout that matches the content.
4. Keep labels short and operationally meaningful.
5. Separate data flow from control flow where needed.
6. Ensure the diagram is readable at normal zoom.
7. Prefer a small number of shapes per level.

---

## Diagram Rules

- One primary message per diagram.
- Keep grouping visually explicit.
- Use consistent naming with the codebase.
- Show trust boundaries or external systems when relevant.
- For workflows, number the main steps.
- For architecture, distinguish orchestrators, processors, storage, and integrations.
- Avoid decorative clutter.

---

## Suggested Output Structure

When producing a diagram specification or file, include:
- title
- nodes
- connectors
- groupings
- notes/legend if needed

---

## Output

Return either:
- a `.drawio` file, or
- a precise diagram specification ready to be turned into one
