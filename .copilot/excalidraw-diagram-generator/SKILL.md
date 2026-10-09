---
name: excalidraw-diagram-generator
description: Use this skill when generating informal but clear Excalidraw diagrams for workflows, system architecture, mental models, or explanatory visuals.
---

# Excalidraw Diagram Generator

## Purpose

Use this skill when a lighter, presentation-friendly diagram is more useful than a formal draw.io asset.

Best fit:
- quick workflow sketches
- architecture overviews
- concept maps
- explanatory diagrams for discussions
- onboarding visuals

---

## Workflow

1. Clarify the diagram’s single purpose.
2. Pick the main entities and flows.
3. Group related elements into visual regions.
4. Use short labels and minimal text.
5. Keep the reading direction consistent.
6. Add emphasis only where it aids comprehension.
7. Avoid packing too much detail into one canvas.

---

## Diagram Design Rules

- Prefer left-to-right or top-to-bottom flow.
- Keep entity naming aligned with repository terms.
- Use one visual style per diagram.
- Show external systems explicitly.
- Use annotations sparingly for key caveats only.
- Split into multiple diagrams if architecture and workflow are both dense.

---

## Output

Return either:
- an `.excalidraw` artifact, or
- a structured scene specification that can be converted into one
