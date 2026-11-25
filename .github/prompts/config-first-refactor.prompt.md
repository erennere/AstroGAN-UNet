---
mode: ask
description: "Refactor code to strict config-driven behavior: remove hardcoded runtime values, eliminate hasattr()/dict.get() fallback patterns for required keys, and wire fields through config ownership and starter resolution."
---

# Config-First Refactor

Refactor the target area to strict config-driven behavior.

## Inputs

- Target files or module (required)
- Behavior to preserve (required)
- Fields to add/propagate (optional)

## Constraints

- No hardcoded runtime values unless explicitly user-approved.
- No hasattr() checks unless explicitly user-approved.
- No dict.get() fallback for required runtime contract keys unless explicitly user-approved.
- No legacy compatibility paths unless explicitly user-approved.

## Steps

1. Map current config ownership and consumers.
2. Define canonical owner section for each runtime field.
3. Wire propagation through config.yaml comments and starter.py resolution/finalizers.
4. Replace consumer fallbacks with strict required-key access for contract fields.
5. Keep non-contract defensive checks only where truly necessary and approved.
6. Add or update focused tests covering propagation and consumer access.
7. Report edits, validation commands, and any remaining approval-required exceptions.

## Deliverable Format

- Short summary of behavior preserved.
- Exact files changed and why.
- List of removed fallback patterns.
- Tests run and outcomes.
- Explicit list of any remaining exception points requiring user confirmation.
