---
name: strict-config-review
description: "Use when reviewing Python changes for strict config-driven compliance, including prohibited hardcoded runtime values, hasattr() usage, dict.get() fallbacks, and unauthorized legacy-compatibility paths."
---

# Strict Config Review

Perform a compliance-oriented code review with findings-first output.

## Review Rules

- Everything runtime-related must be config-driven.
- Hardcoded runtime values require explicit user confirmation.
- hasattr() usage requires explicit user confirmation.
- dict.get() fallback access for required runtime contract keys requires explicit user confirmation.
- Legacy compatibility code paths are disallowed unless explicitly approved.

## Scope Priorities

1. starter.py and config.yaml contract consistency.
2. Stage entrypoints under src/data, src/training, src/evaluation, src/visualization.
3. Derived config dicts passed between modules (kwargs_source, data_kwargs, model_kwargs).

## Review Workflow

1. Find config reads in changed files.
- Flag required-key reads using dict.get() with defaults.
- Flag hidden fallback branches that mask missing config.

2. Find runtime literals.
- Flag literals that should come from config.
- Ignore constants that are purely compile-time/internal and not runtime behavior.

3. Find compatibility shims.
- Flag alternate old/new key paths, dual behavior flags, and soft migration branches.

4. Validate propagation path.
- Confirm key exists in config owner section.
- Confirm resolve/finalizer hop in starter.py.
- Confirm consumer path matches resolved section.

5. Produce findings.
- High: behavior divergence or silent fallback.
- Medium: partial propagation or duplicated ownership.
- Low: maintainability drift.

## Required Report Format

- Findings first, ordered High to Low.
- Each finding includes:
  - file path
  - exact problematic pattern
  - why it violates strict policy
  - concrete fix suggestion
- If no findings: state that explicitly and list residual test gaps.
